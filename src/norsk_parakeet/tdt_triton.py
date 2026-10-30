"""Triton kernels for the TDT forward (alpha) and backward (beta) recursions.

The PyTorch versions in loss.py loop over anti-diagonals in Python, launching a dozen small
kernels per diagonal; for 30 s clips that is ~500 diagonals per pass and most of a training
step. Here one program per utterance walks all diagonals on the GPU. Each diagonal is a vector
over u; the next diagonal reads what this one stored, so programs synchronize with a barrier
between diagonals (stores to global memory are visible within a block after the barrier).

Durations must be 0..D-1 (Parakeet: [0, 1, 2, 3, 4]). Values outside an utterance's lattice
(t >= T_b or u > U_b) are -inf; the torch backward in loss.py only reads inside it.
"""

import torch
import triton
import triton.language as tl

NEG_INF = tl.constexpr(float("-inf"))


@triton.jit
def _lse_add(m, s, c):
    """Online logsumexp: fold candidate c into running max m and scaled sum s."""
    m_new = tl.maximum(m, c)
    safe = m_new > NEG_INF
    s = tl.where(safe, s * tl.exp(m - m_new) + tl.exp(c - m_new), s)
    return m_new, s


@triton.jit
def _alpha_kernel(alpha, blank, label, dur, t_len, u_len,
                  s_ab, s_at, s_lb, s_lt, s_db, s_dt, s_du,
                  D: tl.constexpr, BLOCK_U: tl.constexpr):
    b = tl.program_id(0)
    T = tl.load(t_len + b)
    U = tl.load(u_len + b)
    u = tl.arange(0, BLOCK_U)
    a_base = alpha + b * s_ab
    bl_base = blank + b * s_ab
    lb_base = label + b * s_lb
    d_base = dur + b * s_db
    tl.store(a_base, 0.0)
    tl.debug_barrier()
    for n in range(1, T + U):
        t = n - u
        live = (u <= U) & (t >= 0) & (t < T)
        m = tl.full([BLOCK_U], NEG_INF, tl.float32)
        s = tl.zeros([BLOCK_U], tl.float32)
        for d in tl.static_range(D):
            ts = t - d
            ok = live & (ts >= 0)
            if d > 0:
                c = (tl.load(a_base + ts * s_at + u, mask=ok, other=NEG_INF)
                     + tl.load(bl_base + ts * s_at + u, mask=ok, other=0.0)
                     + tl.load(d_base + ts * s_dt + u * s_du + d, mask=ok, other=0.0))
                m, s = _lse_add(m, s, tl.where(ok, c, NEG_INF))
            okl = ok & (u >= 1)
            c = (tl.load(a_base + ts * s_at + (u - 1), mask=okl, other=NEG_INF)
                 + tl.load(lb_base + ts * s_lt + (u - 1), mask=okl, other=0.0)
                 + tl.load(d_base + ts * s_dt + (u - 1) * s_du + d, mask=okl, other=0.0))
            m, s = _lse_add(m, s, tl.where(okl, c, NEG_INF))
        val = tl.where(m > NEG_INF, m + tl.log(s), NEG_INF)
        tl.store(a_base + t * s_at + u, val, mask=live)
        tl.debug_barrier()


@triton.jit
def _beta_kernel(beta, blank, label, dur, t_len, u_len,
                 s_bb, s_bt, s_ab, s_at, s_lb, s_lt, s_db, s_dt, s_du,
                 D: tl.constexpr, BLOCK_U: tl.constexpr):
    b = tl.program_id(0)
    T = tl.load(t_len + b)
    U = tl.load(u_len + b)
    u = tl.arange(0, BLOCK_U)
    be_base = beta + b * s_bb
    bl_base = blank + b * s_ab
    lb_base = label + b * s_lb
    d_base = dur + b * s_db
    for k in range(0, T + U):
        n = T - 1 + U - k
        t = n - u
        live = (u <= U) & (t >= 0) & (t < T)
        m = tl.full([BLOCK_U], NEG_INF, tl.float32)
        s = tl.zeros([BLOCK_U], tl.float32)
        for d in tl.static_range(D):
            tn = t + d
            if d > 0:
                final = (tn == T) & (u == U)
                ok = live & ((tn < T) | final)
                nxt = tl.load(be_base + tn * s_bt + u, mask=live & (tn < T), other=NEG_INF)
                nxt = tl.where(final, 0.0, nxt)
                c = (tl.load(bl_base + t * s_at + u, mask=live, other=0.0)
                     + tl.load(d_base + t * s_dt + u * s_du + d, mask=live, other=0.0) + nxt)
                m, s = _lse_add(m, s, tl.where(ok, c, NEG_INF))
            okl = live & (u < U) & (tn < T)
            c = (tl.load(lb_base + t * s_lt + u, mask=okl, other=0.0)
                 + tl.load(d_base + t * s_dt + u * s_du + d, mask=okl, other=0.0)
                 + tl.load(be_base + tn * s_bt + (u + 1), mask=okl, other=NEG_INF))
            m, s = _lse_add(m, s, tl.where(okl, c, NEG_INF))
        val = tl.where(m > NEG_INF, m + tl.log(s), NEG_INF)
        tl.store(be_base + t * s_bt + u, val, mask=live)
        tl.debug_barrier()


def _check(blank_lp, label_lp, dur_lp, durations):
    assert list(durations) == list(range(len(durations))), "kernels assume durations 0..D-1"
    assert blank_lp.dtype == torch.float32 and blank_lp.is_cuda
    b, t, u1 = blank_lp.shape
    assert label_lp.shape == (b, t, u1 - 1) and dur_lp.shape == (b, t, u1, len(durations))
    return b, t, u1


def alpha_triton(blank_lp, label_lp, dur_lp, t_len, u_len, durations):
    b, t, u1 = _check(blank_lp, label_lp, dur_lp, durations)
    blank_lp, label_lp, dur_lp = blank_lp.contiguous(), label_lp.contiguous(), dur_lp.contiguous()
    alpha = torch.full((b, t, u1), float("-inf"), device=blank_lp.device, dtype=torch.float32)
    block = max(16, triton.next_power_of_2(u1))
    _alpha_kernel[(b,)](alpha, blank_lp, label_lp, dur_lp, t_len.int().contiguous(), u_len.int().contiguous(),
                        alpha.stride(0), alpha.stride(1), label_lp.stride(0), label_lp.stride(1),
                        dur_lp.stride(0), dur_lp.stride(1), dur_lp.stride(2),
                        D=len(durations), BLOCK_U=block, num_warps=4 if block <= 256 else 8)
    return alpha


def beta_triton(blank_lp, label_lp, dur_lp, t_len, u_len, durations):
    """Padded by max(durations) + 1 frames along t, like loss._beta."""
    b, t, u1 = _check(blank_lp, label_lp, dur_lp, durations)
    blank_lp, label_lp, dur_lp = blank_lp.contiguous(), label_lp.contiguous(), dur_lp.contiguous()
    beta = torch.full((b, t + max(durations) + 1, u1), float("-inf"), device=blank_lp.device, dtype=torch.float32)
    block = max(16, triton.next_power_of_2(u1))
    _beta_kernel[(b,)](beta, blank_lp, label_lp, dur_lp, t_len.int().contiguous(), u_len.int().contiguous(),
                       beta.stride(0), beta.stride(1), blank_lp.stride(0), blank_lp.stride(1),
                       label_lp.stride(0), label_lp.stride(1), dur_lp.stride(0), dur_lp.stride(1), dur_lp.stride(2),
                       D=len(durations), BLOCK_U=block, num_warps=4 if block <= 256 else 8)
    return beta
