"""Memory-lean TDT loss for ParakeetForTDT.

transformers' tdt_loss takes the full joint output [B, T, U+1, V+D] and keeps float32 copies
of it for log_softmax and backward, which runs out of memory on a 24 GB card at useful batch
sizes. The DP only needs three [B, T, U+1] tensors: the blank log-prob, the next label's
log-prob and the duration log-probs. This computes the joint in time chunks under activation
checkpointing and keeps only those, then runs the same anti-diagonal DP as transformers.
"""

import torch
from torch.utils.checkpoint import checkpoint


def _chunk_terms(joint, enc, dec, labels, vocab, blank):
    # enc [B, Tc, H], dec [B, U+1, H], labels [B, U]
    logits = joint(encoder_hidden_states=enc[:, :, None, :], decoder_hidden_states=dec[:, None, :, :]).float()
    tok, dur = logits[..., :vocab], logits[..., vocab:]
    lse = torch.logsumexp(tok, dim=-1)  # [B, Tc, U+1]
    blank_lp = tok[..., blank] - lse
    b, tc, u1, _ = tok.shape
    idx = labels[:, None, :, None].expand(b, tc, u1 - 1, 1)
    label_lp = torch.gather(tok[:, :, :-1, :], 3, idx).squeeze(-1) - lse[:, :, :-1]
    return blank_lp, label_lp, torch.log_softmax(dur, dim=-1)


def tdt_terms(model, enc, dec, labels, chunk=24):
    cfg = model.config
    outs = [checkpoint(_chunk_terms, model.joint, enc[:, s:s + chunk], dec, labels, cfg.vocab_size,
                       cfg.blank_token_id, use_reentrant=False)
            for s in range(0, enc.shape[1], chunk)]
    return [torch.cat(x, dim=1) for x in zip(*outs)]


def tdt_dp(blank_lp, label_lp, dur_lp, logit_lengths, target_lengths, durations):
    """Per-utterance negative log-likelihood; the recursion of transformers.loss.loss_tdt.tdt_loss."""
    device = blank_lp.device
    batch, max_t, max_u = blank_lp.shape
    log_alpha = torch.full((batch, max_t, max_u), float("-inf"), device=device, dtype=blank_lp.dtype)
    log_alpha[:, 0, 0] = 0.0
    neg_inf = torch.tensor(float("-inf"), device=device, dtype=blank_lp.dtype)
    for n in range(1, max_t + max_u - 1):
        u_idx = torch.arange(max(0, n - max_t + 1), min(n + 1, max_u), device=device)
        t_idx = n - u_idx
        cands = []
        for i, d in enumerate(durations):
            t_prev = t_idx - d
            valid_t = t_prev >= 0
            if not valid_t.any():
                continue
            t_src = t_prev.clamp(min=0)
            if d > 0:
                c = log_alpha[:, t_src, u_idx] + blank_lp[:, t_src, u_idx] + dur_lp[:, t_src, u_idx, i]
                cands.append(torch.where(valid_t[None], c, neg_inf))
            valid = valid_t & (u_idx > 0)
            if valid.any() and max_u > 1:
                u_src = (u_idx - 1).clamp(min=0)
                c = (log_alpha[:, t_src, u_src] + label_lp[:, t_src, u_src.clamp(max=max_u - 2)]
                     + dur_lp[:, t_src, u_src, i])
                cands.append(torch.where(valid[None], c, neg_inf))
        if cands:
            log_alpha[:, t_idx, u_idx] = torch.logsumexp(torch.stack(cands), dim=0)
    b_idx = torch.arange(batch, device=device)
    total = torch.full((batch,), float("-inf"), device=device, dtype=blank_lp.dtype)
    for i, d in enumerate(durations):
        if d == 0:
            continue
        t_fin = logit_lengths - d
        ok = t_fin >= 0
        t_c = t_fin.clamp(min=0)
        term = (log_alpha[b_idx, t_c, target_lengths] + blank_lp[b_idx, t_c, target_lengths]
                + dur_lp[b_idx, t_c, target_lengths, i])
        total = torch.where(ok, torch.logsumexp(torch.stack([total, term]), dim=0), total)
    return -total


def tdt_loss(model, input_features, attention_mask, labels, decoder_input_ids, chunk=24, max_token_loss=None):
    """Mean over the batch of per-utterance loss / target length (the checkpoints' "mean" reduction).

    With max_token_loss, utterances whose loss per token is above it (or infinite) are left out
    of the mean: transcripts that do not match their audio give losses in the tens to thousands,
    against ~1 for real ones, and would otherwise steer a whole update. Returns (loss, number
    left out); loss is None when every utterance is left out.
    """
    enc = model.get_audio_features(input_features=input_features, attention_mask=attention_mask)
    dec = model.decoder(decoder_input_ids)
    pad = model.config.pad_token_id
    target_lengths = (labels != pad).sum(-1)
    labels = labels.masked_fill(labels == pad, 0)
    blank_lp, label_lp, dur_lp = tdt_terms(model, enc.pooler_output, dec, labels, chunk)
    nll = tdt_nll(blank_lp, label_lp, dur_lp, enc.attention_mask.sum(-1), target_lengths, model.config.durations)
    per_token = nll / target_lengths.float()
    if max_token_loss is None:
        return per_token.mean(), 0
    keep = torch.isfinite(per_token.detach()) & (per_token.detach() <= max_token_loss)
    dropped = int((~keep).sum())
    if not keep.any():
        return None, dropped
    return per_token[keep].mean(), dropped


class _TDTForwardBackward(torch.autograd.Function):
    """tdt_dp with an alpha-beta backward instead of autograd through the recursion.

    The gradient of -log Z with respect to an arc's log-weight is minus the arc's posterior,
    exp(alpha[src] + weight + beta[dst] - log Z), so the backward is a handful of whole-tensor
    ops after one extra pass for beta, instead of autograd replaying thousands of small ones.
    """

    @staticmethod
    def forward(ctx, blank_lp, label_lp, dur_lp, logit_lengths, target_lengths, durations):
        with torch.no_grad():
            if blank_lp.is_cuda:
                from .tdt_triton import alpha_triton, beta_triton

                alpha = alpha_triton(blank_lp, label_lp, dur_lp, logit_lengths, target_lengths, durations)
                beta = beta_triton(blank_lp, label_lp, dur_lp, logit_lengths, target_lengths, durations)
            else:
                alpha = _alpha(blank_lp, label_lp, dur_lp, durations)
                beta = _beta(blank_lp, label_lp, dur_lp, logit_lengths, target_lengths, durations)
            nll = -beta[:, 0, 0]
        ctx.save_for_backward(blank_lp, label_lp, dur_lp, logit_lengths, target_lengths, alpha, beta, -nll)
        ctx.durations = durations
        return nll

    @staticmethod
    def backward(ctx, grad_nll):
        blank_lp, label_lp, dur_lp, t_len, u_len, alpha, beta, log_z = ctx.saved_tensors
        batch, max_t, max_u = blank_lp.shape
        ok = torch.isfinite(log_z)
        scale = torch.where(ok, grad_nll, torch.zeros_like(grad_nll))[:, None, None]
        log_z = torch.where(ok, log_z, torch.zeros_like(log_z))[:, None, None]
        g_blank = torch.zeros_like(blank_lp)
        g_label = torch.zeros_like(label_lp)
        g_dur = torch.zeros_like(dur_lp)
        b_idx = torch.arange(batch, device=blank_lp.device)
        for i, d in enumerate(ctx.durations):
            if d > 0:
                # The final blank lands on (T_b, U_b), which beta keeps at -inf; its weight is 0.
                nxt = beta[:, d:d + max_t, :max_u].clone()
                t_fin = t_len - d
                ok_fin = t_fin >= 0
                nxt[b_idx[ok_fin], t_fin[ok_fin], u_len[ok_fin]] = 0.0
                post = torch.exp(alpha + blank_lp + dur_lp[..., i] + nxt - log_z)
                post = torch.nan_to_num(post) * scale
                g_blank -= post
                g_dur[..., i] -= post
            if max_u > 1:
                post = torch.exp(alpha[:, :, :-1] + label_lp + dur_lp[:, :, :-1, i]
                                 + beta[:, d:d + max_t, 1:max_u] - log_z)
                post = torch.nan_to_num(post) * scale
                g_label -= post
                g_dur[:, :, :-1, i] -= post
        return g_blank, g_label, g_dur, None, None, None


def _alpha(blank_lp, label_lp, dur_lp, durations):
    """Forward variables of tdt_dp (same recursion, kept as a tensor)."""
    device = blank_lp.device
    batch, max_t, max_u = blank_lp.shape
    alpha = torch.full((batch, max_t, max_u), float("-inf"), device=device, dtype=blank_lp.dtype)
    alpha[:, 0, 0] = 0.0
    for n in range(1, max_t + max_u - 1):
        u_idx = torch.arange(max(0, n - max_t + 1), min(n + 1, max_u), device=device)
        t_idx = n - u_idx
        cands = []
        for i, d in enumerate(durations):
            t_src = t_idx - d
            ok_t = t_src >= 0
            if not ok_t.any():
                continue
            t_src = t_src.clamp(min=0)
            if d > 0:
                cands.append(torch.where(ok_t[None], alpha[:, t_src, u_idx] + blank_lp[:, t_src, u_idx]
                                         + dur_lp[:, t_src, u_idx, i], float("-inf")))
            ok = ok_t & (u_idx > 0)
            if ok.any() and max_u > 1:
                u_src = (u_idx - 1).clamp(min=0)
                cands.append(torch.where(ok[None], alpha[:, t_src, u_src] + label_lp[:, t_src, u_src.clamp(max=max_u - 2)]
                                         + dur_lp[:, t_src, u_src, i], float("-inf")))
        alpha[:, t_idx, u_idx] = torch.logsumexp(torch.stack(cands), dim=0)
    return alpha


def _beta(blank_lp, label_lp, dur_lp, t_len, u_len, durations):
    """Backward variables, padded by max(durations) frames: beta[b, t, u] for t < max_t + max_d.

    A node is live when t < T_b and u <= U_b. The final transition is a blank of duration d from
    (T_b - d, U_b), landing exactly on T_b, as in tdt_dp.
    """
    device = blank_lp.device
    batch, max_t, max_u = blank_lp.shape
    max_d = max(durations)
    beta = torch.full((batch, max_t + max_d + 1, max_u), float("-inf"), device=device, dtype=blank_lp.dtype)
    t_ar = torch.arange(max_t, device=device)
    u_ar = torch.arange(max_u, device=device)
    live = (t_ar[None, :, None] < t_len[:, None, None]) & (u_ar[None, None, :] <= u_len[:, None, None])
    for n in range(max_t + max_u - 2, -1, -1):
        u_idx = torch.arange(max(0, n - max_t + 1), min(n + 1, max_u), device=device)
        t_idx = n - u_idx
        cands = []
        for i, d in enumerate(durations):
            if d > 0:
                nxt = beta[:, t_idx + d, u_idx]
                final = ((t_idx + d)[None] == t_len[:, None]) & (u_idx[None] == u_len[:, None])
                nxt = torch.where(final, torch.zeros_like(nxt), nxt)
                cands.append(blank_lp[:, t_idx, u_idx] + dur_lp[:, t_idx, u_idx, i] + nxt)
            if max_u > 1:
                has_next = u_idx < max_u - 1
                u_lab = u_idx.clamp(max=max_u - 2)
                nxt = beta[:, t_idx + d, (u_idx + 1).clamp(max=max_u - 1)]
                ok = has_next[None] & (u_idx[None] < u_len[:, None])
                cands.append(torch.where(ok, label_lp[:, t_idx, u_lab] + dur_lp[:, t_idx, u_idx, i] + nxt,
                                         float("-inf")))
        val = torch.logsumexp(torch.stack(cands), dim=0)
        beta[:, t_idx, u_idx] = torch.where(live[:, t_idx, u_idx], val, float("-inf"))
    return beta


def tdt_nll(blank_lp, label_lp, dur_lp, logit_lengths, target_lengths, durations):
    return _TDTForwardBackward.apply(blank_lp, label_lp, dur_lp, logit_lengths, target_lengths, list(durations))
