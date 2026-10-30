"""The lean TDT loss must match transformers' formulation: same recursion, same gradients."""

import pytest
import torch

from norsk_parakeet.loss import tdt_dp, tdt_nll

DURATIONS = [0, 1, 2, 3, 4]


def _terms(tok, dur, labels):
    b, t, u1, v = tok.shape
    lp = tok.log_softmax(-1)
    label_lp = torch.gather(lp[:, :, :-1], 3, labels[:, None, :, None].expand(b, t, u1 - 1, 1)).squeeze(-1)
    return lp[..., v - 1], label_lp, dur.log_softmax(-1)


def test_alpha_beta_backward_matches_autograd():
    torch.manual_seed(0)
    b, t, u1, v = 3, 23, 9, 11
    tok = torch.randn(b, t, u1, v, dtype=torch.float64)
    dur = torch.randn(b, t, u1, len(DURATIONS), dtype=torch.float64)
    labels = torch.randint(0, v - 1, (b, u1 - 1))
    t_len, u_len = torch.tensor([23, 17, 9]), torch.tensor([8, 5, 2])
    weights = torch.tensor([1.0, 2.0, 0.5], dtype=torch.float64)
    grads = []
    for fn in (tdt_dp, tdt_nll):
        a, d = tok.clone().requires_grad_(), dur.clone().requires_grad_()
        nll = fn(*_terms(a, d, labels), t_len, u_len, DURATIONS)
        (nll * weights).sum().backward()
        grads.append((nll.detach(), a.grad, d.grad))
    for ref, got in zip(grads[0], grads[1]):
        torch.testing.assert_close(got, ref, rtol=0, atol=1e-10)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Triton kernels need a GPU")
def test_triton_recursions_match_torch():
    from norsk_parakeet.loss import _alpha, _beta
    from norsk_parakeet.tdt_triton import alpha_triton, beta_triton

    torch.manual_seed(0)
    b, t, u1 = 3, 60, 21
    blank = torch.randn(b, t, u1, device="cuda").log_softmax(-1)
    label = torch.randn(b, t, u1 - 1, device="cuda").log_softmax(-1)
    dur = torch.randn(b, t, u1, len(DURATIONS), device="cuda").log_softmax(-1)
    t_len, u_len = torch.tensor([60, 44, 20], device="cuda"), torch.tensor([20, 11, 4], device="cuda")
    live = ((torch.arange(t, device="cuda")[None, :, None] < t_len[:, None, None])
            & (torch.arange(u1, device="cuda")[None, None, :] <= u_len[:, None, None]))
    for got, ref in ((alpha_triton(blank, label, dur, t_len, u_len, DURATIONS), _alpha(blank, label, dur, DURATIONS)),
                     (beta_triton(blank, label, dur, t_len, u_len, DURATIONS)[:, :t],
                      _beta(blank, label, dur, t_len, u_len, DURATIONS)[:, :t])):
        ref = torch.where(live, ref, float("-inf"))
        got = torch.where(live, got, float("-inf"))
        assert torch.equal(torch.isfinite(got), torch.isfinite(ref))
        fin = torch.isfinite(ref)
        torch.testing.assert_close(got[fin], ref[fin], rtol=1e-5, atol=1e-4)
