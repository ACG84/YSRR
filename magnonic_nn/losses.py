"""Objectives and metrics for probe-intensity readouts.

All of these take ``u``, the time-integrated probe intensities, of shape
``(B, n_probes)`` (or ``(n_probes,)`` for a single sample).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .model import normalize_intensities

__all__ = [
    "focus_loss",
    "intensity_cross_entropy",
    "margin_loss",
    "contrast",
    "accuracy",
    "confusion_matrix",
]


def _as_batch(u: torch.Tensor) -> torch.Tensor:
    return u if u.dim() == 2 else u.unsqueeze(0)


def focus_loss(u: torch.Tensor, target: torch.Tensor, eps: float = 1e-30) -> torch.Tensor:
    """Reference implementation's focusing objective.

    ``log10( mean_b [ sum_p u[b,p] / u[b, target[b]] - 1 ] )``

    The ratio is 0 when all the energy lands on the target probe and grows as
    energy leaks elsewhere. Taking the logarithm keeps the gradient useful
    across the several orders of magnitude the ratio spans during training --
    without it the first few epochs dominate and later refinement stalls.

    Scale-free by construction: multiplying every probe by a constant leaves it
    unchanged, so the optimiser cannot cheat by increasing overall throughput.
    """
    u = _as_batch(u)
    target = torch.as_tensor(target, device=u.device).reshape(-1)
    if target.numel() == 1 and u.shape[0] > 1:
        target = target.expand(u.shape[0])

    target_value = u.gather(1, target.reshape(-1, 1).long()).squeeze(1)
    ratio = u.sum(dim=1) / (target_value + eps) - 1.0
    return torch.log10(ratio.mean().clamp_min(eps))


def intensity_cross_entropy(u: torch.Tensor, target: torch.Tensor, eps: float = 1e-30) -> torch.Tensor:
    """Cross-entropy over probe intensities normalised to a distribution.

    Intensities are non-negative and their normalised version already behaves
    like a probability vector, so we take its log directly rather than pushing
    raw intensities through a softmax -- a softmax over unnormalised
    intensities would make the loss depend on absolute drive power.
    """
    u = _as_batch(u)
    target = torch.as_tensor(target, device=u.device).reshape(-1).long()
    p = normalize_intensities(u, eps)
    return F.nll_loss(torch.log(p + eps), target)


def margin_loss(u: torch.Tensor, target: torch.Tensor, margin: float = 0.2, eps: float = 1e-30) -> torch.Tensor:
    """Hinge on the gap between the target probe and the best competitor.

    Stops pushing once the target leads by ``margin`` in normalised intensity,
    which avoids the saturation that cross-entropy drives towards and tends to
    leave more of the design free for the harder samples.
    """
    u = _as_batch(u)
    target = torch.as_tensor(target, device=u.device).reshape(-1).long()
    p = normalize_intensities(u, eps)

    target_p = p.gather(1, target.reshape(-1, 1)).squeeze(1)
    masked = p.clone()
    masked.scatter_(1, target.reshape(-1, 1), -1.0)
    best_other = masked.max(dim=1).values
    return F.relu(margin - (target_p - best_other)).mean()


@torch.no_grad()
def contrast(u: torch.Tensor, target: torch.Tensor, eps: float = 1e-30) -> torch.Tensor:
    """Ratio of target-probe intensity to the strongest other probe, in dB.

    The number to quote for a demultiplexer. The paper reports "more than an
    order of magnitude" spin-wave contrast between its channels, i.e. >10 dB.
    """
    u = _as_batch(u)
    target = torch.as_tensor(target, device=u.device).reshape(-1).long()

    target_value = u.gather(1, target.reshape(-1, 1)).squeeze(1)
    masked = u.clone()
    masked.scatter_(1, target.reshape(-1, 1), -1.0)
    best_other = masked.max(dim=1).values
    return 10.0 * torch.log10((target_value + eps) / (best_other + eps))


@torch.no_grad()
def accuracy(u: torch.Tensor, target: torch.Tensor) -> float:
    """Fraction of samples whose strongest probe is the target probe."""
    u = _as_batch(u)
    target = torch.as_tensor(target, device=u.device).reshape(-1).long()
    return (u.argmax(dim=1) == target).to(torch.get_default_dtype()).mean().item()


@torch.no_grad()
def confusion_matrix(u: torch.Tensor, target: torch.Tensor, n_classes: int | None = None) -> torch.Tensor:
    """``(n_classes, n_classes)`` counts, rows = true class, columns = predicted."""
    u = _as_batch(u)
    target = torch.as_tensor(target, device=u.device).reshape(-1).long()
    n = n_classes or u.shape[1]

    cm = torch.zeros(n, n, dtype=torch.long, device=u.device)
    pred = u.argmax(dim=1)
    for t, p in zip(target.tolist(), pred.tolist()):
        cm[t, p] += 1
    return cm
