"""Training loop.

Nothing here is specific to magnonics -- it is a plain PyTorch loop -- but two
choices are driven by the physics:

**Per-sample gradient accumulation.** magnum.np's field terms have no batch
dimension, so a batch is a loop over samples anyway. Running
``backward()`` inside that loop instead of after it keeps peak memory at one
sample's worth of checkpoints regardless of batch size. Since the loss is a
mean over samples, scaling each sample's contribution by ``1/B`` before its
backward gives exactly the same gradient as the batched version -- but only for
losses that decompose per sample. :func:`focus_loss` does not (its logarithm
sits outside the batch mean), so it is evaluated batched by default.

**Relaxation between steps.** The equilibrium magnetisation depends on the
design, so it must be recomputed after every optimiser step. The model caches
it against the parameter version counter and re-relaxes automatically; you do
not need to do anything, but it is why each epoch costs
``relax_steps`` extra integration steps on top of the rollouts.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path

import torch

__all__ = ["TrainHistory", "train", "evaluate", "accumulate_gradients", "save_checkpoint", "load_checkpoint"]


@dataclass
class TrainHistory:
    """Per-epoch record of a training run."""

    loss: list = field(default_factory=list)
    metrics: dict = field(default_factory=dict)
    epoch_seconds: list = field(default_factory=list)

    def log(self, epoch_loss: float, seconds: float, **metrics):
        self.loss.append(float(epoch_loss))
        self.epoch_seconds.append(float(seconds))
        for k, v in metrics.items():
            self.metrics.setdefault(k, []).append(float(v))

    def to_json(self, path):
        Path(path).write_text(
            json.dumps(
                {"loss": self.loss, "metrics": self.metrics, "epoch_seconds": self.epoch_seconds},
                indent=2,
            )
        )

    @property
    def best_epoch(self):
        return int(min(range(len(self.loss)), key=lambda i: self.loss[i])) if self.loss else None


def accumulate_gradients(model, signals, targets, loss_fn) -> float:
    """Run a batch one sample at a time, backpropagating each immediately.

    :returns: the mean loss over the batch, as a float.

    Peak memory is that of a single sample. Only valid for losses that are a
    mean of per-sample terms.
    """
    total = 0.0
    n = signals.shape[0]
    for i in range(n):
        u = model(signals[i])
        loss = loss_fn(u.unsqueeze(0), targets[i : i + 1]) / n
        loss.backward()
        total += float(loss.detach()) * n
    return total / n


def train(
    model,
    signals: torch.Tensor,
    targets: torch.Tensor,
    loss_fn,
    epochs: int = 10,
    lr: float = 0.01,
    optimizer=None,
    per_sample: bool = False,
    metric_fns: dict | None = None,
    on_epoch=None,
    history: TrainHistory | None = None,
    grad_clip: float | None = None,
    best_path=None,
    monitor: tuple = ("loss", "min"),
    batch_size: int | None = None,
    steps_per_epoch: int | None = None,
    verbose: bool = True,
) -> TrainHistory:
    """Optimise the scatterer design.

    :param signals: ``(N, T, n_sources)`` drive waveforms.
    :param targets: ``(N,)`` target probe indices / class labels.
    :param loss_fn: callable ``(u, targets) -> scalar``.
    :param per_sample: use :func:`accumulate_gradients` instead of a batched
        forward. Slower per epoch in Python overhead, but flat in memory.
        Required once a batch of full-size rollouts no longer fits.
    :param metric_fns: extra ``name -> fn(u, targets)`` evaluated (no grad) each
        epoch.
    :param on_epoch: callback ``(epoch, model, u, loss, history)`` invoked after
        every epoch -- use it for plotting and checkpointing.
    :param grad_clip: clip the gradient norm of the design parameters. The
        gradient of a wave rollout can be badly scaled early on, when the design
        is near-uniform and small changes swing the interference pattern a long
        way; clipping stops a single outlier step from destroying the design.
    :param best_path: if given, write a checkpoint whenever ``monitor`` improves.
        Worth setting on any real run: these trajectories are not monotone, and
        saving only the last epoch routinely throws away a better design than
        the one you keep.
    :param monitor: ``(name, mode)`` selecting what ``best_path`` tracks.
        ``name`` is ``"loss"`` or any key in ``metric_fns``; ``mode`` is
        ``"min"`` or ``"max"``.

        Prefer a continuous quantity. A coarse metric that saturates -- accuracy
        over three inputs takes four possible values -- reaches its ceiling long
        before the design stops improving, and since ties do not update, the
        "best" checkpoint freezes on the first epoch to hit the ceiling. On the
        demultiplexer run that would have kept a 7.9 dB design in preference to
        the 16.2 dB one the same run went on to find.
    :param batch_size: draw this many samples per optimiser step instead of
        using the whole training set. ``None`` keeps full-batch descent.

        This is what makes a realistic training set affordable. Every sample in
        a step costs a full micromagnetic rollout plus its adjoint -- measured
        at ~37 s per token per epoch for a 48x48 mesh over 1000 steps -- so
        full-batch cost is strictly linear in dataset size. Going from 9 tokens
        to a more honest 90 turns a 1.8-hour run into an 18-hour one. Drawing a
        fixed-size minibatch instead makes the cost per step independent of how
        large the pool is, so the design still sees the whole distribution,
        just spread across steps rather than crammed into each one.
    :param steps_per_epoch: optimiser steps per epoch. Defaults to one full
        pass, ``ceil(N / batch_size)``. Set it to 1 to hold the cost of an
        epoch fixed no matter how big the training set grows.

        Samples are drawn by shuffling the whole set and consuming it, then
        reshuffling -- not by independent draws -- so every token is seen
        equally often rather than by luck.
    """
    optimizer = optimizer or torch.optim.Adam(model.parameters(), lr=lr)
    history = history or TrainHistory()
    metric_fns = metric_fns or {}

    monitor_name, monitor_mode = monitor
    better = (lambda a, b: a < b) if monitor_mode == "min" else (lambda a, b: a > b)
    best_value = float("inf") if monitor_mode == "min" else float("-inf")

    # Epochs are numbered globally, continuing whatever history was passed in.
    # Using the local loop index instead makes every resumed run restart its
    # count at zero, so a checkpoint written after a resume claims to be epoch 0
    # and anything reading that field -- reports, renders, comparisons between
    # runs -- silently mislabels it.
    start_epoch = len(history.loss)

    n_samples = signals.shape[0]
    minibatch = batch_size is not None and batch_size < n_samples
    if minibatch:
        n_steps = steps_per_epoch or math.ceil(n_samples / batch_size)
    else:
        n_steps = steps_per_epoch or 1
    pool: list = []

    def next_indices():
        """Shuffle-and-consume, so coverage is even rather than luck-dependent."""
        nonlocal pool
        if not minibatch:
            return torch.arange(n_samples, device=targets.device)
        picked = []
        while len(picked) < batch_size:
            if not pool:
                pool = torch.randperm(n_samples).tolist()
            picked.append(pool.pop())
        return torch.tensor(picked, device=targets.device)

    for local_epoch in range(epochs):
        epoch = start_epoch + local_epoch
        t_start = time.time()

        step_losses, seen_u, seen_t, grad_norm = [], [], [], 0.0
        for _ in range(n_steps):
            idx = next_indices()
            sig_b, tgt_b = signals[idx], targets[idx]

            optimizer.zero_grad(set_to_none=True)
            if per_sample:
                step_loss = accumulate_gradients(model, sig_b, tgt_b, loss_fn)
                with torch.no_grad():
                    u = model(sig_b)
            else:
                u = model(sig_b)
                loss = loss_fn(u, tgt_b)
                loss.backward()
                step_loss = float(loss.detach())

            # Record the gradient norm before any clipping. Without it there is
            # no way to tell a run that is over-stepping from one whose
            # gradients have collapsed -- both look like a loss that stops
            # improving.
            grad_norm = float(
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), grad_clip if grad_clip is not None else float("inf")
                )
            )
            optimizer.step()

            step_losses.append(step_loss)
            seen_u.append(u.detach())
            seen_t.append(tgt_b)

        # Metrics are over the samples this epoch actually visited. With
        # minibatching that is a sample of the training set, not all of it, so
        # the logged numbers are noisier than a full-batch run's -- evaluate on
        # the held-out set for a figure to quote.
        u = torch.cat(seen_u)
        targets_seen = torch.cat(seen_t)
        loss_value = sum(step_losses) / len(step_losses)

        with torch.no_grad():
            metrics = {name: float(fn(u, targets_seen)) for name, fn in metric_fns.items()}
        metrics["grad_norm"] = grad_norm

        elapsed = time.time() - t_start
        history.log(loss_value, elapsed, **metrics)

        current = loss_value if monitor_name == "loss" else metrics.get(monitor_name)
        is_best = current is not None and better(current, best_value)
        if is_best:
            best_value = current
            if best_path is not None:
                save_checkpoint(best_path, model, optimizer, history, epoch,
                                extra={"monitor": monitor_name, "monitor_value": current})

        if verbose:
            extra = "".join(f"  {k}={v:.4f}" for k, v in metrics.items())
            star = "  *best" if is_best else ""
            print(f"epoch {epoch:3d}  loss={loss_value:.6f}{extra}  ({elapsed:.1f}s){star}",
                  flush=True)

        if on_epoch is not None:
            on_epoch(epoch, model, u.detach(), loss_value, history)

    return history


@torch.no_grad()
def evaluate(model, signals: torch.Tensor, targets: torch.Tensor, metric_fns: dict) -> dict:
    """Evaluate metrics on a held-out set."""
    u = model(signals)
    return {name: float(fn(u, targets)) for name, fn in metric_fns.items()}


def save_checkpoint(path, model, optimizer=None, history=None, epoch=None, extra=None):
    """Write model, optimiser and history to ``path``."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_state_dict": model.state_dict(),
        "epoch": epoch,
        "config": _config_to_dict(model.cfg),
    }
    if optimizer is not None:
        payload["optimizer_state_dict"] = optimizer.state_dict()
    if history is not None:
        payload["history"] = {
            "loss": history.loss,
            "metrics": history.metrics,
            "epoch_seconds": history.epoch_seconds,
        }
    if extra:
        payload.update(extra)
    torch.save(payload, path)
    return path


def load_checkpoint(path, model, optimizer=None, lr=None):
    """Restore from :func:`save_checkpoint`. Returns ``(epoch, history)``.

    :param lr: learning rate to force after restoring the optimiser.
        ``Optimizer.load_state_dict`` restores ``param_groups`` wholesale,
        including the learning rate the checkpoint was written with -- so
        resuming a run specifically to lower the step size silently keeps the
        old one unless this is passed.
    """
    payload = torch.load(path, map_location=next(model.parameters()).device, weights_only=False)
    model.load_state_dict(payload["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in payload:
        optimizer.load_state_dict(payload["optimizer_state_dict"])
        if lr is not None:
            for group in optimizer.param_groups:
                group["lr"] = lr

    history = TrainHistory()
    if "history" in payload:
        history.loss = payload["history"].get("loss", [])
        history.metrics = payload["history"].get("metrics", {})
        history.epoch_seconds = payload["history"].get("epoch_seconds", [])
    return payload.get("epoch"), history


def _config_to_dict(cfg):
    """Flatten a :class:`SimConfig` into something ``torch.save`` round-trips cleanly."""
    from dataclasses import asdict, is_dataclass

    return asdict(cfg) if is_dataclass(cfg) else {}
