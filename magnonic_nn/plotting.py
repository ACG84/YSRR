"""Figures.

Every function takes an optional ``path``; if given the figure is written there
and closed, otherwise the ``matplotlib`` figure is returned for further use.
Images are drawn with ``origin='lower'`` and the array transposed, so x runs
right and y runs up -- matching the physical layout rather than array index
order.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

__all__ = [
    "plot_design",
    "plot_snapshot",
    "plot_integrated_intensity",
    "plot_loss",
    "plot_probe_outputs",
    "plot_confusion_matrix",
    "plot_spectrum",
]


def _finish(fig, path):
    if path is None:
        return fig
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def _to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _extent(mesh_cfg):
    nx, ny, _ = mesh_cfg.n
    return [0, nx * mesh_cfg.dx * 1e6, 0, ny * mesh_cfg.dy * 1e6]


def _overlay(ax, model, mesh_cfg):
    """Mark source and probe locations on a field image."""
    sx, sy = mesh_cfg.dx * 1e6, mesh_cfg.dy * 1e6
    for src in model.sources:
        xs, ys = src.coordinates()
        ax.plot(_to_numpy(xs) * sx, _to_numpy(ys) * sy, ".", color="tab:red", ms=1.5, alpha=0.8)
    for probe in model.probes:
        xs, ys = probe.coordinates()
        ax.plot(_to_numpy(xs) * sx, _to_numpy(ys) * sy, ".", color="white", ms=1.5, alpha=0.9)


def plot_design(model, path=None, title=None):
    """The trained scatterer: static field along the bias axis, in mT."""
    mesh_cfg = model.cfg.mesh
    image = _to_numpy(model.design_field_tesla()) * 1e3

    fig, ax = plt.subplots(figsize=(5.2, 4.4))
    vmin, vmax = image.min(), image.max()
    im = ax.imshow(image.T, origin="lower", extent=_extent(mesh_cfg), cmap="RdBu_r", vmin=vmin, vmax=vmax)
    _overlay(ax, model, mesh_cfg)
    ax.set_xlabel("x (um)")
    ax.set_ylabel("y (um)")
    ax.set_title(title or "Scatterer design (bias-axis field)")
    fig.colorbar(im, ax=ax, label="B (mT)")
    return _finish(fig, path)


def plot_snapshot(model, m, path=None, component: int = 2, m0=None, title=None):
    """Instantaneous wave field, i.e. one magnetisation component minus equilibrium."""
    mesh_cfg = model.cfg.mesh
    m = _to_numpy(m)
    if m0 is not None:
        m = m - _to_numpy(m0)
    image = m[:, :, 0, component]

    scale = np.abs(image).max() or 1.0
    fig, ax = plt.subplots(figsize=(5.2, 4.4))
    im = ax.imshow(
        image.T, origin="lower", extent=_extent(mesh_cfg), cmap="RdBu_r", vmin=-scale, vmax=scale
    )
    _overlay(ax, model, mesh_cfg)
    ax.set_xlabel("x (um)")
    ax.set_ylabel("y (um)")
    ax.set_title(title or f"Wave field, m[{'xyz'[component]}]")
    fig.colorbar(im, ax=ax, label="dm")
    return _finish(fig, path)


def plot_integrated_intensity(model, snapshots, path=None, component: int = 2, m0=None, title=None):
    """Time-integrated ``|dm|^2``: where the energy actually goes.

    This is the figure that shows whether the design is doing its job -- a
    trained focuser puts a visible bright spot on the target probe.
    """
    mesh_cfg = model.cfg.mesh
    snaps = _to_numpy(snapshots)
    if m0 is not None:
        snaps = snaps - _to_numpy(m0)[None]
    image = (snaps[:, :, :, 0, component] ** 2).sum(axis=0)

    fig, ax = plt.subplots(figsize=(5.2, 4.4))
    im = ax.imshow(image.T, origin="lower", extent=_extent(mesh_cfg), cmap="inferno")
    _overlay(ax, model, mesh_cfg)
    ax.set_xlabel("x (um)")
    ax.set_ylabel("y (um)")
    ax.set_title(title or "Time-integrated intensity")
    fig.colorbar(im, ax=ax, label="sum |dm|^2")
    return _finish(fig, path)


def plot_loss(history, path=None, metrics=("accuracy",)):
    """Convergence: loss and each recorded metric, stacked on a shared epoch axis.

    Deliberately *not* a twin-axis plot. Loss and a metric like contrast are in
    different units on different scales, and overlaying two y-scales on one
    frame makes any two curves look related -- which is precisely the judgement
    this figure exists to support. One panel per quantity keeps the comparison
    honest and still lines up epoch for epoch.
    """
    shown = [m for m in metrics if m in history.metrics]
    panels = [("loss", history.loss)] + [(m, history.metrics[m]) for m in shown]

    colors = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
    fig, axes = plt.subplots(
        len(panels), 1, figsize=(5.6, 1.9 * len(panels)), sharex=True, squeeze=False
    )

    for ax, (name, values), color in zip(axes[:, 0], panels, colors):
        ax.plot(values, "o-", color=color, linewidth=2, markersize=5,
                markeredgecolor="white", markeredgewidth=0.8)
        ax.set_ylabel(name)
        ax.grid(alpha=0.25, linewidth=0.8)
        ax.spines[["top", "right"]].set_visible(False)
        if values:
            ax.annotate(f"{values[-1]:.3g}", (len(values) - 1, values[-1]),
                        textcoords="offset points", xytext=(6, 0),
                        va="center", fontsize=9, fontweight="bold")

    axes[-1, 0].set_xlabel("epoch")
    fig.align_ylabels(axes[:, 0])
    fig.tight_layout()
    return _finish(fig, path)


def plot_probe_outputs(u, target=None, path=None, labels=None, title=None):
    """Bar chart of probe intensities, target highlighted."""
    u = _to_numpy(u).reshape(-1)
    total = u.sum() or 1.0

    fig, ax = plt.subplots(figsize=(5.2, 3.2))
    colors = ["tab:blue"] * len(u)
    if target is not None:
        colors[int(target)] = "tab:red"
    ax.bar(np.arange(len(u)), u / total, color=colors)
    ax.set_xlabel("probe")
    ax.set_ylabel("normalised intensity")
    if labels is not None:
        ax.set_xticks(np.arange(len(u)), labels)
    ax.set_title(title or "Probe outputs")
    ax.grid(alpha=0.3, axis="y")
    return _finish(fig, path)


def plot_confusion_matrix(cm, classes, path=None, title=None):
    """Row-normalised confusion matrix with counts annotated."""
    cm = _to_numpy(cm).astype(float)
    norm = cm / np.clip(cm.sum(axis=1, keepdims=True), 1, None)

    fig, ax = plt.subplots(figsize=(4.4, 4.0))
    im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(classes)), classes)
    ax.set_yticks(range(len(classes)), classes)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title(title or "Confusion matrix")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i, f"{int(cm[i, j])}",
                ha="center", va="center",
                color="white" if norm[i, j] > 0.5 else "black",
            )
    fig.colorbar(im, ax=ax, label="fraction of true class")
    return _finish(fig, path)


def plot_spectrum(traces, dt, path=None, labels=None, title=None, fmax=None):
    """Amplitude spectrum of per-probe time traces.

    Use with ``model.run(..., record_traces=True)`` to see which frequency each
    probe is actually picking up.
    """
    traces = _to_numpy(traces)
    if traces.ndim == 1:
        traces = traces[:, None]
    T = traces.shape[0]

    freqs = np.fft.rfftfreq(T, dt)
    spec = np.abs(np.fft.rfft(traces - traces.mean(axis=0, keepdims=True), axis=0))

    fig, ax = plt.subplots(figsize=(5.6, 3.4))
    for p in range(spec.shape[1]):
        label = labels[p] if labels is not None else f"probe {p}"
        ax.plot(freqs * 1e-9, spec[:, p], label=label)
    ax.set_xlabel("frequency (GHz)")
    ax.set_ylabel("|FFT|")
    if fmax:
        ax.set_xlim(0, fmax * 1e-9)
    ax.set_title(title or "Probe spectra")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    return _finish(fig, path)
