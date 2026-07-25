"""Ready-made tasks.

Each builder returns a :class:`Task` bundling everything a training script
needs: the network, the drive signals, the targets, the objective and the
metrics worth watching. They are deliberately thin -- read one and you will see
exactly how to build your own.

* :func:`build_focusing` -- steer a single tone onto one probe. The "hello
  world": no classification, one input, an unambiguous right answer, and a
  result you can see in the intensity map.
* :func:`build_demux` -- send three frequencies to three different probes. The
  paper's frequency-separation experiment. Works in the linear regime because
  distinct frequencies do not need to interact.
* :func:`build_vowels` -- classify three vowels. Needs the non-linear regime:
  the classes differ in spectral *envelope*, and a linear medium maps each
  frequency component independently, so it can only ever compute a linear
  functional of the input spectrum.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import torch

from .config import SimConfig, get_preset
from .geometry import build_geometry, interior_mask
from .losses import accuracy, contrast, focus_loss, intensity_cross_entropy
from .model import SpinWaveNetwork
from .probes import linear_probe_array
from .signals import gated_tones, kittel_fmr, tone, usable_band, vowel_dataset
from .sources import LineSource

__all__ = ["Task", "build_focusing", "build_demux", "build_vowels"]


@dataclass
class Task:
    """A network plus the data and objective it is trained against."""

    name: str
    model: SpinWaveNetwork
    signals: torch.Tensor
    targets: torch.Tensor
    loss_fn: object
    metric_fns: dict = field(default_factory=dict)
    classes: list | None = None
    test_signals: torch.Tensor | None = None
    test_targets: torch.Tensor | None = None
    info: dict = field(default_factory=dict)

    @property
    def has_test_set(self) -> bool:
        return self.test_signals is not None and len(self.test_signals) > 0


def _check_band(cfg: SimConfig, freqs, what: str):
    """Warn if a drive frequency falls outside what the film can carry."""
    f_lo, f_hi = usable_band(cfg)
    fmr = kittel_fmr(cfg.fields, cfg.material)
    for f in freqs:
        if f < fmr:
            warnings.warn(
                f"{what}: {f / 1e9:.2f} GHz is below the {fmr / 1e9:.2f} GHz FMR of this "
                f"film -- the response will be evanescent, not a propagating wave. "
                f"Raise the frequency above {f_lo / 1e9:.2f} GHz or lower B0.",
                stacklevel=3,
            )
        elif f > f_hi:
            warnings.warn(
                f"{what}: {f / 1e9:.2f} GHz is above {f_hi / 1e9:.2f} GHz, where the "
                f"wavelength drops below ~6 cells and the mesh starts to alias it. "
                f"Refine dx or lower the frequency.",
                stacklevel=3,
            )


def _default_source(cfg: SimConfig, x: int | None = None):
    """Line source spanning the film, just inside the absorbing boundary."""
    nx, ny, _ = cfg.mesh.n
    if x is None:
        x = cfg.material.abc_width
    return LineSource(cfg.mesh, cfg.fields, x, 0, x, ny - 1)


def _design_mask(cfg: SimConfig, source_x: int, probe_x: int):
    """Trainable region: inside the absorbing boundary, between source and probes."""
    mask = interior_mask(cfg.mesh, cfg.material, margin=0)
    mask[: source_x + 2, :] = 0.0
    mask[probe_x - 1 :, :] = 0.0
    return mask


def build_focusing(
    cfg: SimConfig | None = None,
    freq: float = 4.0e9,
    n_probes: int = 19,
    target: int | None = None,
    probe_radius: float = 2.0,
    geometry_kwargs: dict | None = None,
) -> Task:
    """Focus a single tone onto one of ``n_probes`` detectors.

    Mirrors the reference implementation's ``focus.py``: a line source on the
    left, a column of probes on the right, and the middle probe as the target.
    """
    cfg = cfg or get_preset("focus")
    _check_band(cfg, [freq], "focusing")

    nx, ny, _ = cfg.mesh.n
    source_x = cfg.material.abc_width
    probe_x = nx - 15 if nx > 40 else nx - cfg.material.abc_width - 2

    src = _default_source(cfg, source_x)
    probes = linear_probe_array(cfg.mesh, n_probes, x=probe_x, r=probe_radius,
                                margin=cfg.material.abc_width)

    geometry = build_geometry(
        cfg, design_mask=_design_mask(cfg, source_x, probe_x), **(geometry_kwargs or {})
    )
    model = SpinWaveNetwork(cfg, [src], probes, geometry=geometry)

    signals = tone(cfg, freq).unsqueeze(0)
    target = n_probes // 2 if target is None else int(target)
    targets = torch.tensor([target], dtype=torch.long)

    return Task(
        name="focusing",
        model=model,
        signals=signals,
        targets=targets,
        loss_fn=focus_loss,
        metric_fns={"contrast_dB": lambda u, t: contrast(u, t).mean()},
        info={"freq": freq, "target_probe": target, "fmr": kittel_fmr(cfg.fields, cfg.material)},
    )


def build_demux(
    cfg: SimConfig | None = None,
    freqs=None,
    probe_radius: float = 3.0,
    geometry_kwargs: dict | None = None,
) -> Task:
    """Route each of several frequencies to its own probe.

    The paper demonstrates this at 3.0 / 3.5 / 4.0 GHz. The default here is
    3.5 / 4.0 / 4.5 GHz because at the default 60 mT bias this film's FMR sits
    at ~3.3 GHz, and a 3.0 GHz drive would be evanescent rather than
    propagating. Lower ``B0`` if you want the paper's exact triple: 30 mT puts
    the FMR near 2.3 GHz.
    """
    cfg = cfg or get_preset("demux")
    freqs = list(freqs) if freqs is not None else [3.5e9, 4.0e9, 4.5e9]
    _check_band(cfg, freqs, "demux")

    nx, ny, _ = cfg.mesh.n
    source_x = cfg.material.abc_width
    probe_x = nx - 15 if nx > 40 else nx - cfg.material.abc_width - 2

    src = _default_source(cfg, source_x)
    probes = linear_probe_array(cfg.mesh, len(freqs), x=probe_x, r=probe_radius,
                                margin=cfg.material.abc_width)

    geometry = build_geometry(
        cfg, design_mask=_design_mask(cfg, source_x, probe_x), **(geometry_kwargs or {})
    )
    model = SpinWaveNetwork(cfg, [src], probes, geometry=geometry)

    signals = torch.stack([tone(cfg, f) for f in freqs])
    targets = torch.arange(len(freqs), dtype=torch.long)

    return Task(
        name="demux",
        model=model,
        signals=signals,
        targets=targets,
        loss_fn=intensity_cross_entropy,
        metric_fns={
            "accuracy": accuracy,
            "contrast_dB": lambda u, t: contrast(u, t).mean(),
        },
        classes=[f"{f / 1e9:.1f} GHz" for f in freqs],
        info={"freqs": freqs, "fmr": kittel_fmr(cfg.fields, cfg.material)},
    )


def build_vowels(
    cfg: SimConfig | None = None,
    classes=("ae", "ei", "iy"),
    n_per_class: int = 16,
    n_train_per_class: int = 4,
    jitter: float = 0.06,
    probe_radius: float = 3.0,
    seed: int = 0,
    geometry_kwargs: dict | None = None,
) -> Task:
    """Classify vowels by routing each class to its own probe.

    Follows the paper's protocol of training on a handful of tokens per vowel
    and testing on the rest, so the reported test accuracy measures
    generalisation across "speakers" rather than memorisation.

    Run this with ``cfg.fields.Bt`` in the tens of mT. At 1 mT the film is
    linear, and a linear medium cannot separate classes that differ only in
    spectral envelope: superposition means each frequency is routed
    independently of the others, so the probe intensities are a fixed linear
    functional of the input power spectrum.
    """
    cfg = cfg or get_preset("vowels")

    nx, ny, _ = cfg.mesh.n
    source_x = cfg.material.abc_width
    probe_x = nx - 15 if nx > 40 else nx - cfg.material.abc_width - 2

    src = _default_source(cfg, source_x)
    probes = linear_probe_array(cfg.mesh, len(classes), x=probe_x, r=probe_radius,
                                margin=cfg.material.abc_width)

    geometry = build_geometry(
        cfg, design_mask=_design_mask(cfg, source_x, probe_x), **(geometry_kwargs or {})
    )
    model = SpinWaveNetwork(cfg, [src], probes, geometry=geometry)

    data = vowel_dataset(cfg, classes=classes, n_per_class=n_per_class, jitter=jitter, seed=seed)
    gen = torch.Generator(device=data.labels.device).manual_seed(seed)
    train_set, test_set = data.split(n_train_per_class, generator=gen)

    if cfg.fields.Bt < 5e-3:
        warnings.warn(
            f"Bt = {cfg.fields.Bt * 1e3:.1f} mT keeps the film in the linear regime; "
            f"vowel classification needs the non-linear regime (tens of mT) to beat "
            f"chance by much. See the paper's linear-vs-non-linear comparison.",
            stacklevel=2,
        )

    return Task(
        name="vowels",
        model=model,
        signals=train_set.signals,
        targets=train_set.labels,
        test_signals=test_set.signals,
        test_targets=test_set.labels,
        loss_fn=intensity_cross_entropy,
        metric_fns={"accuracy": accuracy},
        classes=list(classes),
        info={
            "band": usable_band(cfg),
            "fmr": kittel_fmr(cfg.fields, cfg.material),
            "n_train": len(train_set),
            "n_test": len(test_set),
        },
    )
