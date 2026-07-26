"""Film -> spiking-disk reservoir pipeline, and the benchmarks that judge it.

The architecture under trial, front to back:

1. **Film (measured, fixed).** The spin-wave scatterer from the classification
   work, driven by a fixed multitone whose *amplitude* carries the input
   sample ``u_n``. Its probe powers are "what leaks where": a nonlinear,
   frequency-mixing projection of the input measured by real micromagnetics.
   Because the trial has no disk-to-film feedback, the film acts frame-wise as
   a static nonlinear map u -> P, so it is measured once on a grid
   (``scripts/measure_film_response.py``) and interpolated here. That is a
   surrogate for *cost*, not physics: every value it returns is a measured
   film response.
2. **Disks (dynamic, spiking).** Each probe's leaked power drives one Thiele
   disk (`magnonic_nn.thiele`). The disks carry the memory (underdamped
   gyration), the nonlinearity (stiffening, saturation, reversal), and the
   spiking (reversal threshold + refractory + coupling).
3. **Readout (the only trained part).** Ridge regression from per-frame disk
   features to the target. Classic reservoir computing: training touches
   nothing physical.

Benchmarks: NARMA-10, the standard test that needs ~10 steps of memory *and*
input nonlinearity at once, scored as NMSE; a linear memory-capacity curve
(Jaeger's MC); and one-step Mackey-Glass prediction. Baselines are nested so
each stage has to earn its keep: readout on the raw input alone, then on the
memoryless film features, then the full reservoir, then ablations that
lobotomise the spiking to show it mattered.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch

from .thiele import ThieleConfig, ThieleDisks

__all__ = [
    "FilmResponse", "narma10", "mackey_glass", "ridge_fit", "ridge_predict",
    "nmse", "memory_capacity", "ReservoirRunner",
]


# --------------------------------------------------------------------- film
class FilmResponse:
    """Interpolated probe-power response P_i(u) measured from the film.

    Built from the sweep saved by ``scripts/measure_film_response.py``:
    ``u_grid`` (n_u,) and ``P`` (n_u, n_probes). Channels are normalised to
    their sweep maximum so the disk layer sees O(1) drives regardless of the
    film's arbitrary intensity units.
    """

    def __init__(self, u_grid: np.ndarray, P: np.ndarray):
        self.u_grid = np.asarray(u_grid, dtype=np.float64)
        P = np.asarray(P, dtype=np.float64)
        scale = P.max(axis=0)
        scale[scale <= 0] = 1.0
        self.P = P / scale
        self.n_channels = P.shape[1]

    @classmethod
    def load(cls, path) -> "FilmResponse":
        blob = torch.load(path, map_location="cpu", weights_only=False)
        return cls(blob["u_grid"].numpy(), blob["P"].numpy())

    def __call__(self, u: np.ndarray) -> np.ndarray:
        """(n_samples,) -> (n_samples, n_channels), clipped to the sweep range."""
        u = np.clip(np.asarray(u, dtype=np.float64), self.u_grid[0], self.u_grid[-1])
        return np.stack(
            [np.interp(u, self.u_grid, self.P[:, i]) for i in range(self.n_channels)],
            axis=-1,
        )


# --------------------------------------------------------------------- tasks
def narma10(n: int, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """The standard NARMA-10 sequence: u ~ U[0, 0.5], and

        y[t+1] = 0.3 y[t] + 0.05 y[t] * sum_{i=0}^{9} y[t-i]
                 + 1.5 u[t-9] u[t] + 0.1

    Needs both a 10-step memory and a product of inputs 9 steps apart, which
    is exactly the pairing a reservoir is supposed to supply. The recursion
    can diverge for unlucky draws; those seeds are rejected and redrawn.
    """
    rng = np.random.default_rng(seed)
    for attempt in range(100):
        u = rng.uniform(0.0, 0.5, size=n)
        y = np.zeros(n)
        ok = True
        for t in range(9, n - 1):
            y[t + 1] = (0.3 * y[t] + 0.05 * y[t] * y[t - 9:t + 1].sum()
                        + 1.5 * u[t - 9] * u[t] + 0.1)
            if not np.isfinite(y[t + 1]) or abs(y[t + 1]) > 1e3:
                ok = False
                break
        if ok:
            return u, y
        rng = np.random.default_rng(seed + 1000 + attempt)
    raise RuntimeError("NARMA-10 diverged for 100 consecutive seeds")


def mackey_glass(n: int, tau: int = 17, dt: float = 1.0, seed: int = 0,
                 washout: int = 500) -> np.ndarray:
    """Mackey-Glass series (beta=0.2, gamma=0.1, n=10), RK4 at ``dt``."""
    rng = np.random.default_rng(seed)
    hist_len = int(round(tau / dt))
    x = list(1.2 + 0.05 * rng.standard_normal(hist_len + 1))

    def f(xi, xd):
        return 0.2 * xd / (1.0 + xd**10) - 0.1 * xi

    for _ in range(n + washout):
        xi, xd = x[-1], x[-1 - hist_len]
        k1 = f(xi, xd)
        k2 = f(xi + 0.5 * dt * k1, xd)
        k3 = f(xi + 0.5 * dt * k2, xd)
        k4 = f(xi + dt * k3, xd)
        x.append(xi + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4))
    return np.array(x[-n:])


# ------------------------------------------------------------------- readout
def ridge_fit(X: np.ndarray, y: np.ndarray, lam: float) -> np.ndarray:
    """Closed-form ridge with a bias column, in float64."""
    Xb = np.concatenate([X, np.ones((len(X), 1))], axis=1)
    A = Xb.T @ Xb + lam * np.eye(Xb.shape[1])
    return np.linalg.solve(A, Xb.T @ y)


def ridge_predict(X: np.ndarray, w: np.ndarray) -> np.ndarray:
    Xb = np.concatenate([X, np.ones((len(X), 1))], axis=1)
    return Xb @ w


def nmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean((y_true - y_pred) ** 2) / np.var(y_true))


def fit_eval(X: np.ndarray, y: np.ndarray, splits: tuple[int, int, int],
             lams=(1e-8, 1e-6, 1e-4, 1e-2, 1.0)) -> dict:
    """Washout / train / val / test protocol with lambda chosen on val."""
    n_wash, n_train, n_val = splits
    tr = slice(n_wash, n_wash + n_train)
    va = slice(n_wash + n_train, n_wash + n_train + n_val)
    te = slice(n_wash + n_train + n_val, len(y))

    best = None
    for lam in lams:
        w = ridge_fit(X[tr], y[tr], lam)
        e = nmse(y[va], ridge_predict(X[va], w))
        if best is None or e < best[0]:
            best = (e, lam, w)
    _, lam, w = best
    return {
        "lambda": lam,
        "nmse_train": nmse(y[tr], ridge_predict(X[tr], w)),
        "nmse_val": best[0],
        "nmse_test": nmse(y[te], ridge_predict(X[te], w)),
    }


def memory_capacity(X: np.ndarray, u: np.ndarray, splits, max_delay: int = 25,
                    lam: float = 1e-6) -> tuple[np.ndarray, float]:
    """Jaeger's linear memory capacity: MC = sum_k r^2(u[t-k], u_hat[t-k])."""
    n_wash, n_train, n_val = splits
    tr = slice(n_wash, n_wash + n_train)
    te = slice(n_wash + n_train + n_val, len(u))
    r2 = np.zeros(max_delay + 1)
    for k in range(1, max_delay + 1):
        target = np.roll(u, k)
        w = ridge_fit(X[tr], target[tr], lam)
        pred = ridge_predict(X[te], w)
        c = np.corrcoef(target[te], pred)[0, 1]
        r2[k] = max(c, 0.0) ** 2 if np.isfinite(c) else 0.0
    return r2, float(r2.sum())


# ------------------------------------------------------------------ pipeline
@dataclass
class ReservoirRunner:
    """Drive the film + disk stack with an input series and collect features."""

    film: FilmResponse
    thiele_cfg: ThieleConfig
    steps_per_frame: int = 250
    include_input: bool = True

    def features(self, u: np.ndarray, progress_every: int = 0) -> np.ndarray:
        """(n_samples,) input series -> (n_samples, n_features) design matrix."""
        P = self.film(u)                                   # (n, channels)
        disks = ThieleDisks(self.thiele_cfg)
        rows = []
        drive = torch.zeros(self.thiele_cfg.n_disks, dtype=torch.float64)
        for n, p_row in enumerate(P):
            drive.copy_(torch.from_numpy(p_row[: self.thiele_cfg.n_disks]))
            feats = disks.run_frame(drive, self.steps_per_frame)
            rows.append(feats.flatten().numpy())
            if progress_every and (n + 1) % progress_every == 0:
                print(f"  frame {n + 1}/{len(P)}", flush=True)
        F = np.stack(rows)
        if self.include_input:
            F = np.concatenate([F, u[:, None]], axis=1)
        return F
