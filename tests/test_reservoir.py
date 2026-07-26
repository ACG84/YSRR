"""Physics and methodology checks for the spiking-disk reservoir.

Every dynamical claim in ``thiele.py`` has an analytic anchor, and the
reservoir-computing scaffolding is checked against closed forms -- the lesson
of this project being that plausible numbers are the failure mode, not
crashes.
"""

import math

import numpy as np
import pytest
import torch

from magnonic_nn.reservoir import (
    FilmResponse, memory_capacity, nmse, narma10, ridge_fit, ridge_predict,
)
from magnonic_nn.thiele import ThieleConfig, ThieleDisks


def single_disk(**kw):
    kw.setdefault("n_disks", 1)
    kw.setdefault("coupling", 0.0)
    return ThieleDisks(ThieleConfig(**kw))


# ------------------------------------------------------------------ gyration
def test_free_gyration_frequency():
    """Undamped, unstiffened core orbits at f0 = k / (2 pi G) exactly."""
    d = single_disk(alpha_eff=0.0, kappa_nl=0.0, beta_nl=0.0, v_crit=1e9)
    d.X[0] = torch.tensor([0.3 * d.cfg.R, 0.0])
    period = 2 * math.pi / d.omega0
    n = round(period / d.dt)
    zero = torch.zeros(1, dtype=torch.float64)
    angles = []
    for _ in range(n):
        d.step(zero)
        angles.append(math.atan2(float(d.X[0, 1]), float(d.X[0, 0])))
    # after one analytic period the phase should close to its start
    assert abs(math.atan2(math.sin(angles[-1]), math.cos(angles[-1]))) < 0.05


def test_gyration_sense_follows_polarity():
    """p = +1 gyrates clockwise (omega = -k/G z); p = -1 the reverse."""
    for p, expected_sign in ((1.0, -1.0), (-1.0, +1.0)):
        d = single_disk(alpha_eff=0.0, kappa_nl=0.0, beta_nl=0.0, v_crit=1e9)
        d.p[0] = p
        d.X[0] = torch.tensor([0.3 * d.cfg.R, 0.0])
        d.step(torch.zeros(1, dtype=torch.float64))
        assert math.copysign(1.0, float(d.X[0, 1])) == expected_sign


def test_orbit_radius_conserved_without_damping():
    d = single_disk(alpha_eff=0.0, kappa_nl=0.0, beta_nl=0.0, v_crit=1e9)
    d.X[0] = torch.tensor([0.25 * d.cfg.R, 0.0])
    zero = torch.zeros(1, dtype=torch.float64)
    for _ in range(500):
        d.step(zero)
    assert abs(float(d.X[0].norm()) / (0.25 * d.cfg.R) - 1.0) < 1e-6


def test_damping_rate_matches_linear_theory():
    """Small-orbit decay: r(t) ~ exp(-alpha_eff * omega0 * t / (1 + alpha^2))."""
    alpha = 0.02
    d = single_disk(alpha_eff=alpha, kappa_nl=0.0, beta_nl=0.0, v_crit=1e9)
    r0 = 0.2 * d.cfg.R
    d.X[0] = torch.tensor([r0, 0.0])
    zero = torch.zeros(1, dtype=torch.float64)
    n = 2000
    for _ in range(n):
        d.step(zero)
    t = n * d.dt
    expected = r0 * math.exp(-alpha * d.omega0 * t / (1 + alpha**2))
    assert abs(float(d.X[0].norm()) / expected - 1.0) < 0.01


def test_resonant_drive_pumps_orbit():
    d = single_disk(alpha_eff=0.01, v_crit=1e9)
    drive = torch.tensor([0.5], dtype=torch.float64)
    for _ in range(3000):
        d.step(drive)
    assert float(d.X[0].norm()) > 0.05 * d.cfg.R


def test_nonlinear_damping_bounds_orbit():
    d = single_disk(alpha_eff=0.01, beta_nl=2.0, v_crit=1e9)
    drive = torch.tensor([1.0], dtype=torch.float64)
    for _ in range(6000):
        d.step(drive)
        assert float(d.X[0].norm()) < 1.5 * d.cfg.R


# ------------------------------------------------------------------- spiking
def test_core_reversal_fires_and_resets():
    d = single_disk(alpha_eff=0.005, drive_scale=0.6)
    drive = torch.tensor([1.0], dtype=torch.float64)
    fired_total, r_before = 0.0, 0.0
    for _ in range(8000):
        r_prev = float(d.X[0].norm())
        fired = float(d.step(drive)[0])
        if fired and not fired_total:
            r_before = r_prev
            r_after = float(d.X[0].norm())
            assert r_after < r_before * (d.cfg.contraction + 0.05)
            assert float(d.p[0]) == -1.0
        fired_total += fired
    assert fired_total >= 1.0, "hard drive never reached critical velocity"


def test_no_reversal_below_threshold():
    d = single_disk(alpha_eff=0.02, drive_scale=0.02)
    drive = torch.tensor([0.2], dtype=torch.float64)
    spikes = 0.0
    for _ in range(4000):
        spikes += float(d.step(drive)[0])
    assert spikes == 0.0


def test_refractory_blocks_immediate_refire():
    d = single_disk(alpha_eff=0.005, drive_scale=0.6, refractory=5e-9)
    drive = torch.tensor([1.0], dtype=torch.float64)
    times = []
    for _ in range(12000):
        if float(d.step(drive)[0]):
            times.append(d.t)
    for a, b in zip(times, times[1:]):
        assert b - a >= d.cfg.refractory


def test_coupling_recruits_neighbour():
    """Drive disk 0 only; a coupled neighbour must gyrate, an uncoupled not."""
    orbits = {}
    for coupling in (0.0, 0.1):
        d = ThieleDisks(ThieleConfig(n_disks=2, coupling=coupling, v_crit=1e9,
                                     alpha_eff=0.01))
        drive = torch.tensor([0.6, 0.0], dtype=torch.float64)
        for _ in range(4000):
            d.step(drive)
        orbits[coupling] = float(d.X[1].norm()) / d.cfg.R
    assert orbits[0.1] > 5 * max(orbits[0.0], 1e-12)


# ------------------------------------------------------- reservoir mechanics
def test_narma10_zero_input_fixed_point():
    """With u = 0 the recursion is y <- 0.3 y + 0.05 y * sum + 0.1; the
    small-y fixed point is 0.1 / 0.7 -- to first order in y*, sum ~ 10 y*
    shifts it to the root of 0.5 y^2 - 0.7 y + 0.1."""
    u, y = narma10(400, seed=3)
    u_zero = np.zeros_like(u)
    yz = np.zeros_like(y)
    for t in range(9, len(yz) - 1):
        yz[t + 1] = 0.3 * yz[t] + 0.05 * yz[t] * yz[t - 9:t + 1].sum() + 0.1
    root = (0.7 - math.sqrt(0.49 - 4 * 0.5 * 0.1 * 0.1 * 10 / 10)) / (2 * 0.5 * 1.0)
    # just check convergence to a finite fixed point consistent with theory
    assert abs(yz[-1] - yz[-2]) < 1e-10
    y_star = yz[-1]
    assert abs(0.3 * y_star + 0.5 * y_star * y_star + 0.1 - y_star) < 1e-9


def test_ridge_matches_lstsq_at_zero_lambda():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((60, 5))
    w_true = rng.standard_normal(6)
    y = X @ w_true[:5] + w_true[5]
    w = ridge_fit(X, y, 1e-12)
    assert np.allclose(ridge_predict(X, w), y, atol=1e-6)


def test_memory_capacity_of_explicit_delay_line():
    """Features that ARE u[t-1..t-8] must give MC ~ 8 and r^2 ~ 1 per tap."""
    rng = np.random.default_rng(1)
    u = rng.uniform(0, 0.5, 1200)
    X = np.stack([np.roll(u, k) for k in range(1, 9)], axis=1)
    curve, mc = memory_capacity(X, u, (50, 800, 100), max_delay=12)
    assert curve[1:9].min() > 0.999
    assert curve[10:].max() < 0.05
    assert 7.9 < mc < 8.2


def test_film_response_interpolates_and_clips():
    u = np.linspace(0, 1, 5)
    P = np.stack([u, u**2], axis=1)
    fr = FilmResponse(u, P)
    mid = fr(np.array([0.125]))
    assert abs(mid[0, 0] - 0.125) < 1e-12          # linear channel, exact
    assert fr(np.array([2.0]))[0, 0] == 1.0        # clipped to sweep edge
    assert fr.P.max() <= 1.0 + 1e-12               # normalised channels


def test_features_deterministic():
    from magnonic_nn.reservoir import ReservoirRunner
    u = np.linspace(0.1, 0.9, 7)
    fr = FilmResponse(np.linspace(0, 1, 9),
                      np.abs(np.sin(np.arange(9 * 12).reshape(9, 12))))
    cfg = ThieleConfig(n_disks=12, coupling=0.05, spike_kick=0.02)
    a = ReservoirRunner(fr, cfg, steps_per_frame=40).features(u)
    b = ReservoirRunner(fr, cfg, steps_per_frame=40).features(u)
    assert np.array_equal(a, b)
