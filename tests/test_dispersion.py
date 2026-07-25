"""Analytic dispersion: limits, monotonicity, and inversion."""

from __future__ import annotations

import math

import pytest

import magnonic_nn as mnn
from magnonic_nn.config import MU_0, FieldConfig, MaterialConfig
from magnonic_nn.dispersion import GAMMA, frequency_of_k, k_of_frequency, omega_of_k


@pytest.fixture
def film():
    return FieldConfig(), MaterialConfig(), 20e-9


def test_k_zero_is_kittel(film):
    fields, material, d = film
    assert omega_of_k(0.0, fields, material, d) / (2 * math.pi) == pytest.approx(
        mnn.kittel_fmr(fields, material)
    )


def test_kittel_matches_closed_form(film):
    """f0 = gamma/(2 pi) sqrt(H0 (H0 + Ms)), the in-plane thin-film resonance."""
    fields, material, _ = film
    h0 = fields.B0 / MU_0
    expected = GAMMA / (2 * math.pi) * math.sqrt(h0 * (h0 + material.Ms))
    assert mnn.kittel_fmr(fields, material) == pytest.approx(expected)


def test_thick_film_limit_is_damon_eshbach(film):
    """Without exchange, kd >> 1 must approach omega_H + omega_M / 2."""
    fields, material, _ = film
    thick = 1e-3  # so kd >> 1 while k stays small enough that exchange is negligible
    omega_h = GAMMA * fields.B0 / MU_0
    omega_m = GAMMA * material.Ms

    omega = omega_of_k(1e5, fields, material, thick)
    assert omega == pytest.approx(omega_h + omega_m / 2, rel=1e-3)


def test_monotonic_in_k(film):
    fields, material, d = film
    ks = [0.0, 1e5, 1e6, 5e6, 1e7, 5e7, 1e8]
    freqs = [frequency_of_k(k, fields, material, d) for k in ks]
    assert all(b > a for a, b in zip(freqs, freqs[1:]))


@pytest.mark.parametrize("freq", [3.5e9, 4.0e9, 5.0e9, 8.0e9])
def test_inversion_round_trips(film, freq):
    fields, material, d = film
    k = k_of_frequency(freq, fields, material, d)
    assert k > 0
    assert frequency_of_k(k, fields, material, d) == pytest.approx(freq, rel=1e-6)


def test_below_fmr_has_no_propagating_solution(film):
    fields, material, d = film
    below = 0.5 * mnn.kittel_fmr(fields, material)
    assert k_of_frequency(below, fields, material, d) == 0.0
    assert mnn.wavelength(below, fields, material, d) == float("inf")


def test_dipolar_term_dominates_at_4GHz(film):
    """Guards against silently reverting to an exchange-only estimate.

    In 20 nm YIG at 4 GHz the true wavelength is ~460 nm; an exchange-only
    treatment gives ~160 nm. A factor of three in wavelength is a factor of
    three in how many cells you need, so this distinction is not academic.
    """
    fields, material, d = film
    lam = mnn.wavelength(4.0e9, fields, material, d)
    assert 350e-9 < lam < 600e-9

    # exchange-only prediction, for contrast
    omega = 2 * math.pi * 4.0e9
    excess = omega / GAMMA - fields.B0 / MU_0
    k_ex = math.sqrt(excess * MU_0 * material.Ms / (2 * material.A))
    assert 2 * math.pi / k_ex < 0.5 * lam


def test_usable_band_is_ordered_and_sane(f32):
    cfg = mnn.get_preset("focus")
    lo, hi = mnn.usable_band(cfg)
    assert lo > mnn.kittel_fmr(cfg.fields, cfg.material)
    assert hi > lo

    # the upper edge is exactly where the wavelength hits 6 cells
    lam = mnn.wavelength(hi, cfg.fields, cfg.material, cfg.mesh.dz)
    assert lam == pytest.approx(6 * cfg.mesh.dx, rel=1e-3)


@pytest.mark.slow
def test_measured_dispersion_matches_theory(f32):
    """Full-stack check: mesh, field terms, timestep and boundaries together.

    Tolerance is 5 % at 4.0 and 4.5 GHz, in the middle of the band. The edges
    are looser for physical reasons (window length at low k, mesh resolution at
    high k) and are covered by the standalone script instead.
    """
    cfg = mnn.get_preset("focus")
    cfg.solver.timesteps = 600

    results = mnn.measure_dispersion(cfg, freqs=[4.0e9, 4.5e9], nx=160, ny=8)
    for r in results:
        assert r["rel_error"] < 0.05, (
            f"{r['freq'] / 1e9:.1f} GHz: measured {r['lambda_measured'] * 1e9:.0f} nm "
            f"vs analytic {r['lambda_analytic'] * 1e9:.0f} nm"
        )
