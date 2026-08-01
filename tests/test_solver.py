"""The differentiable rollout: correctness of the physics and of the gradients."""

from __future__ import annotations

import math

import pytest
import torch

import magnonic_nn as mnn
from magnonic_nn.config import MU_0
from magnonic_nn.dispersion import GAMMA


def _model(cfg, n_probes=3):
    nx, ny, _ = cfg.mesh.n
    src = mnn.LineSource(cfg.mesh, cfg.fields, cfg.material.abc_width, 0,
                         cfg.material.abc_width, ny - 1)
    probes = mnn.linear_probe_array(cfg.mesh, n_probes, x=nx - cfg.material.abc_width - 2,
                                    r=1.0, margin=cfg.material.abc_width)
    return mnn.SpinWaveNetwork(cfg, [src], probes)


# ------------------------------------------------------------------- physics


def test_larmor_precession_frequency(f64):
    """A single spin in a uniform field precesses at ``gamma * H``.

    The most direct check that gamma, the field units and the sign convention
    in the torque all line up. Everything else in the solver rides on this.
    """
    cfg = mnn.get_preset("tiny")
    cfg.mesh.nx = cfg.mesh.ny = cfg.mesh.nz = 1
    cfg.material.abc_width = 0
    cfg.material.alpha = 0.0
    cfg.solver.demag = False
    # The record has to be long enough to resolve the line: the FFT bin spacing
    # is 1/(dt * timesteps), and at 3.5 GHz a 0.2 ns record gives 5 GHz bins --
    # coarser than the frequency being measured. 32 ns gives ~31 MHz bins.
    cfg.solver.dt = 2e-12
    cfg.solver.timesteps = 16384

    alpha = mnn.absorbing_damping(cfg.mesh, cfg.material)
    roll = mnn.LLGRollout(cfg.mesh, cfg.solver, cfg.material.A, alpha, cfg.material.Ms)

    h0 = 1e5  # A/m
    h_static = torch.zeros(1, 1, 1, 3, dtype=torch.float64)
    h_static[..., 2] = h0

    m = torch.zeros(1, 1, 1, 3, dtype=torch.float64)
    m[..., 0] = 1.0  # start transverse so the precession is fully visible

    trace = []
    for _ in range(cfg.solver.timesteps):
        m = roll.rk4_step(m, h_static)
        trace.append(float(m[0, 0, 0, 0]))

    signal = torch.tensor(trace, dtype=torch.float64)
    spectrum = torch.fft.rfft(signal - signal.mean()).abs()
    freqs = torch.fft.rfftfreq(len(signal), d=cfg.solver.dt)
    measured = float(freqs[int(spectrum.argmax())])

    assert measured == pytest.approx(GAMMA * h0 / (2 * math.pi), rel=0.02)


def test_damping_relaxes_towards_the_field(f64):
    """With damping on, ``m`` must rotate into ``H`` and stay there."""
    cfg = mnn.get_preset("tiny")
    cfg.mesh.nx = cfg.mesh.ny = cfg.mesh.nz = 1
    cfg.material.abc_width = 0
    cfg.solver.demag = False
    cfg.solver.dt = 1e-12

    alpha = mnn.absorbing_damping(cfg.mesh, cfg.material)
    roll = mnn.LLGRollout(cfg.mesh, cfg.solver, cfg.material.A, alpha, cfg.material.Ms)

    h_static = torch.zeros(1, 1, 1, 3, dtype=torch.float64)
    h_static[..., 2] = 1e5

    m = torch.tensor([[[[1.0, 0.0, 0.1]]]], dtype=torch.float64)
    m = m / m.norm()
    # The relaxation time is ~(1 + a^2)/(a * gamma * H) ~ 113 ps here, so 400
    # steps of 1 ps is only ~3.5 time constants and leaves ~1e-3 of the initial
    # tilt behind. 1500 steps clears it by an order of magnitude.
    m = roll.relax(m, h_static, steps=1500, alpha_relax=0.5)

    assert float(m[0, 0, 0, 2]) == pytest.approx(1.0, abs=1e-4)


def test_norm_is_conserved_over_a_rollout(tiny_cfg):
    """RK4 keeps ``|m| = 1`` without explicit renormalisation.

    The rollout does not project the magnetisation back onto the unit sphere
    (matching the reference implementation), so norm drift is the honest
    indicator that the timestep is small enough.
    """
    model = _model(tiny_cfg)
    signal = mnn.tone(tiny_cfg, 4.0e9)
    with torch.no_grad():
        result = model.run(signal)

    norms = result.m.norm(dim=-1)
    assert float((norms - 1.0).abs().max()) < 1e-3


def test_absorbing_boundary_suppresses_reflection(f32):
    """A wave reaching the edge must be damped rather than bounced back.

    Compares the energy left in the film after the wave has had time to reach
    the boundary, with and without the absorbing layer. Reflection shows up as
    a much larger residue.
    """
    cfg = mnn.get_preset("tiny")
    cfg.mesh.nx, cfg.mesh.ny = 64, 16
    cfg.mesh.pbc = (0, 1, 0)
    cfg.solver.timesteps = 400
    cfg.solver.relax_steps = 20

    def residual_energy(abc_width):
        cfg.material.abc_width = abc_width
        src = mnn.LineSource(cfg.mesh, cfg.fields, 6, 0, 6, cfg.mesh.ny - 1)
        model = mnn.SpinWaveNetwork(cfg, [src], [mnn.PointProbe(cfg.mesh, 32, 8)])
        # drive for the first half, then let the film ring down
        signal = mnn.tone(cfg, 4.0e9)
        signal[cfg.solver.timesteps // 2 :] = 0.0
        with torch.no_grad():
            result = model.run(signal)
        m0 = model.equilibrium()
        # measure well away from the source so we see returning energy, not drive
        return float(((result.m - m0)[40:, :, 0, 2] ** 2).sum())

    with_abc = residual_energy(10)
    without_abc = residual_energy(0)
    assert with_abc < 0.5 * without_abc


# ------------------------------------------------------------------ gradients


def test_gradient_matches_finite_differences(grad_cfg):
    """Central differences on the design variables versus autograd.

    This is the load-bearing test for the whole package: if these disagree,
    training optimises something other than the simulated physics.
    """
    torch.manual_seed(0)
    model = _model(grad_cfg, n_probes=2)
    signal = mnn.tone(grad_cfg, 4.0e9)
    rho = model.geometry.rho

    with torch.no_grad():
        rho.copy_(torch.randn_like(rho) * 0.2)

    def loss_of(param):
        with torch.no_grad():
            rho.copy_(param)
        model.equilibrium(force=True)
        u = model(signal)
        return u[0] - u[1]

    base = rho.detach().clone()
    loss = loss_of(base)
    loss.backward()
    analytic = rho.grad.detach().clone()

    # probe a few cells that are actually in the design region
    active = (model.geometry.design_mask > 0).nonzero()
    picks = active[:: max(len(active) // 5, 1)][:5]

    eps = 1e-4
    for ix, iy in picks.tolist():
        plus, minus = base.clone(), base.clone()
        plus[ix, iy] += eps
        minus[ix, iy] -= eps
        with torch.no_grad():
            numeric = (float(loss_of(plus)) - float(loss_of(minus))) / (2 * eps)

        a = float(analytic[ix, iy])
        scale = max(abs(a), abs(numeric), 1e-12)
        assert abs(a - numeric) / scale < 2e-3, (
            f"cell ({ix},{iy}): autograd {a:.6e} vs finite difference {numeric:.6e}"
        )


def test_checkpointing_does_not_change_the_gradient(grad_cfg):
    """Recomputing activations must give bit-comparable gradients."""
    torch.manual_seed(1)
    grads = []
    for use_checkpoint in (True, False):
        grad_cfg.solver.checkpoint = use_checkpoint
        torch.manual_seed(1)
        model = _model(grad_cfg, n_probes=2)
        with torch.no_grad():
            model.geometry.rho.copy_(torch.full_like(model.geometry.rho, 0.1))
        u = model(mnn.tone(grad_cfg, 4.0e9))
        (u[0] - u[1]).backward()
        grads.append(model.geometry.rho.grad.detach().clone())

    assert torch.allclose(grads[0], grads[1], rtol=1e-9, atol=1e-14)


def test_chunk_size_does_not_change_the_result(grad_cfg):
    """Checkpoint segmentation is an implementation detail, not physics."""
    outs = []
    for chunk in (None, 1, 5):
        grad_cfg.solver.chunk_size = chunk
        torch.manual_seed(2)
        model = _model(grad_cfg, n_probes=2)
        with torch.no_grad():
            model.geometry.rho.copy_(torch.full_like(model.geometry.rho, 0.05))
            outs.append(model(mnn.tone(grad_cfg, 4.0e9)))

    assert torch.allclose(outs[0], outs[1], rtol=1e-10, atol=0)
    assert torch.allclose(outs[0], outs[2], rtol=1e-10, atol=0)


def test_gradient_reaches_a_trainable_Ms(grad_cfg):
    """The demag field must stay differentiable with respect to ``Ms``."""
    grad_cfg.geometry = "ms"
    grad_cfg.solver.demag = True
    model = _model(grad_cfg, n_probes=2)

    u = model(mnn.tone(grad_cfg, 4.0e9))
    u.sum().backward()

    grad = model.geometry.rho.grad
    assert grad is not None
    assert float(grad.abs().max()) > 0


# ------------------------------------------------------------------ numerics


def test_estimate_max_timestep_skips_singleton_axes():
    """A one-cell-thick film has no z-neighbours, so dz must not constrain dt."""
    cfg = mnn.get_preset("focus")
    thin = mnn.estimate_max_timestep(cfg.mesh, cfg.material, cfg.fields)

    cfg.mesh.nz = 2
    thick = mnn.estimate_max_timestep(cfg.mesh, cfg.material, cfg.fields)
    assert thick < thin


def test_float32_tracks_float64(tiny_cfg):
    """Single precision must not change the answer beyond a fraction of a percent.

    ``float32`` is the default because it roughly halves runtime and checkpoint
    memory; this pins down what that costs.
    """
    def run(precision):
        mnn.set_precision(precision)
        cfg = mnn.get_preset("tiny")
        cfg.mesh.nx = cfg.mesh.ny = 24
        cfg.material.abc_width = 4
        cfg.solver.timesteps = 24
        cfg.solver.relax_steps = 8
        torch.manual_seed(3)
        model = _model(cfg)
        with torch.no_grad():
            return model(mnn.tone(cfg, 4.0e9)).double()

    u32 = run("float32")
    u64 = run("float64")
    rel = ((u32 - u64).abs() / u64.abs().clamp_min(1e-30)).max()
    assert float(rel) < 5e-3


def test_source_interpolation_modes_agree_at_small_dt(f64):
    """Zero-order hold and linear interpolation converge as dt shrinks.

    They are two discretisations of the same continuous drive, so a growing gap
    between them at the configured timestep means the drive is undersampled.
    """
    def run(mode, dt, steps):
        cfg = mnn.get_preset("tiny")
        cfg.mesh.nx = cfg.mesh.ny = 16
        cfg.material.abc_width = 3
        cfg.solver.demag = False
        cfg.solver.dt = dt
        cfg.solver.timesteps = steps
        cfg.solver.relax_steps = 4
        cfg.solver.source_interp = mode
        torch.manual_seed(4)
        model = _model(cfg, n_probes=2)
        with torch.no_grad():
            return model(mnn.tone(cfg, 4.0e9))

    def gap(dt, steps):
        a = run("hold", dt, steps)
        b = run("linear", dt, steps)
        return float(((a - b).abs() / b.abs().clamp_min(1e-30)).max())

    coarse = gap(2e-11, 20)
    fine = gap(5e-12, 80)
    assert fine < coarse


# ------------------------------------------------- the graph-capturable step


def _rollout(f64_or_f32, nx=12, ny=12):
    """A small rollout with spatially varying damping, which is the case the
    graph path had to be taught to handle."""
    cfg = mnn.get_preset("tiny")
    cfg.mesh.nx, cfg.mesh.ny, cfg.mesh.nz = nx, ny, 1
    cfg.material.abc_width = 2
    cfg.solver.demag = False
    cfg.solver.dt = 1e-12
    cfg.solver.timesteps = 8
    alpha = mnn.absorbing_damping(cfg.mesh, cfg.material)
    roll = mnn.LLGRollout(cfg.mesh, cfg.solver, cfg.material.A, alpha,
                          cfg.material.Ms)
    torch.manual_seed(0)
    m = torch.zeros(nx, ny, 1, 3, dtype=torch.get_default_dtype())
    m[..., 0] = 1.0
    h = torch.zeros_like(m)
    h[..., 2] = 1e4
    return roll, m, h


def test_field_stepper_matches_the_callable_stepper(f64):
    """``rk4_step_fields`` and ``rk4_step`` are the same integrator.

    The whole reason the field form exists is that a Python callable per substep
    forces a graph break, so the two must agree exactly or the CUDA path is
    silently integrating something else. Every NARMA run since the graph stepper
    landed goes through ``rk4_step_fields`` -- including on CPU, where
    ``graph_stepper`` returns it unwrapped -- so this is the step the results
    are actually made of.
    """
    roll, m, h = _rollout(f64)
    drive = torch.zeros_like(h)
    drive[..., 1] = 5e3

    a = m.clone()
    for _ in range(6):
        a = roll.rk4_step(a, h, lambda theta: drive * (1.0 + theta))
    b = m.clone()
    for _ in range(6):
        b = roll.rk4_step_fields(b, h + drive, h + drive * 1.5,
                                 h + drive * 1.5, h + drive * 2.0)
    assert torch.equal(a, b)


def test_graph_stepper_falls_back_to_eager_off_cuda(f64):
    """Off CUDA the compiled path is not available, and the caller must not have
    to know: ``graph_stepper`` hands back the plain function so the driving loop
    is identical either way."""
    roll, _, _ = _rollout(f64)
    if not torch.cuda.is_available():
        fn, compiled = roll.graph_stepper()
        assert compiled is False
        assert fn == roll.rk4_step_fields


def test_graph_stepper_declines_to_compile_with_thermal_noise(f64):
    """A replayed graph freezes its random draws, which would turn Langevin
    noise into a fixed repeating pattern -- noise-shaped, but not noise. The
    stepper must refuse to compile rather than produce that silently."""
    roll, _, _ = _rollout(f64)
    roll.set_temperature(300.0)
    fn, compiled = roll.graph_stepper()
    assert compiled is False
    assert fn == roll.rk4_step_fields


def test_alpha_registration_precomputes_gamma_prime(f64):
    """``gamma'`` must exist before the first step, not be allocated inside it.

    Allocating it lazily inside the stepper let a captured graph take ownership
    of the cached tensor, so every replay read memory that had since been
    overwritten. Precomputing makes it a stable input the graph closes over.
    """
    roll, m, h = _rollout(f64)
    assert roll._gamma_prime is not None
    alpha, gp = roll._gamma_prime
    assert alpha is roll.alpha
    assert torch.allclose(gp, roll.gamma / (1.0 + alpha ** 2))
    # a caller-supplied alpha must NOT poison the cache
    other = torch.full_like(alpha, 0.5)
    roll.torque(m, h, other)
    assert roll._gamma_prime[0] is roll.alpha


def test_thermal_field_is_zero_where_there_is_no_material(f64):
    """Langevin noise is driven by fluctuation-dissipation, and outside the
    magnet there is nothing to dissipate. Noise leaking into Ms = 0 cells would
    excite geometry that does not exist."""
    roll, m, _ = _rollout(f64)
    Ms = torch.full((m.shape[0], m.shape[1], 1, 1), 8e5,
                    dtype=torch.get_default_dtype())
    Ms[:4] = 0.0
    roll.set_Ms(Ms)
    roll.set_temperature(300.0)
    h_th = roll._thermal_field(m)
    assert float(h_th[:4].abs().max()) == 0.0
    assert float(h_th[4:].abs().max()) > 0.0
