"""The network wrapper, the task builders and the training loop."""

from __future__ import annotations

import warnings

import pytest
import torch

import magnonic_nn as mnn


def _model(cfg, n_probes=3):
    nx, ny, _ = cfg.mesh.n
    src = mnn.LineSource(cfg.mesh, cfg.fields, cfg.material.abc_width, 0,
                         cfg.material.abc_width, ny - 1)
    probes = mnn.linear_probe_array(cfg.mesh, n_probes, x=nx - cfg.material.abc_width - 2,
                                    r=1.0, margin=cfg.material.abc_width)
    return mnn.SpinWaveNetwork(cfg, [src], probes)


# --------------------------------------------------------------------- model


def test_forward_shapes_batched_and_unbatched(tiny_cfg):
    model = _model(tiny_cfg, n_probes=4)
    signal = mnn.tone(tiny_cfg, 4.0e9)

    with torch.no_grad():
        single = model(signal)
        batch = model(torch.stack([signal, signal, signal]))

    assert single.shape == (4,)
    assert batch.shape == (3, 4)


def test_batch_is_the_same_as_looping(tiny_cfg):
    model = _model(tiny_cfg, n_probes=3)
    a = mnn.tone(tiny_cfg, 4.0e9)
    b = mnn.tone(tiny_cfg, 4.4e9)

    with torch.no_grad():
        batched = model(torch.stack([a, b]))
        looped = torch.stack([model(a), model(b)])

    assert torch.allclose(batched, looped, rtol=1e-10, atol=0)


def test_intensities_are_non_negative(tiny_cfg):
    model = _model(tiny_cfg)
    with torch.no_grad():
        u = model(mnn.tone(tiny_cfg, 4.0e9))
    assert float(u.min()) >= 0.0


def test_output_scales_with_drive_amplitude(tiny_cfg):
    """In the linear regime the intensity is quadratic in the drive.

    Also confirms the default 1 mT preset really is linear -- the premise the
    demux task rests on, and the baseline the vowel task has to beat.
    """
    def run(bt):
        tiny_cfg.fields.Bt = bt
        model = _model(tiny_cfg)
        with torch.no_grad():
            return model(mnn.tone(tiny_cfg, 4.0e9))

    low = run(0.5e-3)
    high = run(1.0e-3)
    ratio = (high / low.clamp_min(1e-30)).mean()
    assert float(ratio) == pytest.approx(4.0, rel=0.05)


def test_strong_drive_leaves_the_linear_regime(f32):
    """At tens of mT the response must stop being quadratic.

    This is the whole premise of the paper: the non-linearity is intrinsic to
    the LLG equation and appears once the precession angle grows past a few
    degrees. If this test fails, the "non-linear" tasks are running linear
    physics and cannot do better than a linear readout.
    """
    cfg = mnn.get_preset("tiny")
    cfg.mesh.nx = cfg.mesh.ny = 32
    cfg.material.abc_width = 4
    cfg.solver.timesteps = 120
    cfg.solver.relax_steps = 20

    def run(bt):
        cfg.fields.Bt = bt
        model = _model(cfg)
        with torch.no_grad():
            result = model.run(mnn.tone(cfg, 4.0e9))
            m0 = model.equilibrium()
        # precession angle away from equilibrium, and the probe intensities
        angle = float((result.m - m0).norm(dim=-1).max())
        return angle, model(mnn.tone(cfg, 4.0e9))

    angle_lin, u_lin = run(1e-3)
    angle_nl, u_nl = run(80e-3)

    assert angle_lin < 0.05, "1 mT should barely tilt the magnetisation"
    assert angle_nl > 0.15, "80 mT should drive a large precession angle"

    # quadratic scaling would give exactly (80/1)^2 = 6400
    ratio = float((u_nl.sum() / u_lin.sum().clamp_min(1e-30)))
    assert ratio < 0.9 * 6400, f"response still quadratic (ratio {ratio:.0f} vs 6400)"


def test_equilibrium_is_cached_and_invalidated(tiny_cfg):
    """Relaxation is expensive, and the equilibrium moves when the design moves."""
    model = _model(tiny_cfg)
    first = model.equilibrium()
    assert model.equilibrium() is first

    with torch.no_grad():
        model.geometry.rho.add_(0.5)
    assert model.equilibrium() is not first


def test_equilibrium_is_normalised_and_follows_the_bias(tiny_cfg):
    model = _model(tiny_cfg)
    m0 = model.equilibrium()
    assert torch.allclose(m0.norm(dim=-1), torch.ones_like(m0.norm(dim=-1)), atol=1e-6)
    # bias is along +y and the design starts uniform
    assert float(m0[..., 1].mean()) > 0.99


def test_run_rejects_a_batched_signal(tiny_cfg):
    model = _model(tiny_cfg)
    signal = mnn.tone(tiny_cfg, 4.0e9)
    with pytest.raises(ValueError, match="expected a"):
        model.run(torch.stack([signal, signal]))


def test_channel_count_must_match_source_count(tiny_cfg):
    model = _model(tiny_cfg)
    bad = torch.zeros(tiny_cfg.solver.timesteps, 3)
    with pytest.raises(ValueError, match="channels but"):
        model.run(bad)


def test_traces_and_snapshots(tiny_cfg):
    model = _model(tiny_cfg, n_probes=3)
    with torch.no_grad():
        result = model.run(mnn.tone(tiny_cfg, 4.0e9), record_traces=True, snapshot_every=8)

    assert result.traces.shape == (tiny_cfg.solver.timesteps, 3)
    assert torch.allclose(result.traces.sum(dim=0), result.intensities, rtol=1e-6)
    assert result.snapshots is not None and result.snapshots.shape[1:] == (*tiny_cfg.mesh.n, 3)


# ---------------------------------------------------------------------- tasks


def test_focusing_task_builds_consistently(f32):
    cfg = mnn.get_preset("tiny")
    task = mnn.build_focusing(cfg, n_probes=5)

    assert task.model.n_probes == 5
    assert task.signals.shape == (1, cfg.solver.timesteps, 1)
    assert int(task.targets[0]) == 2


def test_demux_task_has_one_probe_per_frequency(f32):
    cfg = mnn.get_preset("tiny")
    task = mnn.build_demux(cfg, freqs=[3.5e9, 4.0e9, 4.5e9])

    assert task.model.n_probes == 3
    assert task.signals.shape[0] == 3
    assert task.targets.tolist() == [0, 1, 2]


def test_task_warns_below_the_fmr(f32):
    """A drive below resonance produces no wave; the user should hear about it
    before spending an hour on a run that cannot work."""
    cfg = mnn.get_preset("tiny")
    with pytest.warns(UserWarning, match="below the"):
        mnn.build_focusing(cfg, freq=1.0e9)


def test_vowel_task_warns_in_the_linear_regime(f32):
    cfg = mnn.get_preset("tiny")
    cfg.fields.Bt = 1e-3
    cfg.solver.timesteps = 64
    with pytest.warns(UserWarning, match="linear regime"):
        mnn.build_vowels(cfg, n_per_class=3, n_train_per_class=1)


def test_vowel_task_splits_train_and_test(f32):
    cfg = mnn.get_preset("tiny")
    cfg.fields.Bt = 50e-3
    cfg.solver.timesteps = 64
    task = mnn.build_vowels(cfg, n_per_class=6, n_train_per_class=2)

    assert task.has_test_set
    assert task.signals.shape[0] == 6      # 2 per class x 3
    assert task.test_signals.shape[0] == 12
    assert task.model.n_probes == 3


def test_design_region_excludes_source_and_probe_columns(f32):
    """The optimiser must not be able to modulate the antenna or the detectors
    themselves -- that would be tuning the measurement, not the medium."""
    cfg = mnn.get_preset("tiny")
    task = mnn.build_focusing(cfg, n_probes=5)
    mask = task.model.geometry.design_mask

    source_x = cfg.material.abc_width
    assert float(mask[: source_x + 2, :].max()) == 0.0
    probe_x = task.model.probes[0].x
    assert float(mask[probe_x - 1 :, :].max()) == 0.0


# ------------------------------------------------------------------- training


def test_training_reduces_the_loss(f32):
    """End-to-end: gradients through the physics actually improve the objective."""
    cfg = mnn.get_preset("tiny")
    cfg.mesh.nx = cfg.mesh.ny = 32
    cfg.material.abc_width = 4
    cfg.solver.timesteps = 120
    cfg.solver.relax_steps = 20

    torch.manual_seed(0)
    task = mnn.build_focusing(cfg, n_probes=5)
    history = mnn.train(
        task.model, task.signals, task.targets, task.loss_fn,
        epochs=4, lr=0.2, metric_fns=task.metric_fns, verbose=False,
    )

    assert len(history.loss) == 4
    assert history.loss[-1] < history.loss[0]


def test_per_sample_accumulation_matches_batched_gradients(f32):
    """Backpropagating one sample at a time keeps memory flat; it must not
    change the gradient."""
    cfg = mnn.get_preset("tiny")
    cfg.mesh.nx = cfg.mesh.ny = 16
    cfg.material.abc_width = 3
    cfg.solver.timesteps = 16
    cfg.solver.relax_steps = 4
    cfg.solver.demag = False

    signals = torch.stack([mnn.tone(cfg, 3.8e9), mnn.tone(cfg, 4.4e9)])
    targets = torch.tensor([0, 1])

    grads = []
    for per_sample in (False, True):
        torch.manual_seed(5)
        model = _model(cfg, n_probes=2)
        model.zero_grad(set_to_none=True)
        if per_sample:
            mnn.accumulate_gradients(model, signals, targets, mnn.intensity_cross_entropy)
        else:
            mnn.intensity_cross_entropy(model(signals), targets).backward()
        grads.append(model.geometry.rho.grad.clone())

    assert torch.allclose(grads[0], grads[1], rtol=1e-6, atol=1e-12)


def test_checkpoint_round_trip(f32, tmp_path):
    cfg = mnn.get_preset("tiny")
    cfg.mesh.nx = cfg.mesh.ny = 16
    cfg.material.abc_width = 3
    cfg.solver.timesteps = 16

    torch.manual_seed(6)
    model = _model(cfg, n_probes=2)
    with torch.no_grad():
        model.geometry.rho.copy_(torch.randn_like(model.geometry.rho))

    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    history = mnn.TrainHistory()
    history.log(1.5, 2.0, accuracy=0.5)

    path = mnn.save_checkpoint(tmp_path / "ckpt.pt", model, optimizer, history, epoch=3)

    torch.manual_seed(7)
    restored = _model(cfg, n_probes=2)
    epoch, restored_history = mnn.load_checkpoint(path, restored)

    assert epoch == 3
    assert restored_history.loss == [1.5]
    assert torch.equal(restored.geometry.rho, model.geometry.rho)


def test_history_records_metrics(f32):
    history = mnn.TrainHistory()
    history.log(0.5, 1.0, accuracy=0.9)
    history.log(0.3, 1.0, accuracy=1.0)
    assert history.loss == [0.5, 0.3]
    assert history.metrics["accuracy"] == [0.9, 1.0]
    assert history.best_epoch == 1


def test_evaluate_runs_without_gradients(f32):
    cfg = mnn.get_preset("tiny")
    cfg.mesh.nx = cfg.mesh.ny = 16
    cfg.material.abc_width = 3
    cfg.solver.timesteps = 16
    cfg.solver.demag = False

    model = _model(cfg, n_probes=2)
    signals = torch.stack([mnn.tone(cfg, 4.0e9)])
    metrics = mnn.evaluate(model, signals, torch.tensor([0]), {"accuracy": mnn.accuracy})
    assert 0.0 <= metrics["accuracy"] <= 1.0
    assert model.geometry.rho.grad is None
