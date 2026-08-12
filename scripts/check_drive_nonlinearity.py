#!/usr/bin/env python3
"""Where does the disk stop responding linearly to drive amplitude?

The capacity decomposition found the chain spending 98% of its measured capacity
on degree-1 targets, with NARMA-10's `u[n-9]*u[n]` product at exactly 0.000. The
readout was tested and cleared: a square-law detector does not supply the
products either, because squaring collapses the state from rank 21 to rank 5 by
discarding the phase that carries the lag information.

That leaves the drive. The runs so far modulate a 12 GHz carrier between 10 and
30 mT, and if the magnetisation's envelope responds linearly over that span then
the whole device is a linear filter and no readout or geometry can rescue it.

Screening this with full NARMA runs would cost hours per amplitude. The response
curve costs seconds per amplitude and answers the prior question -- IS there a
nonlinearity, and where does it start -- so the expensive runs can be aimed.

Three independent signatures, because they fail differently:

    |A|/a       amplitude compression. Flat = linear response. Bending = the
                cone angle is large enough for the LLG's own nonlinearity.
    arg(A)      the NONLINEAR FREQUENCY SHIFT, and the one that matters most
                here. A magnetic oscillator's frequency depends on amplitude, so
                drive amplitude leaks into response PHASE -- and phase is where
                this device's memory lives. Amplitude-into-phase is exactly the
                mixing that manufactures cross-lag products.
    |A_2f|/|A|  genuine second-harmonic generation, which must grow with drive
                if the dynamics are nonlinear. Note the 24 GHz READOUT tone is a
                0.99-correlated copy of the carrier (lock-in leakage), so this
                is measured against the leakage baseline at low amplitude rather
                than against zero.

And the constraint that bounds all of it: the vortex has to survive. Core
annihilation was already measured at 60 mT with lambda = +5.5, far past useful,
so this reports a stability check per amplitude and the useful window is where
nonlinearity has appeared and stability has not yet gone.

    python scripts/check_drive_nonlinearity.py
"""
from __future__ import annotations
import argparse, json, math, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np, torch
import magnonic_nn as mnn
from magnonic_nn.config import MU_0
from magnonic_nn.vortex import PortedVortexConfig, PortedVortexDisk, VortexConfig
from magnonic_nn._compat import get_device


# In-plane axis the AC drive acts along, set from --drive-axis. The saturating
# geometry needs the drive PERPENDICULAR to the static bias.
AXIS = [0]


@torch.no_grad()
def drive(disk, amp, freq, dtype, n_settle, n_meas, h_static=None):
    """Drive at constant amplitude; lock in at f and 2f over the last n_meas.

    Returns (M_f, M_2f, m_final) with M the complex CROSS-PORT MODE amplitudes.

    The decomposition is not decoration. A uniform in-plane drive couples almost
    entirely to n = +-1 by symmetry -- 77.9% of the amplitude, measured on these
    features -- so the mean across the six ports lands on n = 0 and cancels the
    response almost exactly. Averaging first would have reported the disk's
    strongest mode as silence, and the phase of that silence as the nonlinear
    frequency shift. Locking in per port and transforming afterwards is also
    exactly what the reservoir readout does, so this measures the same quantity
    the capacity decomposition was run on.
    """
    cfg = disk.cfg
    # Uniform in-plane drive over the disk BODY -- what the chain runs used, and
    # therefore the drive whose linearity is in question.
    unit = torch.zeros_like(disk.m0)
    unit[:, :, :, AXIS[0]] = disk.disk_only[:, :, :, 0].to(dtype)
    m = disk.m0.clone()
    hz = disk.h_zero if h_static is None else h_static
    n_ports = disk.port_signals(m).shape[0]
    accI = torch.zeros(2, n_ports, dtype=torch.float64)
    accQ = torch.zeros(2, n_ports, dtype=torch.float64)
    for k in range(n_settle + n_meas):
        tk = k * cfg.dt

        def h(theta, tk=tk):
            return unit * (amp * math.sin(2 * math.pi * freq * (tk + theta * cfg.dt)))

        m = disk.rollout.rk4_step(m, hz, h)
        if k >= n_settle:
            p = disk.port_signals(m).double()
            for i, w in enumerate((2 * math.pi * freq, 4 * math.pi * freq)):
                accI[i] += p * math.cos(w * tk)
                accQ[i] += p * math.sin(w * tk)
    # .cpu() first: mnn.set_device sets torch's GLOBAL default device, so
    # these accumulators are allocated on CUDA once --device cuda is used.
    # This script was hardcoded to CPU until that flag was added.
    A = ((accI + 1j * accQ) / n_meas).cpu().numpy()
    return np.fft.fft(A[0]), np.fft.fft(A[1]), m


def core_mz(m, body):
    """m_z at the core: the extremal out-of-plane cell inside the disk body.

    Mean m_z cannot tell a REVERSAL from an EXPULSION -- both drive it toward
    zero -- and those are opposite outcomes. A reversal preserves the vortex and
    flips its polarity, which is a bistable state and the basis of any
    thresholding element; an expulsion destroys the state entirely. Tracking the
    core cell separates them, so a reversal is not discarded as instability.
    """
    mz = m[:, :, 0, 2] * body
    return float(mz.flatten()[int(torch.argmax(mz.abs()))])


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--amps-mT", type=float, nargs="+",
                   default=[0.25, 0.5, 1, 2, 4, 6, 8, 10, 15, 20, 30, 40],
                   help="extended DOWN from the original 5-60 mT. The tapped-bus\n"
                        "balance caps the drive a disk can receive at ~1.6 mT,\n"
                        "so the question is no longer where nonlinearity starts\n"
                        "but whether any element has it THERE.")
    p.add_argument("--bias-mT", type=float, default=0.0,
                   help="static in-plane field on the disk. A bias displaces the\n"
                        "vortex core toward the edge, and near the annihilation\n"
                        "field the restoring potential softens -- the standard\n"
                        "way to get a large nonlinear response from a small\n"
                        "drive is to sit near a bifurcation.")
    p.add_argument("--radius", type=float, default=None,
                   help="nm. A smaller disk holds fewer spins, so the same drive\n"
                        "energy reaches a larger cone angle and the nonlinearity\n"
                        "threshold falls with volume.")
    p.add_argument("--Ms", type=float, default=None,
                   help="A/m. Cone angle goes as drive/Ms, so a low-moment\n"
                        "material is nonlinear at proportionally lower field.\n"
                        "Permalloy is 800e3; YIG is ~140e3, a 5.7x reduction.")
    p.add_argument("--init", default="vortex", choices=("vortex", "uniform"),
                   help="starting state for the relax. `vortex` is the built-in\n"
                        "ansatz; `uniform` starts fully in-plane, which a\n"
                        "SINGLE-DOMAIN candidate needs -- from a vortex ansatz\n"
                        "the relax must unwind a vortex that is not stable and\n"
                        "stalls part-way.")
    p.add_argument("--drive-axis", default="x", choices=("x", "y"),
                   help="in-plane axis the AC drive acts along. The saturating\n"
                        "geometry needs it PERPENDICULAR to --bias-mT: transverse\n"
                        "response is h/H_bias and saturates smoothly as h -> H_bias,\n"
                        "so the bias sets the threshold while damping independently\n"
                        "sets the memory. Driving PARALLEL to the bias -- which the\n"
                        "first bias run did -- only stiffens the resonance and\n"
                        "measured no change at all.")
    p.add_argument("--device", default="cpu")
    p.add_argument("--freq", type=float, default=12.0)
    p.add_argument("--settle", type=int, default=600)
    p.add_argument("--meas", type=int, default=600)
    p.add_argument("--relax-steps", type=int, default=6000)
    p.add_argument("--relax-alpha", type=float, default=0.5,
                   help="damping used during the relax only. Larger settles a\n"
                        "big soft disk faster; it does not change the energy\n"
                        "minimum it settles to.")
    p.add_argument("--relax-tol", type=float, default=2e-3,
                   help="max |dm| per 200 further relax steps for the ground\n"
                        "state to count as converged")
    p.add_argument("--dt-fs", type=float, default=None,
                   help="integration timestep in fs. Default scales with Ms.\n"
                        "The stiffest mode is exchange, h_ex = 2A/(mu0 Ms dx^2),\n"
                        "and it gets FASTER as Ms falls -- 1.04e6 A/m at Ms 800\n"
                        "but 5.91e6 at Ms 140, so f_max goes 36 -> 208 GHz and\n"
                        "dt*omega goes 0.23 -> 1.31. The 1 ps default is stable\n"
                        "at permalloy and diverges below Ms ~300 kA/m, which is\n"
                        "what the low-moment relax failures actually were.")
    p.add_argument("--cycles", type=float, default=None,
                   help="settle and measure windows in DRIVE CYCLES rather than\n"
                        "steps. Required for a frequency sweep: at fixed step\n"
                        "counts the integration window is 7.2 cycles at 12 GHz\n"
                        "but 1.2 at 2 GHz, so the sweep would change its own\n"
                        "lock-in length along with the variable under test.\n"
                        "7.2 reproduces the 600-step default at 12 GHz.")
    p.add_argument("--outdir", default="runs/drive_nonlinearity")
    a = p.parse_args()

    AXIS[0] = 0 if a.drive_axis == "x" else 1
    mnn.set_precision("float32"); mnn.set_device(a.device)
    dtype = torch.float32
    outdir = Path(a.outdir); outdir.mkdir(parents=True, exist_ok=True)

    kw = {}
    if a.radius is not None:
        kw["radius"] = a.radius * 1e-9
    if a.Ms is not None:
        kw["Ms"] = a.Ms
    # dt proportional to Ms, because the exchange field the integrator has to
    # resolve is inversely proportional to it. Holding dt at 1 ps while lowering
    # Ms is what produced the "no static minimum reachable this way" readings in
    # the material survey -- those were divergences, not physics.
    ms_eff = kw.get("Ms", VortexConfig.Ms)
    dt = (a.dt_fs * 1e-15 if a.dt_fs is not None
          else min(VortexConfig.dt, VortexConfig.dt * ms_eff / 800e3))
    kw["dt"] = dt
    cfg = PortedVortexConfig(**kw)
    # Step COUNTS are not the quantity that matters -- settle and measure
    # windows are physical durations, and the lock-in needs whole drive cycles.
    # At 12 GHz the period is 83 steps at 1 ps but 476 at 175 fs, so leaving
    # --meas at 600 would integrate 1.3 cycles instead of 7.2.
    scale = VortexConfig.dt / dt
    n_relax = int(round(a.relax_steps * scale))
    if a.cycles is not None:
        # Windows in DRIVE CYCLES, which is what the lock-in actually needs.
        # Step counts hide a second dependence: at 2 GHz the period is 500
        # steps, so --meas 600 integrates 1.2 cycles where at 12 GHz it
        # integrates 7.2, and a frequency sweep would silently change its own
        # integration length. 7.2 cycles reproduces the 600-step default.
        n_settle = int(round(a.cycles / (a.freq * 1e9) / dt))
        n_meas = n_settle
        win = f"{a.cycles:g} drive cycles"
    else:
        n_settle = int(round(a.settle * scale))
        n_meas = int(round(a.meas * scale))
        win = (f"{n_settle * dt / (1 / (a.freq * 1e9)):.1f} drive cycles "
               f"(from step counts)")
    print(f"[dt] {dt*1e15:.0f} fs "
          f"({'explicit' if a.dt_fs is not None else 'scaled from Ms'}); "
          f"settle {n_settle}, meas {n_meas}, relax {n_relax} steps; "
          f"window = {win}")
    disk = PortedVortexDisk(cfg, timesteps=n_settle + n_meas + 8, dtype=dtype)
    tag = (f"r{cfg.radius*1e9:.0f}_Ms{cfg.Ms/1e3:.0f}_dt{dt*1e15:.0f}"
           + (f"_b{a.bias_mT:g}" if a.bias_mT else "")
           + ("" if a.init == "vortex" else f"_{a.init}")
           + ("" if a.drive_axis == "x" else f"_d{a.drive_axis}"))
    m0c = outdir / f"m0_{tag}.pt"
    if m0c.exists():
        disk.m0 = torch.load(m0c, weights_only=False).to(device=get_device(), dtype=dtype)
        print(f"[m0] restored from {m0c.name}")
    else:
        t0 = time.time()
        # The bias has to be present DURING the relax, or the run begins with
        # the state moving to its true equilibrium and that transient is
        # measured as nonlinearity.
        hb = disk.h_zero
        if a.bias_mT:
            hb = disk.h_zero.clone()
            hb[:, :, :, 0] += ((a.bias_mT * 1e-3 / MU_0)
                               * disk.disk_only[:, :, :, 0].to(dtype))
        if a.init == "uniform":
            # A SINGLE-DOMAIN candidate must not start from a vortex ansatz.
            # The relax then has to unwind a vortex that is not stable, which
            # stalls part-way: Ms 300 at 100 nm sat at circulation +0.657 with
            # |dm| = 1.0078 after 20000 steps -- neither a vortex nor uniform.
            # m_x = 1 EVERYWHERE, not masked. Masking leaves m = (0,0,0)
            # outside the material, and with renormalize=True that is a
            # division by zero -- the first version produced a state with
            # circulation exactly 0 (correct) but |dm| = 1.117 and a cell at
            # m_z = -0.794 (not uniform at all). The built-in vortex ansatz
            # sets m_x = 1 everywhere for the same reason.
            m_init = torch.zeros_like(disk.h_zero)
            m_init[:, :, :, 0] = 1.0
            disk.m0 = disk.rollout.relax(m_init, hb, n_relax,
                                         a.relax_alpha)
        else:
            disk.relax(steps=n_relax, alpha_relax=a.relax_alpha)
            if a.bias_mT:
                disk.m0 = disk.rollout.relax(disk.m0, hb, n_relax,
                                             a.relax_alpha)
        torch.save(disk.m0.cpu(), m0c)
        print(f"relaxed in {time.time()-t0:.0f}s")

    # Static bias, in plane, on the disk body only.
    h_static = disk.h_zero
    if a.bias_mT:
        h_static = disk.h_zero.clone()
        h_static[:, :, :, 0] += (a.bias_mT * 1e-3 / MU_0) * disk.disk_only[:, :, :, 0].to(dtype)

    mask = disk.mask[:, :, 0].to(dtype)
    n_cells = float(mask.sum())
    # The vortex signature: mean out-of-plane component over the disk. The core
    # is a few cells of m_z = 1, so this is small and POSITIVE for an intact
    # vortex and collapses when the core is expelled.
    # mask is (nx, ny, 1) against an (nx, ny) integrand -- the same broadcast
    # that inflated the circulation. It showed here as |mean m_z| > 1, which
    # a mean of a unit-vector component cannot be.
    mz0 = float((disk.m0[:, :, 0, 2] * mask[:, :, 0]).sum() / n_cells)
    # CIRCULATION, not mean m_z, is what identifies a vortex.
    #
    # mean m_z is one-sided: it catches a uniformly OUT-OF-PLANE disk (m_z -> 1)
    # and is blind to a uniformly IN-PLANE one, which gives m_z ~ 0 exactly like
    # a vortex with a small core. Ms 140 relaxed to 0.00127 and would have
    # passed an m_z test while possibly not being a vortex at all.
    #
    # For m = (-Y, X)/r the integrand (X*m_y - Y*m_x)/r is 1 everywhere, so a
    # perfect vortex scores +-1; for any uniform state the mean over a disk is 0.
    nx_, ny_ = disk.mask.shape[0], disk.mask.shape[1]
    xs = (torch.arange(nx_, dtype=dtype) - (nx_ - 1) / 2)
    ys = (torch.arange(ny_, dtype=dtype) - (ny_ - 1) / 2)
    Xc, Yc = torch.meshgrid(xs, ys, indexing="ij")
    Rc = torch.sqrt(Xc ** 2 + Yc ** 2).clamp(min=1e-6)
    # mask is (nx, ny, 1) and the integrand is (nx, ny); multiplying them
    # broadcasts to (nx, ny, ny) and inflates the sum -- the first version of
    # this returned +23.168 for a quantity that cannot exceed 1.
    m2d = mask[:, :, 0]
    # Over the DISK BODY only. The mask includes six readout guides whose
    # magnetisation is radial and contributes no circulation, and their combined
    # area (6 x 80 x 150 nm) is more than twice the disk's own. Averaging over
    # both dilutes a perfect vortex to 31400/103400 = 0.30 -- which is what the
    # baseline returned (0.337) before this was corrected, so the first version
    # would have flagged every real vortex in the survey as not one.
    body = ((Rc <= (cfg.radius / cfg.dx)) & (m2d > 0)).to(dtype)
    n_body = float(body.sum())
    circ = float(((Xc * disk.m0[:, :, 0, 1] - Yc * disk.m0[:, :, 0, 0]) / Rc
                  * body).sum() / max(n_body, 1.0))
    cz0 = core_mz(disk.m0, body)

    # CONVERGENCE, measured rather than trusted.
    #
    # The Ms 450 / r=178 nm run reported a 0.50 mT threshold on a state that was
    # still moving: |A| came back essentially CONSTANT from 0.25 to 40 mT, so
    # |A|/a fell as 1/a and the criterion fired on a drive-INDEPENDENT
    # transient. magnum.np does warn, but the warning is a library UserWarning
    # on stderr and the script printed a threshold regardless.
    #
    # Run the relax a little further and see whether anything moves. A settled
    # ground state does not; one that is still relaxing does, and by how much.
    #
    # The probe window is a physical duration too, so it scales with dt like the
    # rest. Left at a fixed 200 steps it would shrink with dt and let a state
    # that is still moving pass the tolerance simply because it was watched for
    # less time -- the gate would go slack exactly at the low moments it was
    # added to police.
    _m1 = disk.rollout.relax(disk.m0.clone(), h_static, int(round(200 * scale)),
                             a.relax_alpha)
    dm = float((_m1 - disk.m0).abs().max())
    converged = dm <= a.relax_tol
    print(f"ground state mean m_z = {mz0:.5f}, circulation = {circ:+.3f}, "
          f"core m_z = {cz0:+.3f}")
    print(f"  relax check: max |dm| over 200 further steps = {dm:.4f} "
          f"({'converged' if converged else 'NOT CONVERGED'}, tol {a.relax_tol})")
    if not converged:
        print("  A threshold measured here reads the settling transient, not "
              "the response.\n  Raise --relax-steps or --relax-alpha.")
    # The 100 nm baseline relaxes to 0.255 on this mesh, and that is the state
    # every published number in this file was measured on, so the bar is set
    # well above it rather than at the few percent an idealised sharp core would
    # give. What this catches is the genuinely different state: r=60 and r=40 nm
    # relax to 0.93 and 0.97, near-uniform out-of-plane, because permalloy stops
    # preferring a vortex below roughly 50 nm radius at 20 nm thickness. The
    # nonlinearity of a single-domain disk is a different quantity with a
    # different threshold, and reporting one as the other would be the error.
    is_vortex = abs(circ) >= 0.5
    if not is_vortex:
        print(f"  NOT A VORTEX: circulation {circ:+.3f}, against ~+-1 for a "
              f"vortex and 0 for any\n  uniform state. A threshold reported "
              f"here is for a different element.")
    print()

    print(f"{'amp_mT':>7} {'|A|':>11} {'|A|/a':>11} {'norm':>7} "
          f"{'phase_deg':>10} {'d_phase':>8} {'2f/f':>9} {'mean_mz':>9} {'core':>9}")
    rows, ref, mode = [], None, None
    for amp_mT in a.amps_mT:
        amp = amp_mT * 1e-3 / MU_0
        t0 = time.time()
        Mf, M2f, m = drive(disk, amp, a.freq * 1e9, dtype, n_settle,
                           n_meas, h_static=h_static)
        # Fix the mode on the FIRST (smallest, most linear) amplitude and follow
        # that same one up the sweep. Re-picking the argmax per amplitude would
        # let the tracked phase jump between modes and read as a frequency shift.
        if mode is None:
            mode = int(np.argmax(np.abs(Mf)))
            print(f"tracking cross-port mode n = {mode} "
                  f"(strongest at {amp_mT:.0f} mT)\n")
        mag = float(abs(Mf[mode]))
        ph = float(np.angle(Mf[mode], deg=True))
        h2 = float(np.abs(M2f).max() / max(np.abs(Mf).max(), 1e-30))
        mz = float((m[:, :, 0, 2] * mask[:, :, 0]).sum() / n_cells)
        if ref is None:
            ref = (mag / amp_mT, ph)
        norm = (mag / amp_mT) / ref[0]
        dph = ((ph - ref[1] + 180) % 360) - 180
        # Three outcomes, not two. The core can survive, REVERSE (polarity
        # flips, vortex intact -- a bistable state, not a failure), or be
        # expelled. The previous test used mean m_z and read a reversal as a
        # loss, which would discard exactly the event a thresholding layer wants.
        cz = core_mz(m, m2d)
        lost = abs(cz) < 0.5 * abs(cz0)
        reversed_ = (not lost) and np.sign(cz) != np.sign(cz0)
        ok = not lost
        core_state = "lost" if lost else ("REVERSED" if reversed_ else "ok")
        rows.append({"amp_mT": amp_mT, "abs_A": mag, "per_mT": mag / amp_mT,
                     "normalised": norm, "phase_deg": ph, "d_phase_deg": dph,
                     "second_harmonic_ratio": h2, "mean_mz": mz,
                     "core_mz": cz, "core_state": core_state,
                     "vortex_ok": bool(ok), "seconds": round(time.time() - t0, 1)})
        print(f"{amp_mT:>7.0f} {mag:>11.4e} {mag/amp_mT:>11.4e} {norm:>7.3f} "
              f"{ph:>10.2f} {dph:>8.2f} {h2:>9.4f} {mz:>9.5f} "
              f"{core_state:>9}", flush=True)
        (outdir / f"results_{tag}_f{a.freq:g}.json").write_text(
            json.dumps(rows, indent=2))

    # The number the tapped-bus result turns on.
    #
    # The cross term is bilinear, so the delayed copy has to be a comparable
    # fraction of the disk's state, and balancing it caps the fresh drive at
    # ~1.6 mT with the best coupler measured. An element that only compresses
    # above 10 mT is linear at its own operating point, which is why every
    # product family read 0.00 across nine configurations. So: what is the
    # LOWEST stable amplitude at which this element is measurably nonlinear?
    THRESH_COMP, THRESH_PHASE, USABLE_mT = 0.05, 10.0, 1.6
    nl = [r for r in rows if r["vortex_ok"]
          and (abs(r["normalised"] - 1.0) > THRESH_COMP
               or abs(r["d_phase_deg"]) > THRESH_PHASE)]
    thr = nl[0]["amp_mT"] if nl else float("inf")
    print(f"\nNONLINEAR THRESHOLD  {thr if nl else float('nan'):.2f} mT   "
          f"(radius {cfg.radius*1e9:.0f} nm, Ms {cfg.Ms/1e3:.0f} kA/m, "
          f"bias {a.bias_mT:g} mT)")
    if not nl:
        print("  never nonlinear below the stability limit")
    elif thr <= USABLE_mT and not converged:
        print(f"  NOT USABLE despite {thr:.2f} mT: the ground state had not "
              f"converged (|dm| = {dm:.4f}),\n  so this reads a settling "
              f"transient rather than a driven response.")
    elif thr <= USABLE_mT and not is_vortex:
        # Both lines printing together is easy to misread as a success. A low
        # threshold on a state that is not a vortex is not a usable element; it
        # is a measurement of something else. Ms 300 at fixed 100 nm radius hit
        # exactly this: 0.50 mT, and circulation -0.036.
        print(f"  NOT USABLE despite {thr:.2f} mT: this threshold belongs to a "
              f"non-vortex state.\n  Lowering Ms raises the exchange length "
              f"(5.7 nm at 800 kA/m, 15.2 at 300, 32.5\n  at 140), so a "
              f"100 nm disk stops favouring flux closure. Scaling the radius\n"
              f"  with the exchange length is the test that would separate the "
              f"two.")
    elif thr <= USABLE_mT:
        print(f"  USABLE: at or below the {USABLE_mT} mT the tapped-bus balance "
              f"allows, on a\n  confirmed vortex (circulation {circ:+.3f}).")
    else:
        print(f"  short of the {USABLE_mT} mT the balance allows, by "
              f"{thr/USABLE_mT:.1f}x")

    live = [r for r in rows if r["vortex_ok"]]
    print("\ncolumns: 'norm' is |A|/a relative to the lowest amplitude, so 1.000")
    print("is perfectly linear response; 'd_phase' is the nonlinear frequency")
    print("shift, in degrees relative to the lowest amplitude.\n")
    if not live:
        print("No amplitude left the vortex intact -- nothing to conclude.")
    else:
        hi = live[-1]
        comp = abs(hi["normalised"] - 1.0)
        dph = abs(hi["d_phase_deg"])
        print(f"highest STABLE amplitude {hi['amp_mT']:.0f} mT: "
              f"compression {comp*100:.1f}%, phase shift {dph:.1f} deg, "
              f"2f/f {hi['second_harmonic_ratio']:.4f}")
        died = [r for r in rows if not r["vortex_ok"]]
        if died:
            print(f"vortex lost at and above {died[0]['amp_mT']:.0f} mT")
        if comp > 0.05 or dph > 10.0:
            print("\nThere IS usable nonlinearity below the stability limit. A\n"
                  "NARMA run driven in this window is worth its hours: the\n"
                  "amplitude-to-phase conversion is the mechanism that would\n"
                  "manufacture cross-lag products.")
        else:
            print("\nThe response is LINEAR everywhere the vortex survives.\n"
                  "Compression and phase shift both stay negligible right up to\n"
                  "the amplitude that destroys the core, so there is no window\n"
                  "where this disk is both nonlinear and stable. That is a limit\n"
                  "of the device on this task, not a tuning failure.")
    print(f"\nwrote {outdir / f'results_{tag}_f{a.freq:g}.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
