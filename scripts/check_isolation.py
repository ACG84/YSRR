#!/usr/bin/env python3
"""Does an off-centre bus guide buy ISOLATION, or just less coupling?

The split-readout test found that a saturated tap poisons the whole line: taps
3 and 4 never crossed their own thresholds -- 10.6 and 8.9 mT against 12 and
none-in-range -- and still lost their memory, horizon 3 against 20+ when every
tap ran sub-threshold. The corruption arrived through the bus, which every tap
shares. So the architecture the products need is a SATURATING FEEDER driving a
LINEAR delay line, and it needs the feeder's distortion not to come back out
into the line it was driven from.

The proposal under test: move the disk's bus guide off the disk axis. At zero
offset the guide is radial and drives the disk's radial response; displaced
toward the rim it meets the disk on a chord and drives azimuthal modes instead.
Coupling goes as mode overlap, so the two directions need not scale together.

REGISTERED PREDICTION, because this one has a strong prior against it.
Reciprocity says the linear transfer bus->disk equals disk->bus for the same
mode pair, and nothing in this geometry breaks reciprocity: no bias asymmetry,
no gyrotropic term the offset touches. So the honest expectation is that the
offset scales forward and backward TOGETHER and the isolation ratio comes out
flat. Measuring it costs ten minutes and settles whether route 1 is a route at
all; guessing costs a NARMA run.

What could still make the offset earn its place, and is measured here too:

  directionality  a chord-coupled disk re-radiates into the bus with a phase
                  gradient along x, which can favour one direction. That is
                  reciprocal and legal, and a feeder that dumps its distortion
                  DOWNSTREAM of every reservoir tap is as good as one that does
                  not dump it at all.
  distortion      forward coupling is at the carrier; the pollution is at 2w
                  and 3w. Overlap is frequency dependent, so an offset can in
                  principle keep the carrier and shed the harmonics. Both are
                  in the bus band -- 18 and 27 GHz against a band bottom of 8
                  GHz at 160 nm -- so they propagate if they are launched.

Three arms per offset, all lock-in against the drive:

  F     bus injector at a small amplitude -> each disk's readout ports at w.
        Forward coupling. If this collapses, the offset merely decoupled the
        disk and nothing else in the table matters.
  Blin  disk 1's BODY at the same small amplitude -> bus probes at w.
        Backward coupling in the linear regime. F/Blin is the reciprocity
        check and the isolation figure.
  Bsat  disk 1's body ABOVE its in-array threshold -> bus probes at w, 2w, 3w.
        What a saturated feeder actually puts into the shared line.

Forward and backward are read on different observables -- a guide-end m_z
average against a bus-segment m_z average -- so their ratio has no absolute
meaning. Every number is therefore also reported RELATIVE TO THE OFFSET-0 ROW,
and only those relative numbers carry the verdict.

    python scripts/check_isolation.py --device cuda --bus-alpha 0.1 \
        --lags 3 5 7 9 --bus-width 160 --v-g 542.9 --freq 9.0 \
        --offsets 0 40 60 80
"""
from __future__ import annotations
import argparse, json, math, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, torch
import magnonic_nn as mnn
from magnonic_nn.config import MU_0
from _polytap_probe import make_array, ensure_m0


def bus_probes(cfg, arr, dtype, probe_nm=20.0, feeder=0, sym_nm=600.0):
    """Short windows on the bus centreline.

    Two families, for two different questions.

    The MIDPOINT probes -- upstream of tap 1, between each tap pair, past the
    last tap -- answer "how much reaches each stretch of the line". Their
    distances from any given disk are unequal by construction, which is fine
    for that question because every offset is compared at the same geometry.

    The SYMMETRIC pair at +-sym_nm about the feeder answers directionality,
    and needs the equal spacing the midpoints do not have. On the built array
    the midpoint upstream of tap 1 sits 927 nm from it while the tap-1/tap-2
    midpoint sits 652 nm away, so a downstream/upstream ratio read off those
    two would report a 1.4x distance as though it were radiation pattern. The
    smoke test did exactly that and returned dir = 0.23 for a geometry with no
    reason to prefer either direction.

    SHORT windows on purpose. The guided wavelength here is 108.6 nm, so a
    probe approaching that length averages the carrier to nothing and would
    report a dead bus. 20 nm is 0.18 wavelengths.
    """
    yb = cfg._bus_y()
    nx, _ = cfg.grid
    x0 = -(nx - 1) / 2 * cfg.dx
    tx = [x0 + t for t in cfg.tap_x()]
    xs = [0.5 * (cfg.inject_x() + cfg.inject_len + tx[0])]
    names = ["up"]
    for k in range(len(tx) - 1):
        xs.append(0.5 * (tx[k] + tx[k + 1])); names.append(f"{k+1}-{k+2}")
    xs.append(tx[-1] + 0.5 * (cfg.radius + cfg.guide_length)); names.append("after")
    n_mid = len(xs)
    # SEVERAL symmetric pairs, not one. The two sides of a disk sit in
    # different interference environments -- downstream has the next tap as a
    # reflector 703 nm away, upstream has the injector and the end absorber --
    # so a standing wave would produce a directional-looking ratio at any
    # single distance. A genuine radiation asymmetry is the same at every
    # distance; a standing wave oscillates with it. One pair cannot tell them
    # apart and three can, for the cost of three more matvec rows.
    for d_nm in sym_nm:
        d = d_nm * 1e-9
        xs += [tx[feeder] - d, tx[feeder] + d]
        names += [f"s{d_nm:.0f}-", f"s{d_nm:.0f}+"]
    on_bus = (arr._Y - yb).abs() <= cfg.bus_width / 2
    half = probe_nm * 1e-9 / 2
    masks = torch.stack([(on_bus & ((arr._X - xp).abs() <= half)).to(dtype)
                         for xp in xs])
    return masks, names, [float((xp - x0) * 1e9) for xp in xs], n_mid


@torch.no_grad()
def lockin(arr, cfg, unit, amp_mT, freq_ghz, n_settle, n_meas, harmonics,
           probes, dtype=torch.float32, log=None):
    """Drive `unit` at amp*sin(wt); return complex amplitudes at each harmonic.

    Returns (n_harmonic, n_port + n_probe): the tap readouts first, then the
    bus probes, so one rollout answers both directions of the same arm.
    """
    amp = amp_mT * 1e-3 / MU_0
    w = 2 * math.pi * freq_ghz * 1e9
    stepper, graphed = arr.rollout.graph_stepper()
    eager = arr.rollout.rk4_step_fields
    hz = arr.h_zero
    m = arr.m0.clone()
    mz0 = arr.m0[:, :, 0, 2].clone()
    # Probe reduction as a matvec against a flattened mask, and the whole
    # accumulator kept on the device. The obvious version -- a float64 mask
    # times the float64 field, summed over the mesh, pulled to the host every
    # step -- is a 257k-cell double-precision reduction plus a sync per step on
    # a card that runs float64 at 1/32 rate, for twenty numbers.
    pf = probes.reshape(probes.shape[0], -1)
    pw = pf.sum(dim=1).clamp_min(1e-30)
    accI = accQ = None
    t0 = time.time()
    for k in range(n_settle + n_meas):
        tk = k * cfg.dt
        h0 = hz + unit * (amp * math.sin(w * tk))
        hh = hz + unit * (amp * math.sin(w * (tk + 0.5 * cfg.dt)))
        h1 = hz + unit * (amp * math.sin(w * (tk + cfg.dt)))
        if graphed:
            try:
                torch.compiler.cudagraph_mark_step_begin()
                m = stepper(m, h0, hh, hh, h1).clone()
            except Exception as e:
                if log:
                    log(f"  compiled step failed ({type(e).__name__}); eager")
                graphed, stepper = False, eager
                m = stepper(m, h0, hh, hh, h1)
        else:
            m = stepper(m, h0, hh, hh, h1)
        if k >= n_settle:
            p = arr.port_signals(m)
            b = (pf @ (m[:, :, 0, 2] - mz0).reshape(-1)) / pw
            v = torch.cat([p, b])
            if accI is None:
                accI = torch.zeros(len(harmonics), v.shape[0], dtype=v.dtype,
                                   device=v.device)
                accQ = torch.zeros_like(accI)
            for j, h in enumerate(harmonics):
                accI[j] += v * math.cos(h * w * tk)
                accQ[j] += v * math.sin(h * w * tk)
    if log:
        log(f"  {n_settle+n_meas} steps in {time.time()-t0:.0f}s")
    A = (accI.double() + 1j * accQ.double()) / n_meas
    return A.cpu().numpy()


def run_offset(a, off_nm, outdir, dtype):
    n_win = int(round(a.cycles / (a.freq * 1e9) / 1e-12))
    cfg, arr, run, _ = make_array(
        a.n_taps, a.lags, a.gap, 0.0, a.tap_alpha, 2 * n_win + 8, dtype,
        bus_alpha_mult=a.bus_alpha, bus_guide_width_nm=a.bus_guide_width,
        bus_width_nm=a.bus_width, steps_per_frame=a.steps_per_frame,
        v_g=a.v_g, bus_guide_offset_nm=off_nm,
        barrier=(a.barrier_len, a.barrier_alpha, a.barrier_after))
    log = lambda s: print(f"  {s}", flush=True)
    print(f"\n===== offset {off_nm:g} nm =====\n{run}\n"
          f"mesh {cfg.grid[0]}x{cfg.grid[1]}, cells {int(arr.mask.sum())}, "
          f"window {n_win} steps = {a.cycles:g} cycles", flush=True)
    probes, pnames, pxs, n_mid = bus_probes(cfg, arr, dtype, a.probe_nm,
                                            a.feeder, a.sym_nm)
    # A symmetric probe inside the barrier reads the field in an absorber and
    # reports it as radiation pattern. With the route-2 layout -- feeder last,
    # barrier between taps 3 and 4 -- the default 600 nm standoff lands 51 nm
    # inside a 400 nm barrier, so this is the configuration that would fail
    # silently rather than a hypothetical one.
    if a.barrier_len > 0 and 0 <= a.barrier_after < a.n_taps - 1:
        tx = cfg.tap_x()
        xb = 0.5 * (tx[a.barrier_after] + tx[a.barrier_after + 1]) * 1e9
        half = a.barrier_len / 2
        for i in range(n_mid, len(pxs)):
            if abs(pxs[i] - xb) < half:
                raise SystemExit(
                    f"probe {pnames[i]} at {pxs[i]:.0f} nm is inside the "
                    f"barrier\n({xb:.0f} +- {half:.0f} nm). Lower --sym-nm "
                    f"below {abs(tx[a.feeder]*1e9 - xb) - half:.0f} nm or move "
                    f"the barrier.")
    # Only now: the relax is ~100 s of GPU per point, and a geometry check that
    # runs after it is a check that costs what it was meant to save.
    ensure_m0(arr, outdir, run, relax_steps=a.relax_steps, dtype=dtype, log=log)
    npr = arr.n_readout
    n_port = npr * a.n_taps

    bus_unit = torch.zeros(*arr.mask.shape[:3], 3, dtype=dtype)
    bus_unit[:, :, 0, 2] += arr.inject_mask
    disk_unit = torch.zeros(*arr.mask.shape[:3], 3, dtype=dtype)
    disk_unit[:, :, 0, 0] += arr.disk_masks[a.feeder]

    log("arm F: bus -> disks")
    F = lockin(arr, cfg, bus_unit, a.amp_lin, a.freq, n_win, n_win, [1],
               probes, dtype, log)
    log("arm Blin: disk -> bus, linear")
    Bl = lockin(arr, cfg, disk_unit, a.amp_lin, a.freq, n_win, n_win, [1],
                probes, dtype, log)
    log("arm Bsat: disk -> bus, above threshold")
    Bs = lockin(arr, cfg, disk_unit, a.amp_sat, a.freq, n_win, n_win,
                [1, 2, 3], probes, dtype, log)

    def tap_mag(A, k):
        return float(np.abs(A[k * npr:(k + 1) * npr]).mean())

    fwd = [tap_mag(F[0], k) for k in range(a.n_taps)]
    # Probes run upstream-to-downstream as ["up", "1-2", ... , "after"], so the
    # windows flanking tap k are exactly k and k+1.
    up, down = a.feeder, a.feeder + 1
    rec = {
        "offset_nm": off_nm, "run": run,
        "probe_names": pnames, "probe_x_nm": pxs,
        "fwd_per_tap": fwd,
        "fwd": fwd[a.feeder] / a.amp_lin,
        # Arm F's BUS probes: what the injected signal itself puts on each
        # stretch of line. This is the denominator route 2 lives or dies on --
        # the barrier is worth having only if it drops the feeder's leak
        # relative to the reservoir's own signal, and without this the leak is
        # an amplitude with nothing to be small compared to.
        "fwd_probes": [float(x) for x in np.abs(F[0, n_port:]) / a.amp_lin],
        "blin_probes": [float(x) for x in np.abs(Bl[0, n_port:]) / a.amp_lin],
        "bsat_probes_w": [float(x) for x in np.abs(Bs[0, n_port:]) / a.amp_sat],
        "bsat_probes_2w": [float(x) for x in np.abs(Bs[1, n_port:]) / a.amp_sat],
        "bsat_probes_3w": [float(x) for x in np.abs(Bs[2, n_port:]) / a.amp_sat],
        "bsat_taps_w": [tap_mag(Bs[0], k) for k in range(a.n_taps)],
        "i_down": down, "i_up": up,
    }
    # Back-leak off the SYMMETRIC probes, not the midpoints: the midpoints sit
    # at whatever distance the tap spacing leaves, and the one past the last
    # tap is only 125 nm out -- near field, not radiation. The symmetric pair
    # is a controlled 600 nm for any feeder, so `iso` means the same thing
    # whether the feeder is first (route 1) or last (route 2).
    sm, sp = n_mid, n_mid + 1
    rec["back"] = rec["blin_probes"][sp]
    rec["back_up"] = rec["blin_probes"][sm]
    rec["back_mid"] = rec["blin_probes"][down]
    rec["iso"] = rec["fwd"] / max(rec["back"], 1e-30)
    rec["iso_up"] = rec["fwd"] / max(rec["back_up"], 1e-30)
    rec["dir"] = rec["blin_probes"][sp] / max(rec["blin_probes"][sm], 1e-30)
    rec["dir_sat"] = rec["bsat_probes_w"][sp] / max(rec["bsat_probes_w"][sm], 1e-30)
    # dir at every symmetric distance: flat means radiation, oscillating means
    # a standing wave and the single-distance number means nothing.
    rec["dir_by_d"] = {
        f"{d:.0f}": rec["blin_probes"][n_mid + 2 * i + 1]
                    / max(rec["blin_probes"][n_mid + 2 * i], 1e-30)
        for i, d in enumerate(a.sym_nm)}
    rec["h2"] = rec["bsat_probes_2w"][down] / max(rec["bsat_probes_w"][down], 1e-30)
    rec["h3"] = rec["bsat_probes_3w"][down] / max(rec["bsat_probes_w"][down], 1e-30)
    print(f"  fwd/tap  " + " ".join(f"{v:.3e}" for v in fwd), flush=True)
    print(f"  Blin bus " + " ".join(f"{n}={v:.2e}"
                                    for n, v in zip(pnames, rec["blin_probes"])),
          flush=True)
    print(f"  Bsat  w  " + " ".join(f"{n}={v:.2e}"
                                    for n, v in zip(pnames, rec["bsat_probes_w"])),
          flush=True)
    print(f"  Bsat 2w  " + " ".join(f"{n}={v:.2e}"
                                    for n, v in zip(pnames, rec["bsat_probes_2w"])),
          flush=True)
    return rec


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n-taps", type=int, default=4)
    p.add_argument("--lags", type=float, nargs="+", default=[3, 5, 7, 9])
    p.add_argument("--gap", type=float, default=30.0)
    p.add_argument("--bus-guide-width", type=float, default=80.0)
    p.add_argument("--bus-width", type=float, default=160.0)
    p.add_argument("--bus-alpha", type=float, default=0.1)
    p.add_argument("--tap-alpha", type=float, default=1.0)
    p.add_argument("--v-g", type=float, default=542.9)
    p.add_argument("--freq", type=float, default=9.0)
    p.add_argument("--steps-per-frame", type=int, default=1200)
    p.add_argument("--offsets", type=float, nargs="+", default=[0, 40, 60, 80])
    p.add_argument("--feeder", type=int, default=0,
                   help="0-indexed disk driven in the backward arms")
    p.add_argument("--amp-lin", type=float, default=3.0,
                   help="small-signal amplitude, well under the 10 mT in-array\n"
                        "threshold, so F and Blin are both linear transfers")
    p.add_argument("--amp-sat", type=float, default=20.0,
                   help="above the in-array threshold; the amplitude tap 1 saw\n"
                        "in the run whose reservoir died")
    # Route 2 reuses this script rather than adding another one: put the feeder
    # LAST (--feeder 3) and a barrier before it, and the same three arms answer
    # "how much of the saturated feeder's output reaches the reservoir taps"
    # with and without the barrier.
    p.add_argument("--barrier-len", type=float, default=0.0,
                   help="length of the lossy bus segment, nm. 0 disables.")
    p.add_argument("--barrier-alpha", type=float, default=0.05)
    p.add_argument("--barrier-after", type=int, default=-1,
                   help="0-indexed tap; the barrier goes midway between it and\n"
                        "the next one. -1 disables.")
    p.add_argument("--cycles", type=float, default=9.0)
    p.add_argument("--probe-nm", type=float, default=20.0)
    p.add_argument("--sym-nm", type=float, nargs="+", default=[400, 600, 800],
                   help="half-separations of the symmetric probe pairs used for\n"
                        "directionality. Past the 280 nm guide-plus-gap near\n"
                        "field, inside the 1303 nm tap spacing. SEVERAL, so a\n"
                        "standing wave can be told from a radiation pattern.")
    p.add_argument("--relax-steps", type=int, default=8000)
    p.add_argument("--device", default="cpu")
    p.add_argument("--outdir", default="runs/isolation")
    a = p.parse_args()

    mnn.set_precision("float32"); mnn.set_device(a.device)
    dtype = torch.float32
    outdir = Path(a.outdir); outdir.mkdir(parents=True, exist_ok=True)
    if not 0 <= a.feeder < a.n_taps:
        raise SystemExit(f"--feeder {a.feeder} outside 0..{a.n_taps-1}")

    recs = []
    for off in a.offsets:
        recs.append(run_offset(a, off, outdir, dtype))
        (outdir / "isolation.json").write_text(json.dumps(recs, indent=2))

    base = recs[0]
    print(f"\n{'offset':>7} {'fwd':>10} {'rel':>6} {'back':>10} {'rel':>6} "
          f"{'iso':>10} {'rel':>6} {'sym+/-':>8} {'2w/w':>7} {'3w/w':>7}")
    for r in recs:
        print(f"{r['offset_nm']:>6.0f}n {r['fwd']:>10.3e} "
              f"{r['fwd']/max(base['fwd'],1e-30):>6.2f} "
              f"{r['back']:>10.3e} {r['back']/max(base['back'],1e-30):>6.2f} "
              f"{r['iso']:>10.3e} {r['iso']/max(base['iso'],1e-30):>6.2f} "
              f"{r['dir']:>8.2f} {r['h2']:>7.3f} {r['h3']:>7.3f}")

    # Leak against the reservoir's own signal, on the same stretch of bus. The
    # per-mT normalisation means this is the ratio at equal drive amplitudes;
    # in operation the feeder's fresh drive and the bus injection are set
    # independently, so scale by their ratio to get the operating figure.
    print(f"\nfeeder leak / injected signal, per stretch of bus "
          f"(saturating feeder, per mT each)")
    print(f"{'offset':>7} " + " ".join(f"{n:>9}" for n in recs[0]["probe_names"]))
    for r in recs:
        row = [l / max(s, 1e-30) for l, s in zip(r["bsat_probes_w"], r["fwd_probes"])]
        print(f"{r['offset_nm']:>6.0f}n " + " ".join(f"{v:>9.3f}" for v in row))

    print(f"\ndownstream/upstream leak, by probe half-separation")
    print(f"{'offset':>7} " + " ".join(f"{d:>8.0f}nm" for d in a.sym_nm)
          + "   verdict")
    for r in recs:
        vs = [r["dir_by_d"][f"{d:.0f}"] for d in a.sym_nm]
        spread = max(vs) / max(min(vs), 1e-30)
        tag = ("consistent" if spread < 1.5 else
               "SPREAD -- standing wave, not a radiation pattern")
        print(f"{r['offset_nm']:>6.0f}n " + " ".join(f"{v:>10.2f}" for v in vs)
              + f"   {tag}")
    print("A directional ratio read at one distance cannot be told from a")
    print("standing wave. Flat across distances is radiation; spread is not.")

    print(f"\nfwd  disk-{a.feeder+1} readout per mT of BUS drive (want: kept)")
    print(f"back bus downstream of disk {a.feeder+1} per mT of DISK drive (want: low)")
    print("iso  fwd/back. Cross-offset ratios only; the two have different units.")
    print(f"sym  radiated downstream over upstream, from probes {a.sym_nm:g} nm")
    print("     either side of the feeder -- equal distances, so this is")
    print("     radiation pattern rather than the midpoints' unequal spacing.")
    print("2w/w how much of what the SATURATED feeder puts into the bus is")
    print("     distortion rather than carrier.")

    best = max(recs[1:], key=lambda r: r["iso"], default=None)
    print()
    if best is None:
        print("Only one offset run; nothing to compare.")
        return 0
    gain = best["iso"] / max(base["iso"], 1e-30)
    keep = best["fwd"] / max(base["fwd"], 1e-30)
    if gain >= 1.5 and keep >= 0.3:
        print(f"Offset {best['offset_nm']:.0f} nm buys {gain:.2f}x isolation while\n"
              f"keeping {keep*100:.0f}% of the forward coupling. Reciprocity is\n"
              f"not binding here -- worth taking to a NARMA run with the feeder\n"
              f"off-centre and the reservoir taps radial.")
    elif gain >= 1.5:
        print(f"Isolation rises {gain:.2f}x but forward coupling falls to "
              f"{keep*100:.0f}%.\nThat is not isolation, it is decoupling: the "
              f"feeder can no longer be\ndriven hard enough to saturate without "
              f"raising the drive by the same\nfactor, which puts the leak back.")
    else:
        print(f"Isolation is flat ({gain:.2f}x at best) -- forward and backward "
              f"scale\ntogether, which is what reciprocity requires and what the "
              f"prediction\nregistered. Geometry alone does not separate the two "
              f"directions.\nRoute 1 is closed; go to route 2 (an absorbing "
              f"barrier between the\nfeeder and the reservoir taps, which breaks "
              f"the PATH rather than the\ncoupling) and route 3 (Damon-Eshbach, "
              f"which breaks reciprocity itself).")
    d = max(recs, key=lambda r: r["dir"])
    if d["dir"] >= 2.0:
        print(f"\nBut note the directionality: at {d['offset_nm']:.0f} nm the "
              f"feeder radiates\n{d['dir']:.1f}x more downstream than upstream. A "
              f"feeder placed AFTER every\nreservoir tap would then dump its "
              f"distortion into the absorber, not\ninto the delay line -- "
              f"isolation by layout rather than by coupling.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
