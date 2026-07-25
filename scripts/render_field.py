#!/usr/bin/env python3
"""Render a trained scatterer and the spin waves propagating through it.

Loads a checkpoint, replays the rollout, and writes a self-contained HTML page
showing three things side by side:

* the **design** -- the static field pattern the optimiser converged on, which
  is the entire "weight matrix" of this network;
* the **wave field**, animated -- the instantaneous transverse magnetisation,
  which is where you actually see the computation happen;
* the **time-integrated intensity** -- where the energy ended up, i.e. the
  network's output.

Frames are quantised to one byte per cell against a global scale and embedded
as base64, so the page stays self-contained and still scrubs smoothly.

    python scripts/render_field.py runs/focus_check/checkpoint.pt
    python scripts/render_field.py runs/demux/checkpoint.pt --task demux --input 1
"""

from __future__ import annotations

import argparse
import base64
import html
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch

import magnonic_nn as mnn
from magnonic_nn.config import MU_0, FieldConfig, MaterialConfig, MeshConfig, SimConfig, SolverConfig


def config_from_payload(payload) -> SimConfig:
    """Rebuild the :class:`SimConfig` a checkpoint was trained with."""
    raw = payload.get("config") or {}
    cfg = SimConfig(
        mesh=MeshConfig(**raw["mesh"]),
        material=MaterialConfig(**raw["material"]),
        fields=FieldConfig(**raw["fields"]),
        solver=SolverConfig(**raw["solver"]),
        geometry=raw.get("geometry", "freeform"),
        precision=raw.get("precision", "float32"),
        device="cpu",
        seed=raw.get("seed", 0),
    )
    return cfg


def quantise(frames: np.ndarray):
    """Map a stack of signed fields to int8 against one global scale.

    A single scale across all frames is the point: per-frame normalisation
    would rescale every frame to full contrast and hide the fact that the wave
    grows, propagates and is absorbed.
    """
    scale = float(np.abs(frames).max()) or 1.0
    quantised = np.clip(np.round(frames / scale * 127.0), -127, 127).astype(np.int8)
    return quantised, scale


def encode(array: np.ndarray) -> str:
    return base64.b64encode(array.tobytes()).decode("ascii")


def build_payload(model, cfg, signal, label, every: int):
    """Run the model and collect everything the page needs to draw."""
    with torch.no_grad():
        result = model.run(signal, record_traces=True, snapshot_every=every)
        m0 = model.equilibrium()

    snaps = result.snapshots
    if snaps is None:
        raise RuntimeError("no snapshots recorded; lower --every")

    # transverse (out-of-plane) component is the wave; the bias is in-plane
    wave = (snaps - m0.unsqueeze(0))[:, :, :, 0, 2].cpu().numpy()
    intensity = (wave**2).sum(axis=0)

    design = (model.design_field_tesla().cpu().numpy() - cfg.fields.B0) * 1e3  # mT offset
    u = result.intensities.cpu().numpy()

    frames, wave_scale = quantise(wave)

    sources = []
    for src in model.sources:
        xs, ys = src.coordinates()
        sources.append({"x": xs.cpu().tolist(), "y": ys.cpu().tolist()})
    probes = [{"x": int(p.x), "y": int(p.y), "r": float(getattr(p, "r", 1.0))}
              for p in model.probes if hasattr(p, "x")]

    return {
        "label": label,
        "nx": cfg.mesh.nx,
        "ny": cfg.mesh.ny,
        "dx_um": cfg.mesh.dx * 1e6,
        "n_frames": int(frames.shape[0]),
        "frame_step": every,
        "dt_ps": cfg.solver.dt * 1e12,
        "wave_scale": wave_scale,
        "wave": encode(frames),
        # Intensity is small enough to ship as plain numbers; only the frame
        # stack needs the byte-quantised path.
        "intensity_values": np.round(
            (intensity / (intensity.max() or 1.0)).reshape(-1), 5
        ).tolist(),
        "design": np.round(design, 4).tolist(),
        "design_range": [float(design.min()), float(design.max())],
        "probe_intensities": (u / (u.sum() or 1.0)).tolist(),
        "sources": sources,
        "probes": probes,
    }


CSS = """
.viz-root {
  color-scheme: light;
  --surface-1: #fcfcfb; --plane: #f9f9f7;
  --text-primary: #0b0b0b; --text-secondary: #52514e; --muted: #898781;
  --border: rgba(11,11,11,0.10); --grid: #e1e0d9;
  --accent: #2a78d6;
  --div-neg: #2a78d6; --div-mid: #f0efec; --div-pos: #e34948;
  --seq-lo: #cde2fb; --seq-hi: #0d366b;
  --marker-src: #eb6834; --marker-probe: #0b0b0b;
}
@media (prefers-color-scheme: dark) {
  :root:where(:not([data-theme="light"])) .viz-root {
    color-scheme: dark;
    --surface-1: #1a1a19; --plane: #0d0d0d;
    --text-primary: #ffffff; --text-secondary: #c3c2b7; --muted: #898781;
    --border: rgba(255,255,255,0.10); --grid: #2c2c2a;
    --accent: #3987e5;
    --div-neg: #3987e5; --div-mid: #383835; --div-pos: #e66767;
    --seq-lo: #0d366b; --seq-hi: #cde2fb;
    --marker-src: #d95926; --marker-probe: #ffffff;
  }
}
:root[data-theme="dark"] .viz-root {
  color-scheme: dark;
  --surface-1: #1a1a19; --plane: #0d0d0d;
  --text-primary: #ffffff; --text-secondary: #c3c2b7; --muted: #898781;
  --border: rgba(255,255,255,0.10); --grid: #2c2c2a;
  --accent: #3987e5;
  --div-neg: #3987e5; --div-mid: #383835; --div-pos: #e66767;
  --seq-lo: #0d366b; --seq-hi: #cde2fb;
  --marker-src: #d95926; --marker-probe: #ffffff;
}

.viz-root {
  font-family: system-ui, -apple-system, "Segoe UI", sans-serif;
  background: var(--plane); color: var(--text-primary);
  margin: 0; padding: 28px 20px 48px; line-height: 1.5;
}
.wrap { max-width: 940px; margin: 0 auto; }
h1 { font-size: 1.45rem; font-weight: 600; margin: 0 0 4px; letter-spacing: -0.01em; }
.sub { color: var(--text-secondary); font-size: .93rem; margin: 0 0 16px; max-width: 68ch; }
h2.section {
  font-size: .78rem; font-weight: 700; text-transform: uppercase; letter-spacing: .07em;
  color: var(--muted); margin: 30px 0 12px; padding-bottom: 6px; border-bottom: 1px solid var(--border);
}
.meta-row { display: flex; flex-wrap: wrap; gap: 6px; margin-bottom: 12px; }
.chip {
  font-size: .76rem; color: var(--text-secondary); background: var(--surface-1);
  border: 1px solid var(--border); border-radius: 999px; padding: 2px 10px;
}
.chip b { color: var(--text-primary); font-weight: 600; margin-right: 4px; }

.cards { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 12px; }
.card {
  background: var(--surface-1); border: 1px solid var(--border);
  border-radius: 12px; padding: 14px 14px 12px;
}
.card h3 { font-size: .95rem; font-weight: 600; margin: 0 0 2px; }
.card .cap { font-size: .79rem; color: var(--text-secondary); margin: 0 0 10px; }
.stage { position: relative; max-width: 560px; margin: 0 auto; }
canvas { width: 100%; height: auto; display: block; border-radius: 8px; image-rendering: auto; }
.card.wide .ramp, .card.wide .controls, .card.wide .axis-note { max-width: 560px; margin-left: auto; margin-right: auto; }
.axis-note { font-size: .72rem; color: var(--muted); margin-top: 6px; display: flex;
             justify-content: space-between; font-variant-numeric: tabular-nums; }

.ramp { display: flex; align-items: center; gap: 8px; margin-top: 8px;
        font-size: .72rem; color: var(--muted); font-variant-numeric: tabular-nums; }
.ramp .bar { flex: 1; height: 8px; border-radius: 4px; }

.controls { display: flex; align-items: center; gap: 12px; margin-top: 10px; flex-wrap: wrap; }
button {
  font: inherit; font-size: .84rem; font-weight: 600; color: var(--text-primary);
  background: var(--surface-1); border: 1px solid var(--border);
  border-radius: 8px; padding: 5px 14px; cursor: pointer;
}
button:hover { border-color: var(--accent); }
button:focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; }
input[type=range] { flex: 1; min-width: 140px; accent-color: var(--accent); }
.clock { font-size: .8rem; color: var(--text-secondary); font-variant-numeric: tabular-nums;
         min-width: 92px; text-align: right; }

.legend { display: flex; flex-wrap: wrap; gap: 14px; margin-top: 10px; font-size: .78rem;
          color: var(--text-secondary); }
.legend span { display: inline-flex; align-items: center; gap: 6px; }
.dot { width: 9px; height: 9px; border-radius: 50%; display: inline-block; }

table { border-collapse: collapse; margin-top: 8px; font-size: .82rem; width: 100%; }
th, td { text-align: right; padding: 4px 10px; border-bottom: 1px solid var(--border);
         font-variant-numeric: tabular-nums; }
th:first-child, td:first-child { text-align: left; }
th { color: var(--text-secondary); font-weight: 600; }
.bar-cell { position: relative; }
.bar-cell i { position: absolute; left: 0; top: 3px; bottom: 3px; background: var(--accent);
              opacity: .18; border-radius: 2px; }
details { margin-top: 16px; }
summary { cursor: pointer; font-size: .86rem; color: var(--text-secondary); padding: 2px 4px;
          border-radius: 4px; }
summary:focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; }
.table-scroll { overflow-x: auto; }
@media (prefers-reduced-motion: reduce) { .autoplay-off { display: none; } }
"""

JS = r"""
const D = DATA;
const NX = D.nx, NY = D.ny;

function cssVar(name) {
  return getComputedStyle(document.querySelector('.viz-root')).getPropertyValue(name).trim();
}
function hexToRgb(h) {
  h = h.replace('#', '');
  if (h.length === 3) h = h.split('').map(c => c + c).join('');
  return [parseInt(h.slice(0,2),16), parseInt(h.slice(2,4),16), parseInt(h.slice(4,6),16)];
}
const lerp = (a, b, t) => a.map((v, i) => Math.round(v + (b[i] - v) * t));

/* Diverging: two hues through a neutral midpoint, so zero reads as "nothing"
   and the two signs read as opposites. Never a rainbow. */
function divergingRamp() {
  const neg = hexToRgb(cssVar('--div-neg'));
  const mid = hexToRgb(cssVar('--div-mid'));
  const pos = hexToRgb(cssVar('--div-pos'));
  return t => t < 0.5 ? lerp(neg, mid, t * 2) : lerp(mid, pos, (t - 0.5) * 2);
}
/* Sequential: one hue, light to dark (inverted on the dark surface). */
function sequentialRamp() {
  const lo = hexToRgb(cssVar('--seq-lo'));
  const hi = hexToRgb(cssVar('--seq-hi'));
  return t => lerp(lo, hi, t);
}

function paint(canvas, values, ramp, opts) {
  opts = opts || {};
  const ctx = canvas.getContext('2d');
  canvas.width = NX; canvas.height = NY;
  const img = ctx.createImageData(NX, NY);
  const lut = [];
  for (let i = 0; i < 256; i++) lut.push(ramp(i / 255));

  for (let ix = 0; ix < NX; ix++) {
    for (let iy = 0; iy < NY; iy++) {
      const t = Math.max(0, Math.min(1, values(ix, iy)));
      const c = lut[Math.round(t * 255)];
      // draw with y up, matching the physical layout rather than array order
      const p = ((NY - 1 - iy) * NX + ix) * 4;
      img.data[p] = c[0]; img.data[p+1] = c[1]; img.data[p+2] = c[2]; img.data[p+3] = 255;
    }
  }
  ctx.putImageData(img, 0, 0);

  if (opts.markers !== false) drawMarkers(canvas);
}

function drawMarkers(canvas) {
  const ctx = canvas.getContext('2d');
  const up = y => NY - 1 - y;
  ctx.fillStyle = cssVar('--marker-src');
  D.sources.forEach(s => s.x.forEach((x, i) => ctx.fillRect(x, up(s.y[i]), 1, 1)));

  ctx.strokeStyle = cssVar('--marker-probe');
  ctx.lineWidth = 0.35;
  ctx.globalAlpha = 0.75;
  D.probes.forEach(p => {
    ctx.beginPath();
    ctx.arc(p.x + 0.5, up(p.y) + 0.5, Math.max(p.r, 1.2), 0, Math.PI * 2);
    ctx.stroke();
  });
  ctx.globalAlpha = 1;
}

/* decode base64 -> Int8Array of shape (frames, nx, ny) */
function decodeInt8(b64) {
  const bin = atob(b64);
  const out = new Int8Array(bin.length);
  for (let i = 0; i < bin.length; i++) out[i] = (bin.charCodeAt(i) << 24) >> 24;
  return out;
}
const WAVE = decodeInt8(D.wave);
const FRAME = NX * NY;

function render() {
  const div = divergingRamp(), seq = sequentialRamp();

  // design: diverging about the bias field
  const dmax = Math.max(Math.abs(D.design_range[0]), Math.abs(D.design_range[1])) || 1;
  paint(document.getElementById('design'),
        (ix, iy) => 0.5 + D.design[ix][iy] / (2 * dmax), div);
  document.getElementById('lo-design').textContent = '\u2212' + dmax.toFixed(1) + ' mT';
  document.getElementById('hi-design').textContent = '+' + dmax.toFixed(1) + ' mT';

  // integrated intensity: sequential magnitude, gamma-compressed so the
  // low-level structure of the beam is visible next to the bright focus
  paint(document.getElementById('intensity'),
        (ix, iy) => Math.pow(D.intensity_values[ix * NY + iy], 0.4), seq);

  drawFrame(+document.getElementById('scrub').value);
}

function drawFrame(f) {
  const div = divergingRamp();
  const base = f * FRAME;
  paint(document.getElementById('wave'),
        (ix, iy) => 0.5 + WAVE[base + ix * NY + iy] / 254, div);
  const ps = f * D.frame_step * D.dt_ps;
  document.getElementById('clock').textContent = (ps / 1000).toFixed(2) + ' ns';
}

/* controls */
const scrub = document.getElementById('scrub');
scrub.max = D.n_frames - 1;
scrub.addEventListener('input', () => { stop(); drawFrame(+scrub.value); });

let timer = null;
const playBtn = document.getElementById('play');
function stop() { if (timer) { clearInterval(timer); timer = null; playBtn.textContent = 'Play'; } }
function start() {
  playBtn.textContent = 'Pause';
  timer = setInterval(() => {
    scrub.value = (+scrub.value + 1) % D.n_frames;
    drawFrame(+scrub.value);
  }, 70);
}
playBtn.addEventListener('click', () => timer ? stop() : start());

/* probe table */
const rows = D.probe_intensities.map((v, i) => {
  const best = v === Math.max(...D.probe_intensities);
  return `<tr><td>${i}${best ? ' &larr; strongest' : ''}</td>
    <td class="bar-cell"><i style="width:${(v * 100).toFixed(1)}%"></i>${(v * 100).toFixed(2)}%</td></tr>`;
}).join('');
document.getElementById('probe-table').innerHTML =
  `<table><thead><tr><th>probe</th><th>share of output intensity</th></tr></thead><tbody>${rows}</tbody></table>`;

/* ramp swatches */
function fillRamp(id, ramp) {
  const el = document.getElementById(id);
  const stops = [];
  for (let i = 0; i <= 10; i++) {
    const c = ramp(i / 10);
    stops.push(`rgb(${c[0]},${c[1]},${c[2]}) ${i * 10}%`);
  }
  el.style.background = `linear-gradient(to right, ${stops.join(',')})`;
}

function redraw() {
  fillRamp('ramp-design', divergingRamp());
  fillRamp('ramp-wave', divergingRamp());
  fillRamp('ramp-intensity', sequentialRamp());
  render();
}
redraw();

// the viewer's theme toggle restyles the page; the canvases have to be repainted
new MutationObserver(redraw).observe(document.documentElement, {attributes: true, attributeFilter: ['data-theme']});
matchMedia('(prefers-color-scheme: dark)').addEventListener('change', redraw);

if (!matchMedia('(prefers-reduced-motion: reduce)').matches) start();
"""

BODY = """<title>__TITLE__</title>
<style>__CSS__</style>

<div class="viz-root">
<div class="wrap">
  <h1>__TITLE__</h1>
  <p class="sub">There are no neurons here. The drive enters at the antenna on the left,
     spin waves propagate through the trained field pattern, and interference alone decides
     which detector the energy reaches. __SUBTITLE__</p>
  __META__

  <h2 class="section">The trained design</h2>
  <div class="cards">
    <div class="card">
      <h3>Scatterer</h3>
      <p class="cap">Local field offset from the __B0__ mT bias. This pattern <em>is</em> the
         network's weights &mdash; every degree of freedom it has.</p>
      <div class="stage"><canvas id="design"></canvas></div>
      <div class="ramp"><span id="lo-design"></span><div class="bar" id="ramp-design"></div><span id="hi-design"></span></div>
      <div class="legend">
        <span><i class="dot" style="background:var(--marker-src)"></i>antenna</span>
        <span><i class="dot" style="background:var(--marker-probe)"></i>detectors</span>
      </div>
    </div>

    <div class="card">
      <h3>Where the energy ends up</h3>
      <p class="cap">Time-integrated wave intensity over the whole rollout, on a compressed
         scale so the beam is visible alongside the focus. This is the network's output.</p>
      <div class="stage"><canvas id="intensity"></canvas></div>
      <div class="ramp"><span>0</span><div class="bar" id="ramp-intensity"></div><span>max</span></div>
      <div id="probe-table"></div>
    </div>
  </div>

  <h2 class="section">Spin waves in flight</h2>
  <div class="card wide">
    <h3>Transverse magnetisation <span style="font-weight:400;color:var(--text-secondary)">m<sub>z</sub> &minus; m<sub>0</sub></span></h3>
    <p class="cap">One frame every __EVERY__ steps. All frames share a single colour scale, so
       the wave visibly builds, propagates and is swallowed by the absorbing boundary rather
       than each frame being stretched to full contrast.</p>
    <div class="stage"><canvas id="wave"></canvas></div>
    <div class="ramp"><span>&minus;__WSCALE__</span><div class="bar" id="ramp-wave"></div><span>+__WSCALE__</span></div>
    <div class="controls">
      <button id="play">Pause</button>
      <input type="range" id="scrub" min="0" value="0" step="1" aria-label="simulation time">
      <span class="clock" id="clock">0.00 ns</span>
    </div>
    <div class="axis-note"><span>x &rarr; propagation, __WIDTH__ &micro;m</span><span>y &uarr;</span></div>
  </div>
</div>
</div>

<script>const DATA = __DATA__;</script>
<script>__JS__</script>
"""


def build_html(payload, cfg, title, subtitle, meta):
    meta_html = "<div class='meta-row'>" + "".join(
        f"<span class='chip'><b>{html.escape(k)}</b> {html.escape(v)}</span>" for k, v in meta.items()
    ) + "</div>"

    return (
        BODY.replace("__CSS__", CSS)
        .replace("__JS__", JS)
        .replace("__DATA__", json.dumps(payload))
        .replace("__META__", meta_html)
        .replace("__TITLE__", html.escape(title))
        .replace("__SUBTITLE__", html.escape(subtitle))
        .replace("__B0__", f"{cfg.fields.B0 * 1e3:g}")
        .replace("__EVERY__", str(payload["frame_step"]))
        .replace("__WSCALE__", f"{payload['wave_scale']:.4f}")
        .replace("__WIDTH__", f"{cfg.mesh.extent[0] * 1e6:.1f}")
    )


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("checkpoint", type=Path)
    p.add_argument("-o", "--out", type=Path, default=Path("field.html"))
    p.add_argument("--task", default="focus", choices=["focus", "demux", "vowels"])
    p.add_argument("--freq", type=float, default=4.0e9)
    p.add_argument("--input", type=int, default=0, help="which task input to replay")
    p.add_argument("--every", type=int, default=12, help="record a frame every N steps")
    p.add_argument("--probes", type=int, default=11)
    p.add_argument("--threads", type=int, default=None)
    p.add_argument("--title", default=None)
    args = p.parse_args()

    if args.threads:
        torch.set_num_threads(args.threads)

    payload_raw = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = config_from_payload(payload_raw)
    mnn.set_precision(cfg.precision)
    mnn.set_device("cpu")
    cfg = config_from_payload(payload_raw)  # rebuild after dtype is set

    if args.task == "focus":
        task = mnn.build_focusing(cfg, freq=args.freq, n_probes=args.probes)
        label = f"{args.freq / 1e9:.1f} GHz"
    elif args.task == "demux":
        task = mnn.build_demux(cfg)
        label = task.classes[args.input]
    else:
        task = mnn.build_vowels(cfg)
        label = task.classes[int(task.targets[args.input])]

    model = task.model
    model.load_state_dict(payload_raw["model_state_dict"])
    signal = task.signals[args.input]

    print(f"replaying {args.task} ({label}) from {args.checkpoint} ...")
    payload = build_payload(model, cfg, signal, label, args.every)

    u = np.array(payload["probe_intensities"])
    target = int(task.targets[args.input])
    meta = {
        "mesh": f"{cfg.mesh.nx}×{cfg.mesh.ny} @ {cfg.mesh.dx * 1e9:g} nm "
                f"({cfg.mesh.extent[0] * 1e6:.1f} × {cfg.mesh.extent[1] * 1e6:.1f} µm)",
        "input": label,
        "rollout": f"{cfg.solver.timesteps} × {cfg.solver.dt * 1e12:g} ps = {cfg.duration * 1e9:.1f} ns",
        "drive": f"{cfg.fields.Bt * 1e3:g} mT",
        "epoch": str(payload_raw.get("epoch", "?")),
    }
    subtitle = (f"Trained to focus on detector {target}; it now receives "
                f"{u[target] * 100:.1f}% of the total output intensity.")

    title = args.title or f"Trained magnonic scatterer — {label}"
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(build_html(payload, cfg, title, subtitle, meta))

    size_kb = args.out.stat().st_size / 1024
    print(f"wrote {args.out} ({size_kb:.0f} kB, {payload['n_frames']} frames)")
    print(f"  target probe {target}: {u[target] * 100:.2f}% of output   "
          f"(strongest = probe {int(u.argmax())})")


if __name__ == "__main__":
    main()
