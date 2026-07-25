#!/usr/bin/env python3
"""Render a trained scatterer and the spin waves each dataset input produces.

Loads a checkpoint, replays every input in the task's dataset, and writes one
self-contained HTML page showing:

* the **design** -- the static field pattern the optimiser converged on. There
  is exactly one of these, shared by every input: that is what makes this a
  network rather than a lookup. The same fixed medium has to route every class
  correctly;
* per input, the **drive waveform and its spectrum**, so it is clear what was
  actually fed in;
* per input, the **wave field** animated, and the **time-integrated intensity**
  showing which detector the energy reached.

Frames are quantised to one byte per cell against a scale shared by all inputs,
so brightness is comparable between them, and embedded as base64.

    python scripts/render_field.py runs/demux_3f/checkpoint.pt --task demux
    python scripts/render_field.py runs/focus_check/checkpoint.pt --task focus
    python scripts/render_field.py runs/focus_check/checkpoint.pt --task focus \\
        --sweep 3.6 4.0 4.4      # probe one design at frequencies it never saw
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
from magnonic_nn.config import FieldConfig, MaterialConfig, MeshConfig, SimConfig, SolverConfig


def config_from_payload(payload) -> SimConfig:
    """Rebuild the :class:`SimConfig` a checkpoint was trained with."""
    raw = payload.get("config") or {}
    return SimConfig(
        mesh=MeshConfig(**raw["mesh"]),
        material=MaterialConfig(**raw["material"]),
        fields=FieldConfig(**raw["fields"]),
        solver=SolverConfig(**raw["solver"]),
        geometry=raw.get("geometry", "freeform"),
        precision=raw.get("precision", "float32"),
        device="cpu",
        seed=raw.get("seed", 0),
    )


def encode(array: np.ndarray) -> str:
    return base64.b64encode(array.tobytes()).decode("ascii")


def spectrum_of(signal: torch.Tensor, dt: float, fmax: float, n_bins: int = 160):
    """Amplitude spectrum of a drive waveform, resampled onto a fixed axis.

    Shown next to each input because the classes in these tasks differ by
    *spectrum*, not by anything visible in the time trace -- three vowels look
    like three similar-looking wiggles until you transform them.
    """
    wave = signal.reshape(-1).cpu().numpy()
    wave = wave - wave.mean()
    spec = np.abs(np.fft.rfft(wave * np.hanning(len(wave))))
    freqs = np.fft.rfftfreq(len(wave), d=dt)

    keep = freqs <= fmax
    freqs, spec = freqs[keep], spec[keep]
    if spec.max() > 0:
        spec = spec / spec.max()

    axis = np.linspace(0, fmax, n_bins)
    return axis, np.interp(axis, freqs, spec)


def replay(model, cfg, signal, every: int):
    """Run one input and return its raw (unquantised) fields."""
    with torch.no_grad():
        result = model.run(signal, snapshot_every=every)
        m0 = model.equilibrium()

    if result.snapshots is None:
        raise RuntimeError("no snapshots recorded; lower --every")

    wave = (result.snapshots - m0.unsqueeze(0))[:, :, :, 0, 2].cpu().numpy()
    return wave, (wave**2).sum(axis=0), result.intensities.cpu().numpy()


CSS = """
.viz-root {
  color-scheme: light;
  --surface-1: #fcfcfb; --surface-2: #f3f2ee; --plane: #f9f9f7;
  --text-primary: #0b0b0b; --text-secondary: #52514e; --muted: #898781;
  --border: rgba(11,11,11,0.10); --grid: #e1e0d9;
  --accent: #2a78d6;
  --div-neg: #2a78d6; --div-mid: #f0efec; --div-pos: #e34948;
  --seq-lo: #cde2fb; --seq-hi: #0d366b;
  --marker-src: #eb6834; --marker-probe: #0b0b0b;
  --good: #0ca30c; --critical: #d03b3b;
}
@media (prefers-color-scheme: dark) {
  :root:where(:not([data-theme="light"])) .viz-root {
    color-scheme: dark;
    --surface-1: #1a1a19; --surface-2: #232321; --plane: #0d0d0d;
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
  --surface-1: #1a1a19; --surface-2: #232321; --plane: #0d0d0d;
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
.wrap { max-width: 960px; margin: 0 auto; }
h1 { font-size: 1.45rem; font-weight: 600; margin: 0 0 4px; letter-spacing: -0.01em; }
.sub { color: var(--text-secondary); font-size: .93rem; margin: 0 0 16px; max-width: 70ch; }
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

.card {
  background: var(--surface-1); border: 1px solid var(--border);
  border-radius: 12px; padding: 14px 16px 14px;
}
.card h3 { font-size: .95rem; font-weight: 600; margin: 0 0 2px; }
.card .cap { font-size: .79rem; color: var(--text-secondary); margin: 0 0 10px; }
.two { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
       gap: 12px; align-items: start; }

.stage { position: relative; max-width: 420px; margin: 0 auto; }
canvas.field { width: 100%; height: auto; display: block; border-radius: 8px; }
.axis-note { font-size: .72rem; color: var(--muted); margin-top: 6px; display: flex;
             justify-content: space-between; }
.ramp { display: flex; align-items: center; gap: 8px; margin: 8px auto 0; max-width: 420px;
        font-size: .72rem; color: var(--muted); font-variant-numeric: tabular-nums; }
.ramp .bar { flex: 1; height: 8px; border-radius: 4px; }

/* input picker */
.tabs { display: flex; flex-wrap: wrap; gap: 8px; margin-bottom: 14px; }
.tab {
  font: inherit; font-size: .84rem; font-weight: 600; color: var(--text-secondary);
  background: var(--surface-1); border: 1px solid var(--border);
  border-radius: 999px; padding: 6px 15px; cursor: pointer; display: inline-flex;
  align-items: center; gap: 8px;
}
.tab[aria-selected="true"] { color: var(--text-primary); border-color: var(--accent);
                             background: var(--surface-2); }
.tab:focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; }
.tab .tag { font-size: .72rem; font-weight: 600; padding: 1px 7px; border-radius: 999px;
            background: var(--grid); color: var(--text-secondary); }

.inputcard { display: grid; grid-template-columns: 1fr 1fr; gap: 14px; }
@media (max-width: 620px) { .inputcard { grid-template-columns: 1fr; } }
.trace { width: 100%; height: auto; display: block; }
.small-title { font-size: .74rem; color: var(--muted); text-transform: uppercase;
               letter-spacing: .04em; margin-bottom: 2px; }

.verdict { display: flex; align-items: baseline; gap: 10px; flex-wrap: wrap; margin: 2px 0 10px; }
.verdict .big { font-size: 1.5rem; font-weight: 600; letter-spacing: -0.02em;
                font-variant-numeric: tabular-nums; }
.pill { font-size: .74rem; font-weight: 700; padding: 2px 10px; border-radius: 999px;
        display: inline-flex; align-items: center; gap: 5px; }
.pill.hit { background: color-mix(in srgb, var(--good) 16%, transparent); color: var(--good); }
.pill.miss { background: color-mix(in srgb, var(--critical) 16%, transparent); color: var(--critical); }

.controls { display: flex; align-items: center; gap: 12px; margin-top: 10px; flex-wrap: wrap;
            max-width: 420px; margin-left: auto; margin-right: auto; }
button.ctrl {
  font: inherit; font-size: .84rem; font-weight: 600; color: var(--text-primary);
  background: var(--surface-1); border: 1px solid var(--border);
  border-radius: 8px; padding: 5px 14px; cursor: pointer;
}
button.ctrl:hover { border-color: var(--accent); }
button.ctrl:focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; }
input[type=range] { flex: 1; min-width: 120px; accent-color: var(--accent); }
.clock { font-size: .8rem; color: var(--text-secondary); font-variant-numeric: tabular-nums;
         min-width: 86px; text-align: right; }
.legend { display: flex; flex-wrap: wrap; gap: 14px; margin-top: 10px; font-size: .78rem;
          color: var(--text-secondary); justify-content: center; }
.legend span { display: inline-flex; align-items: center; gap: 6px; }
.dot { width: 9px; height: 9px; border-radius: 50%; display: inline-block; }

table { border-collapse: collapse; margin-top: 6px; font-size: .82rem; width: 100%; }
th, td { text-align: right; padding: 4px 9px; border-bottom: 1px solid var(--border);
         font-variant-numeric: tabular-nums; }
th:first-child, td:first-child { text-align: left; }
th { color: var(--text-secondary); font-weight: 600; }
.bar-cell { position: relative; }
.bar-cell i { position: absolute; left: 0; top: 3px; bottom: 3px; border-radius: 2px;
              background: var(--accent); opacity: .2; }
tr.target td { font-weight: 600; }
.table-scroll { overflow-x: auto; }
details { margin-top: 14px; }
summary { cursor: pointer; font-size: .86rem; color: var(--text-secondary); padding: 2px 4px;
          border-radius: 4px; }
summary:focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; }
"""

JS = r"""
const D = DATA;
const NX = D.nx, NY = D.ny, FRAME = NX * NY;
let current = 0, timer = null;

const root = document.querySelector('.viz-root');
const cssVar = n => getComputedStyle(root).getPropertyValue(n).trim();
function hexToRgb(h) {
  h = h.replace('#', '');
  if (h.length === 3) h = h.split('').map(c => c + c).join('');
  return [parseInt(h.slice(0,2),16), parseInt(h.slice(2,4),16), parseInt(h.slice(4,6),16)];
}
const lerp = (a, b, t) => a.map((v, i) => Math.round(v + (b[i] - v) * t));

/* Diverging for signed fields: two hues through a neutral midpoint, so zero
   reads as nothing and the signs read as opposites. Sequential for intensity:
   one hue, since it is unsigned magnitude. */
function diverging() {
  const neg = hexToRgb(cssVar('--div-neg')), mid = hexToRgb(cssVar('--div-mid')),
        pos = hexToRgb(cssVar('--div-pos'));
  return t => t < 0.5 ? lerp(neg, mid, t * 2) : lerp(mid, pos, (t - 0.5) * 2);
}
function sequential() {
  const lo = hexToRgb(cssVar('--seq-lo')), hi = hexToRgb(cssVar('--seq-hi'));
  return t => lerp(lo, hi, t);
}

function paint(canvas, at, ramp) {
  const ctx = canvas.getContext('2d');
  canvas.width = NX; canvas.height = NY;
  const img = ctx.createImageData(NX, NY);
  const lut = []; for (let i = 0; i < 256; i++) lut.push(ramp(i / 255));

  for (let ix = 0; ix < NX; ix++) {
    for (let iy = 0; iy < NY; iy++) {
      const c = lut[Math.round(Math.max(0, Math.min(1, at(ix, iy))) * 255)];
      const p = ((NY - 1 - iy) * NX + ix) * 4;   // y up, matching the physical layout
      img.data[p] = c[0]; img.data[p+1] = c[1]; img.data[p+2] = c[2]; img.data[p+3] = 255;
    }
  }
  ctx.putImageData(img, 0, 0);

  const up = y => NY - 1 - y;
  ctx.fillStyle = cssVar('--marker-src');
  D.sources.forEach(s => s.x.forEach((x, i) => ctx.fillRect(x, up(s.y[i]), 1, 1)));
  ctx.strokeStyle = cssVar('--marker-probe');
  ctx.lineWidth = 0.35; ctx.globalAlpha = 0.75;
  D.probes.forEach(p => {
    ctx.beginPath();
    ctx.arc(p.x + 0.5, up(p.y) + 0.5, Math.max(p.r, 1.2), 0, Math.PI * 2);
    ctx.stroke();
  });
  ctx.globalAlpha = 1;
}

function decodeInt8(b64) {
  const bin = atob(b64);
  const out = new Int8Array(bin.length);
  for (let i = 0; i < bin.length; i++) out[i] = (bin.charCodeAt(i) << 24) >> 24;
  return out;
}
D.inputs.forEach(inp => { inp._wave = decodeInt8(inp.wave); });

/* ------------------------------------------------------------- line plots */
function linePlot(canvas, series, opts) {
  const W = 340, H = opts.height || 76, M = {t: 6, r: 6, b: 16, l: 6};
  canvas.width = W * 2; canvas.height = H * 2;         // 2x for crisp text on HiDPI
  const ctx = canvas.getContext('2d');
  ctx.setTransform(2, 0, 0, 2, 0, 0);
  ctx.clearRect(0, 0, W, H);

  const n = series.length;
  const lo = opts.symmetric ? -1 : 0, hi = 1;
  const x = i => M.l + i * (W - M.l - M.r) / (n - 1);
  const y = v => M.t + (hi - v) * (H - M.t - M.b) / (hi - lo);

  ctx.strokeStyle = cssVar('--grid'); ctx.lineWidth = 1;
  ctx.beginPath(); ctx.moveTo(M.l, y(0)); ctx.lineTo(W - M.r, y(0)); ctx.stroke();

  ctx.strokeStyle = opts.color || cssVar('--accent');
  ctx.lineWidth = opts.fill ? 1 : 1.4;
  ctx.beginPath();
  series.forEach((v, i) => i ? ctx.lineTo(x(i), y(v)) : ctx.moveTo(x(i), y(v)));
  ctx.stroke();

  if (opts.fill) {
    ctx.lineTo(x(n - 1), y(0)); ctx.lineTo(x(0), y(0)); ctx.closePath();
    ctx.globalAlpha = 0.18; ctx.fillStyle = opts.color || cssVar('--accent');
    ctx.fill(); ctx.globalAlpha = 1;
  }

  ctx.fillStyle = cssVar('--muted');
  ctx.font = '10px system-ui, sans-serif';
  (opts.ticks || []).forEach(t => {
    ctx.textAlign = 'center';
    ctx.fillText(t.label, M.l + t.at * (W - M.l - M.r), H - 4);
  });
}

/* --------------------------------------------------------------- rendering */
function renderDesign() {
  const dmax = Math.max(Math.abs(D.design_range[0]), Math.abs(D.design_range[1])) || 1;
  paint(document.getElementById('design'), (ix, iy) => 0.5 + D.design[ix * NY + iy] / (2 * dmax), diverging());
  document.getElementById('lo-design').textContent = '−' + dmax.toFixed(1) + ' mT';
  document.getElementById('hi-design').textContent = '+' + dmax.toFixed(1) + ' mT';
}

function renderInput(k) {
  current = k;
  const inp = D.inputs[k];

  document.querySelectorAll('.tab').forEach((t, i) =>
    t.setAttribute('aria-selected', String(i === k)));

  // what was actually fed in
  document.getElementById('input-name').textContent = inp.label;
  document.getElementById('input-desc').innerHTML = inp.description;
  linePlot(document.getElementById('trace'), inp.waveform,
           {symmetric: true, color: cssVar('--accent'),
            ticks: [{at: 0, label: '0'}, {at: 1, label: D.duration_ns.toFixed(1) + ' ns'}]});
  linePlot(document.getElementById('spectrum'), inp.spectrum,
           {fill: true, color: cssVar('--div-pos'),
            ticks: [{at: 0, label: '0'}, {at: 0.5, label: (D.fmax_GHz / 2).toFixed(1)},
                    {at: 1, label: D.fmax_GHz.toFixed(1) + ' GHz'}]});

  // outcome
  const hit = inp.argmax === inp.target;
  document.getElementById('verdict').innerHTML =
    `<span class="big">${(inp.probe_intensities[inp.target] * 100).toFixed(1)}%</span>
     <span style="color:var(--text-secondary);font-size:.85rem">of output on detector ${inp.target}</span>
     <span class="pill ${hit ? 'hit' : 'miss'}">${hit ? '✓ routed correctly' : '✗ routed to ' + inp.argmax}</span>`;

  const rows = inp.probe_intensities.map((v, i) =>
    `<tr class="${i === inp.target ? 'target' : ''}"><td>${i}${i === inp.target ? ' &larr; target' : ''}</td>
      <td class="bar-cell"><i style="width:${(v * 100).toFixed(1)}%"></i>${(v * 100).toFixed(2)}%</td></tr>`).join('');
  document.getElementById('probe-table').innerHTML =
    `<table><thead><tr><th>detector</th><th>share of output</th></tr></thead><tbody>${rows}</tbody></table>`;

  paint(document.getElementById('intensity'),
        (ix, iy) => Math.pow(inp.intensity[ix * NY + iy], 0.4), sequential());

  const scrub = document.getElementById('scrub');
  scrub.max = D.n_frames - 1;
  drawFrame(+scrub.value);
}

function drawFrame(f) {
  const inp = D.inputs[current];
  const base = f * FRAME;
  paint(document.getElementById('wave'),
        (ix, iy) => 0.5 + inp._wave[base + ix * NY + iy] / 254, diverging());
  document.getElementById('clock').textContent =
    ((f * D.frame_step * D.dt_ps) / 1000).toFixed(2) + ' ns';
}

/* ---------------------------------------------------------------- controls */
document.getElementById('tabs').innerHTML = D.inputs.map((inp, i) =>
  `<button class="tab" role="tab" aria-selected="${i === 0}">${inp.label}
     <span class="tag">&rarr; ${inp.target}</span></button>`).join('');
document.querySelectorAll('.tab').forEach((t, i) =>
  t.addEventListener('click', () => renderInput(i)));

const scrub = document.getElementById('scrub');
scrub.addEventListener('input', () => { stop(); drawFrame(+scrub.value); });

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

function fillRamp(id, ramp) {
  const stops = [];
  for (let i = 0; i <= 10; i++) { const c = ramp(i / 10); stops.push(`rgb(${c[0]},${c[1]},${c[2]}) ${i * 10}%`); }
  document.getElementById(id).style.background = `linear-gradient(to right, ${stops.join(',')})`;
}

function redraw() {
  fillRamp('ramp-design', diverging());
  fillRamp('ramp-wave', diverging());
  fillRamp('ramp-intensity', sequential());
  renderDesign();
  renderInput(current);
}
redraw();

new MutationObserver(redraw).observe(document.documentElement,
  {attributes: true, attributeFilter: ['data-theme']});
matchMedia('(prefers-color-scheme: dark)').addEventListener('change', redraw);
if (!matchMedia('(prefers-reduced-motion: reduce)').matches) start();
"""

BODY = """<title>__TITLE__</title>
<style>__CSS__</style>

<div class="viz-root">
<div class="wrap">
  <h1>__TITLE__</h1>
  <p class="sub">__SUBTITLE__</p>
  __META__

  <h2 class="section">The design &mdash; one pattern, shared by every input</h2>
  <div class="two">
    <div class="card">
      <h3>Scatterer</h3>
      <p class="cap">Local field offset from the __B0__ mT bias. These cells are the network's
         only trainable parameters, and they do not change between inputs. The same fixed
         medium has to route every class correctly &mdash; that is the whole trick.</p>
      <div class="stage"><canvas class="field" id="design"></canvas></div>
      <div class="ramp"><span id="lo-design"></span><div class="bar" id="ramp-design"></div><span id="hi-design"></span></div>
      <div class="legend">
        <span><i class="dot" style="background:var(--marker-src)"></i>antenna</span>
        <span><i class="dot" style="background:var(--marker-probe)"></i>detectors</span>
      </div>
    </div>
    <div class="card">
      <h3>Where the energy ends up</h3>
      <p class="cap">Time-integrated wave intensity for the selected input, on a compressed
         scale so the beam stays visible next to the focus. This is the network's output.</p>
      <div class="stage"><canvas class="field" id="intensity"></canvas></div>
      <div class="ramp"><span>0</span><div class="bar" id="ramp-intensity"></div><span>max</span></div>
      <div class="table-scroll" id="probe-table"></div>
    </div>
  </div>

  <h2 class="section">Dataset input</h2>
  <div class="tabs" id="tabs" role="tablist"></div>
  <div class="card">
    <h3 id="input-name"></h3>
    <p class="cap" id="input-desc"></p>
    <div class="verdict" id="verdict"></div>
    <div class="inputcard">
      <div>
        <div class="small-title">drive waveform</div>
        <canvas class="trace" id="trace"></canvas>
      </div>
      <div>
        <div class="small-title">its spectrum</div>
        <canvas class="trace" id="spectrum"></canvas>
      </div>
    </div>
  </div>

  <h2 class="section">Spin waves in flight</h2>
  <div class="card">
    <h3>Transverse magnetisation <span style="font-weight:400;color:var(--text-secondary)">m<sub>z</sub> &minus; m<sub>0</sub></span></h3>
    <p class="cap">One frame every __EVERY__ steps. All inputs share one colour scale, so
       brightness is comparable between them and the wave visibly builds, propagates and is
       swallowed by the absorbing boundary.</p>
    <div class="stage"><canvas class="field" id="wave"></canvas></div>
    <div class="ramp"><span>&minus;__WSCALE__</span><div class="bar" id="ramp-wave"></div><span>+__WSCALE__</span></div>
    <div class="controls">
      <button class="ctrl" id="play">Pause</button>
      <input type="range" id="scrub" min="0" value="0" step="1" aria-label="simulation time">
      <span class="clock" id="clock">0.00 ns</span>
    </div>
    <div class="axis-note" style="max-width:420px;margin:6px auto 0">
      <span>x &rarr; propagation, __WIDTH__ &micro;m</span><span>y &uarr;</span>
    </div>
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
        .replace("__SUBTITLE__", subtitle)
        .replace("__B0__", f"{cfg.fields.B0 * 1e3:g}")
        .replace("__EVERY__", str(payload["frame_step"]))
        .replace("__WSCALE__", f"{payload['wave_scale']:.4f}")
        .replace("__WIDTH__", f"{cfg.mesh.extent[0] * 1e6:.1f}")
    )


def describe(task_name, task, index, cfg):
    """Human-readable annotation of what this dataset input actually is."""
    if task_name == "demux":
        freq = task.info["freqs"][index]
        lam = mnn.wavelength(freq, cfg.fields, cfg.material, cfg.mesh.dz)
        return task.classes[index], (
            f"A continuous tone at <b>{freq / 1e9:.2f} GHz</b>, launched from the antenna as a "
            f"plane wave. Wavelength in this film is {lam * 1e9:.0f} nm "
            f"({lam / cfg.mesh.dx:.1f} cells, {cfg.mesh.extent[0] / lam:.1f} across the film). "
            f"Nothing but the frequency distinguishes it from the other inputs."
        )
    if task_name == "vowels":
        cls = task.classes[int(task.targets[index])]
        return f"vowel /{cls}/", (
            f"One synthesised token of the vowel <b>/{cls}/</b>: a harmonic comb shaped by three "
            f"formant resonances, mapped from the speech band into the film's propagating band. "
            f"Class identity lives in the <em>envelope</em> of the spectrum, not in any single "
            f"frequency &mdash; which is why a linear medium cannot separate these."
        )
    freq = task.info.get("freq", 0.0)
    return f"{freq / 1e9:.2f} GHz", (
        f"A continuous tone at <b>{freq / 1e9:.2f} GHz</b>."
    )


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("checkpoint", type=Path)
    p.add_argument("-o", "--out", type=Path, default=Path("field.html"))
    p.add_argument("--task", default="focus", choices=["focus", "demux", "vowels"])
    p.add_argument("--freq", type=float, default=4.0e9)
    p.add_argument("--freqs", type=float, nargs="+", default=None,
                   help="demux frequencies in GHz (must match training)")
    p.add_argument("--sweep", type=float, nargs="+", default=None,
                   help="probe a focus design at these frequencies in GHz, trained or not")
    p.add_argument("--inputs", type=int, nargs="+", default=None,
                   help="dataset indices to replay (default: one per class)")
    p.add_argument("--every", type=int, default=8, help="record a frame every N steps")
    p.add_argument("--probes", type=int, default=11)
    p.add_argument("--threads", type=int, default=None)
    p.add_argument("--title", default=None)
    args = p.parse_args()

    if args.threads:
        torch.set_num_threads(args.threads)

    raw = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = config_from_payload(raw)
    mnn.set_precision(cfg.precision)
    mnn.set_device("cpu")
    cfg = config_from_payload(raw)

    freqs = [f * 1e9 for f in args.freqs] if args.freqs else None
    if args.task == "focus":
        task = mnn.build_focusing(cfg, freq=args.freq, n_probes=args.probes)
    elif args.task == "demux":
        task = mnn.build_demux(cfg, freqs=freqs)
    else:
        task = mnn.build_vowels(cfg)

    model = task.model
    model.load_state_dict(raw["model_state_dict"])

    # Which inputs to replay
    if args.sweep:
        signals, labels, descs, targets = [], [], [], []
        for f in args.sweep:
            signals.append(mnn.tone(cfg, f * 1e9))
            lam = mnn.wavelength(f * 1e9, cfg.fields, cfg.material, cfg.mesh.dz)
            trained = abs(f * 1e9 - args.freq) < 1e6
            labels.append(f"{f:.2f} GHz")
            descs.append(
                f"A continuous tone at <b>{f:.2f} GHz</b> (&lambda; = {lam * 1e9:.0f} nm, "
                f"{lam / cfg.mesh.dx:.1f} cells). "
                + ("This is the frequency the design was trained on."
                   if trained else
                   "The design never saw this frequency during training.")
            )
            targets.append(int(task.targets[0]))
    else:
        indices = args.inputs
        if indices is None:
            if task.classes:
                seen, indices = set(), []
                for i, t in enumerate(task.targets.tolist()):
                    if t not in seen:
                        seen.add(t)
                        indices.append(i)
            else:
                indices = list(range(len(task.signals)))
        signals = [task.signals[i] for i in indices]
        targets = [int(task.targets[i]) for i in indices]
        labels, descs = [], []
        for i in indices:
            lab, desc = describe(args.task, task, i, cfg)
            labels.append(lab)
            descs.append(desc)

    print(f"replaying {len(signals)} input(s) from {args.checkpoint} ...")

    fmax = 6.0e9
    raw_inputs = []
    for signal, label in zip(signals, labels):
        wave, intensity, u = replay(model, cfg, signal, args.every)
        raw_inputs.append({"wave": wave, "intensity": intensity, "u": u, "signal": signal})
        print(f"  {label:>12}: detector {int(u.argmax())} takes {u.max() / u.sum() * 100:.1f}%")

    # One quantisation scale across all inputs, so brightness is comparable
    wave_scale = max(float(np.abs(r["wave"]).max()) for r in raw_inputs) or 1.0

    inputs = []
    for r, label, desc, target in zip(raw_inputs, labels, descs, targets):
        q = np.clip(np.round(r["wave"] / wave_scale * 127.0), -127, 127).astype(np.int8)
        trace = r["signal"].reshape(-1).cpu().numpy()
        step = max(len(trace) // 320, 1)
        _, spec = spectrum_of(r["signal"], cfg.solver.dt, fmax)
        u = r["u"]
        inputs.append({
            "label": label,
            "description": desc,
            "target": target,
            "argmax": int(u.argmax()),
            "wave": encode(q),
            "intensity": np.round(r["intensity"] / (r["intensity"].max() or 1.0), 5).reshape(-1).tolist(),
            "probe_intensities": (u / (u.sum() or 1.0)).tolist(),
            "waveform": np.round(trace[::step] / (np.abs(trace).max() or 1.0), 4).tolist(),
            "spectrum": np.round(spec, 4).tolist(),
        })

    design = (model.design_field_tesla().cpu().numpy() - cfg.fields.B0) * 1e3
    sources = []
    for src in model.sources:
        xs, ys = src.coordinates()
        sources.append({"x": xs.cpu().tolist(), "y": ys.cpu().tolist()})

    payload = {
        "nx": cfg.mesh.nx, "ny": cfg.mesh.ny,
        "n_frames": int(raw_inputs[0]["wave"].shape[0]),
        "frame_step": args.every,
        "dt_ps": cfg.solver.dt * 1e12,
        "duration_ns": cfg.duration * 1e9,
        "fmax_GHz": fmax / 1e9,
        "wave_scale": wave_scale,
        "design": np.round(design.reshape(-1), 4).tolist(),
        "design_range": [float(design.min()), float(design.max())],
        "sources": sources,
        "probes": [{"x": int(p.x), "y": int(p.y), "r": float(getattr(p, "r", 1.0))}
                   for p in model.probes if hasattr(p, "x")],
        "inputs": inputs,
    }

    n_hit = sum(1 for i in inputs if i["argmax"] == i["target"])
    meta = {
        "mesh": f"{cfg.mesh.nx}×{cfg.mesh.ny} @ {cfg.mesh.dx * 1e9:g} nm "
                f"({cfg.mesh.extent[0] * 1e6:.1f} × {cfg.mesh.extent[1] * 1e6:.1f} µm)",
        "rollout": f"{cfg.solver.timesteps} × {cfg.solver.dt * 1e12:g} ps = {cfg.duration * 1e9:.1f} ns",
        "drive": f"{cfg.fields.Bt * 1e3:g} mT",
        "inputs": f"{len(inputs)}",
        "epoch": str(raw.get("epoch", "?")),
    }
    subtitle = (
        "There are no neurons here. Each input enters at the antenna on the left, spin waves "
        "propagate through <em>one fixed</em> trained field pattern, and interference alone "
        f"decides which detector the energy reaches. {n_hit} of {len(inputs)} inputs land on "
        "their target detector."
    )

    title = args.title or f"Magnonic network — {len(inputs)} inputs, one design"
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(build_html(payload, cfg, title, subtitle, meta))
    print(f"wrote {args.out} ({args.out.stat().st_size / 1024:.0f} kB, "
          f"{payload['n_frames']} frames x {len(inputs)} inputs)")


if __name__ == "__main__":
    main()
