#!/usr/bin/env python3
"""Render a training run's convergence as a self-contained HTML page.

Reads a checkpoint written by :func:`magnonic_nn.save_checkpoint` and emits:

* one panel per tracked quantity, stacked on a shared epoch axis. Loss and
  contrast are in different units on different scales, so they get their own
  panels rather than a shared pair of y-axes -- two y-scales on one frame make
  any two curves look correlated, which is exactly the judgement this page
  exists to support;
* correlation statistics: Spearman's rho for monotonicity, R-squared for a
  straight-line fit, and the running rho as evidence accumulates;
* a scatter of the physical figure of merit against the loss, which is where
  you see whether the objective is a faithful proxy for it.

    python scripts/plot_convergence.py runs/focus_check/checkpoint.pt
    python scripts/plot_convergence.py runs/*/checkpoint.pt -o convergence.html
"""

from __future__ import annotations

import argparse
import html
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from magnonic_nn.stats import correlation_report, running_spearman

PANELS = [
    {
        "key": "loss",
        "title": "Training loss",
        "subtitle": "log₁₀ of leaked-to-target intensity ratio — lower is better",
        "unit": "",
        "decimals": 3,
        "zero_line": None,
    },
    {
        "key": "contrast_dB",
        "title": "Probe contrast",
        "subtitle": "target probe vs strongest competitor — higher is better",
        "unit": " dB",
        "decimals": 2,
        "zero_line": 0.0,
    },
    {
        "key": "accuracy",
        "title": "Training accuracy",
        "subtitle": "fraction of training tokens whose strongest detector is the target",
        "unit": "",
        "decimals": 3,
        "zero_line": None,
    },
    {
        "key": "grad_norm",
        "title": "Gradient norm",
        "subtitle": "L2 norm of the design gradient, before clipping — falling as the loss "
                    "flattens means approaching a minimum, not collapsing",
        "unit": "",
        "decimals": 3,
        "zero_line": 0.0,
    },
    {
        "key": "epoch_seconds",
        "title": "Wall-clock per epoch",
        "subtitle": "one forward rollout plus one backward pass",
        "unit": " s",
        "decimals": 1,
        "zero_line": None,
    },
]

LABELS = {
    "epoch": "epoch",
    "grad_norm": "grad norm",
    "loss": "loss",
    "contrast_dB": "contrast",
    "accuracy": "accuracy",
    "epoch_seconds": "seconds",
}

PAIR_NOTES = {
    ("epoch", "loss"): "Is the loss falling monotonically?",
    ("epoch", "contrast_dB"): "Is the contrast rising monotonically?",
    ("loss", "contrast_dB"): "Does minimising the loss really maximise contrast?",
    ("epoch", "accuracy"): "Is accuracy improving monotonically?",
    ("loss", "accuracy"): "Does the loss track accuracy?",
    ("loss", "grad_norm"): "Is the gradient shrinking as the loss falls?",
}


def load_run(path: Path):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    history = payload.get("history", {})

    series = {"loss": list(history.get("loss", []))}
    series.update({k: list(v) for k, v in history.get("metrics", {}).items()})
    series["epoch_seconds"] = list(history.get("epoch_seconds", []))
    series["epoch"] = list(range(len(series["loss"])))

    cfg = payload.get("config", {}) or {}
    mesh, solver, fields = cfg.get("mesh", {}), cfg.get("solver", {}), cfg.get("fields", {})

    meta = {}
    if mesh:
        meta["mesh"] = f"{mesh.get('nx')}×{mesh.get('ny')} @ {mesh.get('dx', 0) * 1e9:g} nm"
    if solver:
        meta["rollout"] = (
            f"{solver.get('timesteps')} × {solver.get('dt', 0) * 1e12:g} ps "
            f"= {solver.get('timesteps', 0) * solver.get('dt', 0) * 1e9:.1f} ns"
        )
    if fields:
        bt = fields.get("Bt", 0)
        meta["drive"] = f"{bt * 1e3:g} mT ({'non-linear' if bt >= 20e-3 else 'linear'})"
    if cfg.get("geometry"):
        meta["design"] = str(cfg["geometry"])

    pairs = [p for p in PAIR_NOTES if p[0] in series and p[1] in series and series[p[1]]]
    stats = correlation_report(series, pairs=pairs)
    for row in stats:
        row["note"] = PAIR_NOTES.get((row["x"], row["y"]), "")
        row["x_label"] = LABELS.get(row["x"], row["x"])
        row["y_label"] = LABELS.get(row["y"], row["y"])
        row["running"] = running_spearman(series[row["x"]], series[row["y"]])

    return {
        "name": path.parent.name,
        "series": series,
        "meta": meta,
        "stats": stats,
        "epoch": payload.get("epoch"),
    }


CSS = """
.viz-root {
  color-scheme: light;
  --surface-1: #fcfcfb;  --plane: #f9f9f7;
  --text-primary: #0b0b0b; --text-secondary: #52514e; --muted: #898781;
  --grid: #e1e0d9; --axis: #c3c2b7; --border: rgba(11,11,11,0.10);
  --s1: #2a78d6; --s2: #eb6834; --s3: #1baf7a;
  --seq-1: #86b6ef; --seq-2: #5598e7; --seq-3: #2a78d6;
  --seq-4: #256abf; --seq-5: #1c5cab; --seq-6: #184f95;
}
@media (prefers-color-scheme: dark) {
  :root:where(:not([data-theme="light"])) .viz-root {
    color-scheme: dark;
    --surface-1: #1a1a19; --plane: #0d0d0d;
    --text-primary: #ffffff; --text-secondary: #c3c2b7; --muted: #898781;
    --grid: #2c2c2a; --axis: #383835; --border: rgba(255,255,255,0.10);
    --s1: #3987e5; --s2: #d95926; --s3: #199e70;
    --seq-1: #184f95; --seq-2: #1c5cab; --seq-3: #256abf;
    --seq-4: #2a78d6; --seq-5: #3987e5; --seq-6: #5598e7;
  }
}
:root[data-theme="dark"] .viz-root {
  color-scheme: dark;
  --surface-1: #1a1a19; --plane: #0d0d0d;
  --text-primary: #ffffff; --text-secondary: #c3c2b7; --muted: #898781;
  --grid: #2c2c2a; --axis: #383835; --border: rgba(255,255,255,0.10);
  --s1: #3987e5; --s2: #d95926; --s3: #199e70;
  --seq-1: #184f95; --seq-2: #1c5cab; --seq-3: #256abf;
  --seq-4: #2a78d6; --seq-5: #3987e5; --seq-6: #5598e7;
}

.viz-root {
  font-family: system-ui, -apple-system, "Segoe UI", sans-serif;
  background: var(--plane); color: var(--text-primary);
  margin: 0; padding: 28px 20px 48px; line-height: 1.5;
}
.wrap { max-width: 880px; margin: 0 auto; }
h1 { font-size: 1.45rem; font-weight: 600; margin: 0 0 4px; letter-spacing: -0.01em; }
h2.section {
  font-size: .78rem; font-weight: 700; text-transform: uppercase; letter-spacing: .07em;
  color: var(--muted); margin: 30px 0 12px; padding-bottom: 6px;
  border-bottom: 1px solid var(--border);
}
.sub { color: var(--text-secondary); font-size: .93rem; margin: 0 0 18px; }
.note {
  color: var(--text-secondary); font-size: .85rem; margin: 0 0 18px;
  padding: 9px 13px; background: var(--surface-1);
  border: 1px solid var(--border); border-radius: 8px;
}
.meta-row { display: flex; flex-wrap: wrap; gap: 6px; margin-bottom: 8px; align-items: center; }
.run-name { font-size: .8rem; font-weight: 600; margin-right: 4px; }
.chip {
  font-size: .76rem; color: var(--text-secondary);
  background: var(--surface-1); border: 1px solid var(--border);
  border-radius: 999px; padding: 2px 10px;
}
.chip b { color: var(--text-primary); font-weight: 600; margin-right: 4px; }

.tiles { display: flex; flex-wrap: wrap; gap: 10px; margin: 18px 0 0; }
.tile {
  flex: 1 1 130px; background: var(--surface-1); border: 1px solid var(--border);
  border-radius: 10px; padding: 12px 14px;
}
.tile .label { font-size: .74rem; color: var(--muted); text-transform: uppercase; letter-spacing: .04em; }
.tile .value {
  font-size: 1.6rem; font-weight: 600; letter-spacing: -0.02em; margin-top: 2px;
  font-variant-numeric: tabular-nums;
}
.tile .value small { font-size: .85rem; font-weight: 500; color: var(--text-secondary); }

.stat-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 10px; }
.stat-card {
  background: var(--surface-1); border: 1px solid var(--border);
  border-radius: 10px; padding: 13px 15px;
}
.stat-card .pair { font-size: .82rem; font-weight: 600; margin-bottom: 1px; }
.stat-card .q { font-size: .77rem; color: var(--text-secondary); margin-bottom: 10px; }
.stat-card .figs { display: flex; gap: 18px; align-items: baseline; }
.stat-card .fig .k { font-size: .7rem; color: var(--muted); text-transform: uppercase; letter-spacing: .04em; }
.stat-card .fig .v { font-size: 1.28rem; font-weight: 600; font-variant-numeric: tabular-nums; }
.strength { height: 5px; border-radius: 3px; background: var(--grid); margin-top: 11px; overflow: hidden; }
.strength span { display: block; height: 100%; border-radius: 3px; background: var(--s1); }
.stat-card .p { font-size: .72rem; color: var(--muted); margin-top: 7px; font-variant-numeric: tabular-nums; }

.panel {
  background: var(--surface-1); border: 1px solid var(--border);
  border-radius: 12px; padding: 16px 8px 8px 8px; margin-bottom: 14px;
}
.panel h3 { font-size: .98rem; font-weight: 600; margin: 0 0 2px; padding: 0 10px; }
.panel .cap { font-size: .8rem; color: var(--text-secondary); margin: 0 0 6px; padding: 0 10px; }
.legend { display: flex; flex-wrap: wrap; gap: 14px; padding: 0 10px; margin: 4px 0 2px; }
.legend span { font-size: .78rem; color: var(--text-secondary); display: inline-flex; align-items: center; gap: 5px; }
.swatch { width: 10px; height: 10px; border-radius: 2px; display: inline-block; }
.chart-scroll { overflow-x: auto; }
svg { display: block; width: 100%; height: auto; }
.grid-line { stroke: var(--grid); stroke-width: 1; }
.axis-line { stroke: var(--axis); stroke-width: 1; }
.fit-line { stroke: var(--text-secondary); stroke-width: 2; stroke-dasharray: 5 4; }
.tick { fill: var(--muted); font-size: 11px; font-variant-numeric: tabular-nums; }
.axis-title { fill: var(--text-secondary); font-size: 11.5px; }
.end-label { fill: var(--text-primary); font-size: 12px; font-weight: 600; }
.fit-label { fill: var(--text-secondary); font-size: 11.5px; font-weight: 600; }
.hit { fill: transparent; }
.crosshair { stroke: var(--axis); stroke-width: 1; stroke-dasharray: 3 3; opacity: 0; }

.tip {
  position: fixed; pointer-events: none; opacity: 0; transition: opacity .08s;
  background: var(--surface-1); border: 1px solid var(--border);
  border-radius: 8px; padding: 6px 10px; font-size: .8rem;
  box-shadow: 0 4px 16px rgba(0,0,0,.16); z-index: 20; white-space: nowrap;
}
.tip .k { color: var(--text-secondary); }
.tip .v { font-weight: 600; font-variant-numeric: tabular-nums; }

details { margin-top: 18px; }
summary { cursor: pointer; font-size: .86rem; color: var(--text-secondary);
          border-radius: 4px; padding: 2px 4px; }
summary:focus-visible { outline: 2px solid var(--s1); outline-offset: 2px; }
.table-scroll { overflow-x: auto; }
table { border-collapse: collapse; margin-top: 10px; font-size: .82rem; width: 100%; }
th, td { text-align: right; padding: 4px 10px; border-bottom: 1px solid var(--border);
         font-variant-numeric: tabular-nums; white-space: nowrap; }
th:first-child, td:first-child { text-align: left; }
th { color: var(--text-secondary); font-weight: 600; }

@media (prefers-reduced-motion: reduce) { .tip { transition: none; } }
"""

JS = r"""
const W = 820, H = 190, M = {t: 14, r: 84, b: 34, l: 58};
const tip = document.getElementById('tip');
const SERIES_VARS = ['var(--s1)', 'var(--s2)', 'var(--s3)'];

function niceTicks(lo, hi, count) {
  if (lo === hi) { lo -= 1; hi += 1; }
  const raw = (hi - lo) / count;
  const mag = Math.pow(10, Math.floor(Math.log10(raw)));
  const norm = raw / mag;
  const step = (norm <= 1 ? 1 : norm <= 2 ? 2 : norm <= 5 ? 5 : 10) * mag;
  const out = [];
  for (let v = Math.ceil(lo / step) * step; v <= hi + step * 1e-9; v += step) out.push(v);
  return out;
}
const fmt = (v, d, u) => v.toFixed(d) + (u || '');
const esc = s => String(s).replace(/[&<>]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));

/* ---------------------------------------------------------------- line panel */
function linePanel(panel, colorIndex, opts) {
  opts = opts || {};
  const runs = panel.runs;
  const all = runs.flatMap(r => r.values.filter(Number.isFinite));
  if (!all.length) return '';
  const n = Math.max(...runs.map(r => r.values.length));

  let lo, hi;
  if (opts.domain) { lo = opts.domain[0]; hi = opts.domain[1]; }
  else {
    lo = Math.min(...all); hi = Math.max(...all);
    if (panel.zero_line !== null && panel.zero_line !== undefined) {
      lo = Math.min(lo, panel.zero_line); hi = Math.max(hi, panel.zero_line);
    }
    const pad = (hi - lo) * 0.12 || 1;
    lo -= pad; hi += pad;
  }

  const x = i => M.l + (n <= 1 ? 0 : i * (W - M.l - M.r) / (n - 1));
  const y = v => M.t + (hi - v) * (H - M.t - M.b) / (hi - lo);
  const xstep = Math.max(1, Math.ceil(n / 10));

  let svg = `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="${esc(panel.title)} by epoch">`;
  niceTicks(lo, hi, 4).forEach(t => {
    svg += `<line class="grid-line" x1="${M.l}" x2="${W - M.r}" y1="${y(t).toFixed(1)}" y2="${y(t).toFixed(1)}"/>`;
    svg += `<text class="tick" x="${M.l - 8}" y="${(y(t) + 4).toFixed(1)}" text-anchor="end">${fmt(t, panel.decimals, '')}</text>`;
  });
  if (panel.zero_line !== null && panel.zero_line !== undefined) {
    svg += `<line class="axis-line" x1="${M.l}" x2="${W - M.r}" y1="${y(panel.zero_line).toFixed(1)}" y2="${y(panel.zero_line).toFixed(1)}"/>`;
  }
  svg += `<line class="axis-line" x1="${M.l}" x2="${M.l}" y1="${M.t}" y2="${H - M.b}"/>`;
  svg += `<line class="axis-line" x1="${M.l}" x2="${W - M.r}" y1="${H - M.b}" y2="${H - M.b}"/>`;
  for (let i = 0; i < n; i += xstep) {
    svg += `<text class="tick" x="${x(i).toFixed(1)}" y="${H - M.b + 16}" text-anchor="middle">${i}</text>`;
  }
  svg += `<text class="axis-title" x="${(M.l + (W - M.r)) / 2}" y="${H - 4}" text-anchor="middle">epoch</text>`;

  const endLabels = [];
  runs.forEach((run, ri) => {
    const color = SERIES_VARS[(colorIndex + ri) % 3];
    const pts = [];
    run.values.forEach((v, i) => { if (Number.isFinite(v)) pts.push(`${x(i).toFixed(1)},${y(v).toFixed(1)}`); });
    svg += `<polyline points="${pts.join(' ')}" fill="none" stroke="${color}" stroke-width="2"
             stroke-linejoin="round" stroke-linecap="round"/>`;
    run.values.forEach((v, i) => {
      if (!Number.isFinite(v)) return;
      svg += `<circle cx="${x(i).toFixed(1)}" cy="${y(v).toFixed(1)}" r="4"
               fill="${color}" stroke="var(--surface-1)" stroke-width="2"/>`;
    });
    let last = run.values.length - 1;
    while (last >= 0 && !Number.isFinite(run.values[last])) last--;
    if (last >= 0) endLabels.push({y: y(run.values[last]), x: x(last), color,
                                   text: fmt(run.values[last], panel.decimals, panel.unit)});
  });

  // Series that converge to the same value would stack their end labels on top
  // of one another, hiding all but the last one drawn. Nudge them apart, and
  // tint each to its series so the pairing survives the displacement.
  endLabels.sort((a, b) => a.y - b.y);
  const MIN_GAP = 13;
  for (let i = 1; i < endLabels.length; i++) {
    if (endLabels[i].y - endLabels[i - 1].y < MIN_GAP) {
      endLabels[i].y = endLabels[i - 1].y + MIN_GAP;
    }
  }
  // Pushing apart can shove the lowest label past the axis into the tick row;
  // slide the whole group back up by however much it overflowed.
  const overflow = endLabels.length
    ? Math.max(0, endLabels[endLabels.length - 1].y - (H - M.b - 2))
    : 0;
  endLabels.forEach(l => {
    const yy = Math.max(M.t + 4, l.y - overflow);
    svg += `<text class="end-label" fill="${runs.length > 1 ? l.color : 'var(--text-primary)'}"
             x="${(l.x + 10).toFixed(1)}" y="${(yy + 4).toFixed(1)}">${l.text}</text>`;
  });

  svg += `<line class="crosshair" id="ch-${panel.key}" y1="${M.t}" y2="${H - M.b}"/>`;
  svg += `<rect class="hit" x="${M.l}" y="0" width="${W - M.l - M.r}" height="${H}"
           data-key="${panel.key}" data-n="${n}"/></svg>`;

  const legend = runs.length > 1
    ? `<div class="legend">` + runs.map((r, ri) =>
        `<span><i class="swatch" style="background:${SERIES_VARS[(colorIndex + ri) % 3]}"></i>${esc(r.name)}</span>`
      ).join('') + `</div>`
    : '';

  return `<div class="panel"><h3>${esc(panel.title)}</h3>
      <p class="cap">${panel.subtitle}</p>${legend}
      <div class="chart-scroll">${svg}</div></div>`;
}

/* ------------------------------------------------------------------ scatter */
function scatterPanel(sc) {
  const SH = 300, SM = {t: 16, r: 20, b: 44, l: 64};
  const xs = sc.x, ys = sc.y;
  const xlo = Math.min(...xs), xhi = Math.max(...xs);
  const ylo = Math.min(...ys), yhi = Math.max(...ys);
  const xp = (xhi - xlo) * 0.08 || 1, yp = (yhi - ylo) * 0.08 || 1;
  const X = v => SM.l + (v - (xlo - xp)) * (W - SM.l - SM.r) / ((xhi + xp) - (xlo - xp));
  const Y = v => SM.t + ((yhi + yp) - v) * (SH - SM.t - SM.b) / ((yhi + yp) - (ylo - yp));

  let svg = `<svg viewBox="0 0 ${W} ${SH}" role="img" aria-label="${esc(sc.y_label)} against ${esc(sc.x_label)}">`;
  niceTicks(ylo - yp, yhi + yp, 5).forEach(t => {
    svg += `<line class="grid-line" x1="${SM.l}" x2="${W - SM.r}" y1="${Y(t).toFixed(1)}" y2="${Y(t).toFixed(1)}"/>`;
    svg += `<text class="tick" x="${SM.l - 8}" y="${(Y(t) + 4).toFixed(1)}" text-anchor="end">${t.toFixed(1)}</text>`;
  });
  niceTicks(xlo - xp, xhi + xp, 5).forEach(t => {
    svg += `<text class="tick" x="${X(t).toFixed(1)}" y="${SH - SM.b + 17}" text-anchor="middle">${t.toFixed(2)}</text>`;
  });
  svg += `<line class="axis-line" x1="${SM.l}" x2="${SM.l}" y1="${SM.t}" y2="${SH - SM.b}"/>`;
  svg += `<line class="axis-line" x1="${SM.l}" x2="${W - SM.r}" y1="${SH - SM.b}" y2="${SH - SM.b}"/>`;
  svg += `<text class="axis-title" x="${(SM.l + W - SM.r) / 2}" y="${SH - 8}" text-anchor="middle">${esc(sc.x_label)}</text>`;
  svg += `<text class="axis-title" transform="translate(16,${(SM.t + SH - SM.b) / 2}) rotate(-90)" text-anchor="middle">${esc(sc.y_label)}</text>`;

  const fx0 = xlo - xp, fx1 = xhi + xp;
  svg += `<line class="fit-line" x1="${X(fx0).toFixed(1)}" y1="${Y(sc.slope * fx0 + sc.intercept).toFixed(1)}"
           x2="${X(fx1).toFixed(1)}" y2="${Y(sc.slope * fx1 + sc.intercept).toFixed(1)}"/>`;

  xs.forEach((v, i) => {
    const step = Math.min(5, Math.floor(i / Math.max(xs.length - 1, 1) * 5.999));
    svg += `<circle class="pt" cx="${X(v).toFixed(1)}" cy="${Y(ys[i]).toFixed(1)}" r="6"
             fill="var(--seq-${step + 1})" stroke="var(--surface-1)" stroke-width="2"
             data-epoch="${i}" data-x="${v}" data-y="${ys[i]}"><title>epoch ${i}</title></circle>`;
  });

  svg += `<text class="fit-label" x="${W - SM.r - 6}" y="${SM.t + 14}" text-anchor="end">R² = ${sc.r_squared.toFixed(3)}  ·  ρ = ${sc.spearman.toFixed(3)}</text>`;
  svg += `</svg>`;

  const ramp = [1, 2, 3, 4, 5, 6].map(i => `<i class="swatch" style="background:var(--seq-${i})"></i>`).join('');
  return `<div class="panel"><h3>Does the loss track the physics?</h3>
    <p class="cap">Each point is one epoch. The dashed line is the least-squares fit whose R² is quoted.</p>
    <div class="legend"><span>early ${ramp} late</span></div>
    <div class="chart-scroll">${svg}</div></div>`;
}

/* ------------------------------------------------------------------- render */
const run = DATA.runs[0];
const S = run.series;

let tiles = '';
if (S.loss && S.loss.length) {
  tiles += `<div class="tile"><div class="label">epochs</div><div class="value">${S.loss.length}</div></div>`;
  tiles += `<div class="tile"><div class="label">loss</div><div class="value">${S.loss[S.loss.length - 1].toFixed(3)}
            <small>from ${S.loss[0].toFixed(3)}</small></div></div>`;
}
if (S.contrast_dB && S.contrast_dB.length) {
  const c = S.contrast_dB;
  tiles += `<div class="tile"><div class="label">contrast</div><div class="value">${c[c.length - 1].toFixed(1)}
            <small>dB, from ${c[0].toFixed(1)}</small></div></div>`;
}
if (S.epoch_seconds && S.epoch_seconds.length) {
  // Median, not mean: an epoch that lost the CPU to something else should not
  // become the headline number for how long an epoch takes.
  const v = S.epoch_seconds.slice().sort((a, b) => a - b);
  const mid = v.length % 2 ? v[(v.length - 1) / 2] : (v[v.length / 2 - 1] + v[v.length / 2]) / 2;
  tiles += `<div class="tile"><div class="label">per epoch</div><div class="value">${mid.toFixed(0)}
            <small>s median</small></div></div>`;
}
document.getElementById('tiles').innerHTML = tiles;
document.getElementById('panels').innerHTML = DATA.panels.map((p, i) => linePanel(p, i)).join('');

/* stat cards */
document.getElementById('stat-cards').innerHTML = run.stats.map(s => {
  const pct = Math.round(Math.abs(s.spearman) * 100);
  const p = s.p_value < 1e-4 ? 'p &lt; 0.0001' : 'p = ' + s.p_value.toFixed(4);
  return `<div class="stat-card">
    <div class="pair">${esc(s.y_label)} vs ${esc(s.x_label)}</div>
    <div class="q">${esc(s.note)}</div>
    <div class="figs">
      <div class="fig"><div class="k">Spearman ρ</div><div class="v">${s.spearman.toFixed(3)}</div></div>
      <div class="fig"><div class="k">R²</div><div class="v">${s.r_squared.toFixed(3)}</div></div>
    </div>
    <div class="strength"><span style="width:${pct}%"></span></div>
    <div class="p">|ρ| = ${pct}% · n = ${s.n} · ${p}</div>
  </div>`;
}).join('');

/* running rho — all three are dimensionless correlations on the same [-1,1]
   scale, so they legitimately share one frame */
const runningPanel = {
  key: 'running_rho',
  title: 'Running Spearman ρ',
  subtitle: 'recomputed over epochs 0…k — shows whether each relationship holds up as evidence accumulates',
  unit: '', decimals: 3, zero_line: 0.0,
  runs: run.stats.map(s => ({name: `${s.y_label} vs ${s.x_label}`, values: s.running})),
};
document.getElementById('running').innerHTML = linePanel(runningPanel, 0, {domain: [-1.08, 1.08]});
if (DATA.scatter) document.getElementById('scatter').innerHTML = scatterPanel(DATA.scatter);

/* table */
const cols = DATA.panels.map(p => ({title: p.title, key: p.key, d: p.decimals, u: p.unit}));
let rows = '<table><thead><tr><th>epoch</th>' +
  cols.map(c => `<th>${esc(c.title)}</th>`).join('') +
  run.stats.map(s => `<th>ρ · ${esc(s.y_label)} vs ${esc(s.x_label)}</th>`).join('') +
  '</tr></thead><tbody>';
const maxN = Math.max.apply(null, cols.map(c => (S[c.key] || []).length));
for (let i = 0; i < maxN; i++) {
  rows += `<tr><td>${i}</td>` +
    cols.map(c => {
      const v = (S[c.key] || [])[i];
      return `<td>${v === undefined ? '&mdash;' : v.toFixed(c.d) + c.u}</td>`;
    }).join('') +
    run.stats.map(s => {
      const v = s.running[i];
      return `<td>${Number.isFinite(v) ? v.toFixed(3) : '&mdash;'}</td>`;
    }).join('') + '</tr>';
}
document.getElementById('table').innerHTML = rows + '</tbody></table>';

/* hover */
document.querySelectorAll('.hit').forEach(hit => {
  const svg = hit.ownerSVGElement;
  const key = hit.dataset.key, n = +hit.dataset.n;
  const panel = key === 'running_rho' ? runningPanel : DATA.panels.find(p => p.key === key);
  const ch = svg.querySelector('#ch-' + key);

  hit.addEventListener('pointermove', ev => {
    const box = svg.getBoundingClientRect();
    const px = (ev.clientX - box.left) / box.width * W;
    const i = Math.max(0, Math.min(n - 1, Math.round((px - M.l) / ((W - M.l - M.r) / Math.max(n - 1, 1)))));
    const cx = M.l + (n <= 1 ? 0 : i * (W - M.l - M.r) / (n - 1));
    ch.setAttribute('x1', cx); ch.setAttribute('x2', cx); ch.style.opacity = 1;

    const lines = panel.runs.map(r => !Number.isFinite(r.values[i]) ? null :
      `<div><span class="k">${panel.runs.length > 1 ? esc(r.name) + ' ' : ''}</span>
       <span class="v">${r.values[i].toFixed(panel.decimals)}${panel.unit}</span></div>`)
      .filter(Boolean).join('');
    tip.innerHTML = `<div class="k">epoch ${i}</div>${lines || '<div class="k">&mdash;</div>'}`;
    tip.style.opacity = 1;
    tip.style.left = Math.min(ev.clientX + 14, window.innerWidth - tip.offsetWidth - 8) + 'px';
    tip.style.top = (ev.clientY - 12) + 'px';
  });
  hit.addEventListener('pointerleave', () => { tip.style.opacity = 0; ch.style.opacity = 0; });
});

document.querySelectorAll('.pt').forEach(pt => {
  pt.addEventListener('pointerenter', ev => {
    tip.innerHTML = `<div class="k">epoch ${pt.dataset.epoch}</div>
      <div><span class="k">loss </span><span class="v">${(+pt.dataset.x).toFixed(3)}</span></div>
      <div><span class="k">contrast </span><span class="v">${(+pt.dataset.y).toFixed(2)} dB</span></div>`;
    tip.style.opacity = 1;
    tip.style.left = Math.min(ev.clientX + 14, window.innerWidth - tip.offsetWidth - 8) + 'px';
    tip.style.top = (ev.clientY - 12) + 'px';
  });
  pt.addEventListener('pointerleave', () => { tip.style.opacity = 0; });
});
"""

BODY = """<title>__TITLE__</title>
<style>__CSS__</style>

<div class="viz-root">
<div class="wrap">
  <h1>__TITLE__</h1>
  <p class="sub">Gradient descent through the Landau&ndash;Lifshitz&ndash;Gilbert equation
     &mdash; every point is one full micromagnetic rollout plus its adjoint.</p>
  __NOTE__
  __META__
  <div class="tiles" id="tiles"></div>

  <h2 class="section">Convergence</h2>
  <div id="panels"></div>

  <h2 class="section">Correlation statistics</h2>
  <p class="sub">Spearman&rsquo;s &rho; is a <em>rank</em> correlation: it measures monotonicity
     without assuming a shape, so a loss that falls steeply then plateaus still scores near
     &minus;1. R&sup2; is for an ordinary least-squares straight line. Where &rho; is near
     &plusmn;1 but R&sup2; sits well below it, the relationship is monotone but curved
     &mdash; the normal signature of a converging loss, not a fault. p-values use the
     t approximation and, at these sample sizes, are order-of-magnitude guides.</p>
  <div class="stat-grid" id="stat-cards"></div>
  <div id="running" style="margin-top:14px"></div>
  <div id="scatter"></div>

  <details>
    <summary>Show the numbers</summary>
    <div class="table-scroll" id="table"></div>
  </details>
</div>
</div>
<div class="tip" id="tip"></div>

<script>const DATA = __DATA__;</script>
<script>__JS__</script>
"""


def build_html(runs, title: str, note: str | None) -> str:
    panels = []
    for panel in PANELS:
        present = [r for r in runs if r["series"].get(panel["key"])]
        if present:
            panels.append({
                **panel,
                "runs": [{"name": r["name"], "values": r["series"][panel["key"]]} for r in present],
            })

    # scatter of the figure of merit against the loss, for the first run
    scatter = None
    primary = runs[0]
    row = next((s for s in primary["stats"] if (s["x"], s["y"]) == ("loss", "contrast_dB")), None)
    if row:
        n = row["n"]
        xs = primary["series"]["loss"][:n]
        ys = primary["series"]["contrast_dB"][:n]
        intercept = sum(ys) / n - row["slope"] * (sum(xs) / n)
        scatter = {
            "x": xs, "y": ys,
            "x_label": "loss", "y_label": "contrast (dB)",
            "slope": row["slope"], "intercept": intercept,
            "r_squared": row["r_squared"], "spearman": row["spearman"],
        }

    data = json.dumps({"panels": panels, "runs": runs, "scatter": scatter})

    meta_html = ""
    for run in runs:
        items = "".join(
            f"<span class='chip'><b>{html.escape(k)}</b> {html.escape(v)}</span>"
            for k, v in run["meta"].items()
        )
        label = f"<span class='run-name'>{html.escape(run['name'])}</span>" if len(runs) > 1 else ""
        meta_html += f"<div class='meta-row'>{label}{items}</div>"

    note_html = f"<p class='note'>{html.escape(note)}</p>" if note else ""

    return (
        BODY.replace("__CSS__", CSS)
        .replace("__JS__", JS)
        .replace("__DATA__", data)
        .replace("__NOTE__", note_html)
        .replace("__META__", meta_html)
        .replace("__TITLE__", html.escape(title))
    )


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("checkpoints", nargs="+", type=Path)
    p.add_argument("-o", "--out", type=Path, default=Path("convergence.html"))
    p.add_argument("--title", default="Magnonic network training")
    p.add_argument("--note", default=None, help="callout shown above the charts")
    args = p.parse_args()

    runs = [load_run(path) for path in args.checkpoints]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(build_html(runs, args.title, args.note))

    print(f"wrote {args.out}")
    for run in runs:
        print(f"  {run['name']}: {len(run['series']['loss'])} epochs")
        for s in run["stats"]:
            print(f"    {s['y_label']:>9} vs {s['x_label']:<8} "
                  f"rho={s['spearman']:+.3f}  R2={s['r_squared']:.3f}  p={s['p_value']:.2g}")


if __name__ == "__main__":
    main()
