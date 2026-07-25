#!/usr/bin/env python3
"""Render a training run's convergence as a self-contained HTML page.

Reads a checkpoint written by :func:`magnonic_nn.save_checkpoint` and emits one
panel per tracked quantity. Loss and contrast are on different scales and in
different units, so they get stacked panels sharing an epoch axis rather than a
shared pair of y-axes -- two y-scales on one frame make any two curves look
correlated, which is exactly the judgement this chart exists to support.

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

# Slots 1 and 2 of the validated categorical palette, in both modes.
SERIES = [
    {"light": "#2a78d6", "dark": "#3987e5"},
    {"light": "#eb6834", "dark": "#d95926"},
    {"light": "#1baf7a", "dark": "#199e70"},
]

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
        "key": "epoch_seconds",
        "title": "Wall-clock per epoch",
        "subtitle": "one forward rollout plus one backward pass",
        "unit": " s",
        "decimals": 1,
        "zero_line": None,
    },
]


def load_run(path: Path):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    history = payload.get("history", {})
    series = {"loss": list(history.get("loss", []))}
    series.update({k: list(v) for k, v in history.get("metrics", {}).items()})
    series["epoch_seconds"] = list(history.get("epoch_seconds", []))

    cfg = payload.get("config", {}) or {}
    mesh = cfg.get("mesh", {})
    solver = cfg.get("solver", {})
    fields = cfg.get("fields", {})

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

    return {
        "name": path.parent.name,
        "series": series,
        "meta": meta,
        "epoch": payload.get("epoch"),
    }


def build_html(runs, title: str, note: str | None) -> str:
    panels = []
    for panel in PANELS:
        present = [r for r in runs if r["series"].get(panel["key"])]
        if present:
            panels.append({
                **panel,
                "runs": [{"name": r["name"], "values": r["series"][panel["key"]]} for r in present],
            })

    payload = json.dumps({"panels": panels, "series_colors": SERIES})

    meta_rows = ""
    for run in runs:
        items = "".join(
            f"<span class='chip'><b>{html.escape(k)}</b> {html.escape(v)}</span>"
            for k, v in run["meta"].items()
        )
        label = f"<span class='run-name'>{html.escape(run['name'])}</span>" if len(runs) > 1 else ""
        meta_rows += f"<div class='meta-row'>{label}{items}</div>"

    note_html = f"<p class='note'>{html.escape(note)}</p>" if note else ""

    return f"""<title>{html.escape(title)}</title>
<style>
  .viz-root {{
    color-scheme: light;
    --surface-1: #fcfcfb;
    --plane: #f9f9f7;
    --text-primary: #0b0b0b;
    --text-secondary: #52514e;
    --muted: #898781;
    --grid: #e1e0d9;
    --axis: #c3c2b7;
    --border: rgba(11,11,11,0.10);
    --s1: #2a78d6;
    --s2: #eb6834;
    --s3: #1baf7a;
  }}
  @media (prefers-color-scheme: dark) {{
    :root:where(:not([data-theme="light"])) .viz-root {{
      color-scheme: dark;
      --surface-1: #1a1a19;
      --plane: #0d0d0d;
      --text-primary: #ffffff;
      --text-secondary: #c3c2b7;
      --muted: #898781;
      --grid: #2c2c2a;
      --axis: #383835;
      --border: rgba(255,255,255,0.10);
      --s1: #3987e5;
      --s2: #d95926;
      --s3: #199e70;
    }}
  }}
  :root[data-theme="dark"] .viz-root {{
    color-scheme: dark;
    --surface-1: #1a1a19;
    --plane: #0d0d0d;
    --text-primary: #ffffff;
    --text-secondary: #c3c2b7;
    --muted: #898781;
    --grid: #2c2c2a;
    --axis: #383835;
    --border: rgba(255,255,255,0.10);
    --s1: #3987e5;
    --s2: #d95926;
    --s3: #199e70;
  }}

  .viz-root {{
    font-family: system-ui, -apple-system, "Segoe UI", sans-serif;
    background: var(--plane);
    color: var(--text-primary);
    margin: 0;
    padding: 28px 20px 48px;
    line-height: 1.5;
  }}
  .wrap {{ max-width: 860px; margin: 0 auto; }}
  h1 {{ font-size: 1.45rem; font-weight: 600; margin: 0 0 4px; letter-spacing: -0.01em; }}
  .sub {{ color: var(--text-secondary); font-size: .93rem; margin: 0 0 18px; }}
  .note {{
    color: var(--text-secondary); font-size: .85rem; margin: 0 0 18px;
    padding: 9px 13px; background: var(--surface-1);
    border: 1px solid var(--border); border-radius: 8px;
  }}
  .meta-row {{ display: flex; flex-wrap: wrap; gap: 6px; margin-bottom: 8px; align-items: center; }}
  .run-name {{ font-size: .8rem; font-weight: 600; margin-right: 4px; }}
  .chip {{
    font-size: .76rem; color: var(--text-secondary);
    background: var(--surface-1); border: 1px solid var(--border);
    border-radius: 999px; padding: 2px 10px;
  }}
  .chip b {{ color: var(--text-primary); font-weight: 600; margin-right: 4px; }}

  .tiles {{ display: flex; flex-wrap: wrap; gap: 10px; margin: 18px 0 24px; }}
  .tile {{
    flex: 1 1 130px; background: var(--surface-1); border: 1px solid var(--border);
    border-radius: 10px; padding: 12px 14px;
  }}
  .tile .label {{ font-size: .74rem; color: var(--muted); text-transform: uppercase;
                  letter-spacing: .04em; }}
  .tile .value {{
    font-size: 1.6rem; font-weight: 600; letter-spacing: -0.02em; margin-top: 2px;
    font-variant-numeric: tabular-nums;
  }}
  .tile .value small {{ font-size: .85rem; font-weight: 500; color: var(--text-secondary); }}

  .panel {{
    background: var(--surface-1); border: 1px solid var(--border);
    border-radius: 12px; padding: 16px 8px 8px 8px; margin-bottom: 14px;
  }}
  .panel h2 {{ font-size: .98rem; font-weight: 600; margin: 0 0 2px; padding: 0 10px; }}
  .panel .cap {{ font-size: .8rem; color: var(--text-secondary); margin: 0 0 6px; padding: 0 10px; }}
  .chart-scroll {{ overflow-x: auto; }}
  svg {{ display: block; width: 100%; min-width: 320px; height: auto; }}
  .grid-line {{ stroke: var(--grid); stroke-width: 1; }}
  .axis-line {{ stroke: var(--axis); stroke-width: 1; }}
  .tick {{ fill: var(--muted); font-size: 11px; font-variant-numeric: tabular-nums; }}
  .end-label {{ fill: var(--text-primary); font-size: 12px; font-weight: 600; }}
  .hit {{ fill: transparent; }}
  .crosshair {{ stroke: var(--axis); stroke-width: 1; stroke-dasharray: 3 3; opacity: 0; }}
  .focus-dot {{ opacity: 0; }}

  .tip {{
    position: fixed; pointer-events: none; opacity: 0; transition: opacity .08s;
    background: var(--surface-1); border: 1px solid var(--border);
    border-radius: 8px; padding: 6px 10px; font-size: .8rem;
    box-shadow: 0 4px 16px rgba(0,0,0,.16); z-index: 20; white-space: nowrap;
  }}
  .tip .k {{ color: var(--text-secondary); }}
  .tip .v {{ font-weight: 600; font-variant-numeric: tabular-nums; }}

  details {{ margin-top: 18px; }}
  summary {{ cursor: pointer; font-size: .86rem; color: var(--text-secondary);
             border-radius: 4px; padding: 2px 4px; }}
  summary:focus-visible, a:focus-visible {{
    outline: 2px solid var(--s1); outline-offset: 2px;
  }}
  .table-scroll {{ overflow-x: auto; }}

  @media (prefers-reduced-motion: reduce) {{
    .tip {{ transition: none; }}
  }}
  table {{ border-collapse: collapse; margin-top: 10px; font-size: .82rem; width: 100%; }}
  th, td {{ text-align: right; padding: 4px 10px; border-bottom: 1px solid var(--border);
            font-variant-numeric: tabular-nums; }}
  th:first-child, td:first-child {{ text-align: left; }}
  th {{ color: var(--text-secondary); font-weight: 600; }}
</style>

<div class="viz-root">
<div class="wrap">
  <h1>{html.escape(title)}</h1>
  <p class="sub">Gradient descent through the Landau&ndash;Lifshitz&ndash;Gilbert
     equation &mdash; every point is one full micromagnetic rollout plus its adjoint.</p>
  {note_html}
  {meta_rows}
  <div class="tiles" id="tiles"></div>
  <div id="panels"></div>

  <details>
    <summary>Show the numbers</summary>
    <div class="table-scroll" id="table"></div>
  </details>
</div>
</div>
<div class="tip" id="tip"></div>

<script>
const DATA = {payload};

const W = 820, H = 190, M = {{t: 14, r: 84, b: 30, l: 56}};
const tip = document.getElementById('tip');

function niceTicks(lo, hi, count) {{
  if (lo === hi) {{ lo -= 1; hi += 1; }}
  const raw = (hi - lo) / count;
  const mag = Math.pow(10, Math.floor(Math.log10(raw)));
  const norm = raw / mag;
  const step = (norm <= 1 ? 1 : norm <= 2 ? 2 : norm <= 5 ? 5 : 10) * mag;
  const ticks = [];
  for (let v = Math.ceil(lo / step) * step; v <= hi + step * 1e-9; v += step) ticks.push(v);
  return ticks;
}}

function fmt(v, d, unit) {{ return v.toFixed(d) + (unit || ''); }}

function renderPanel(panel, colorIndex) {{
  const runs = panel.runs;
  const all = runs.flatMap(r => r.values);
  const n = Math.max(...runs.map(r => r.values.length));

  let lo = Math.min(...all), hi = Math.max(...all);
  if (panel.zero_line !== null && panel.zero_line !== undefined) {{
    lo = Math.min(lo, panel.zero_line); hi = Math.max(hi, panel.zero_line);
  }}
  const pad = (hi - lo) * 0.12 || 1;
  lo -= pad; hi += pad;

  const x = i => M.l + (n <= 1 ? 0 : i * (W - M.l - M.r) / (n - 1));
  const y = v => M.t + (hi - v) * (H - M.t - M.b) / (hi - lo);

  const yticks = niceTicks(lo, hi, 4);
  const xstep = Math.max(1, Math.ceil(n / 10));

  let svg = `<svg viewBox="0 0 ${{W}} ${{H}}" role="img" aria-label="${{panel.title}} by epoch">`;
  yticks.forEach(t => {{
    svg += `<line class="grid-line" x1="${{M.l}}" x2="${{W - M.r}}" y1="${{y(t).toFixed(1)}}" y2="${{y(t).toFixed(1)}}"/>`;
    svg += `<text class="tick" x="${{M.l - 8}}" y="${{(y(t) + 4).toFixed(1)}}" text-anchor="end">${{fmt(t, panel.decimals, '')}}</text>`;
  }});
  if (panel.zero_line !== null && panel.zero_line !== undefined) {{
    svg += `<line class="axis-line" x1="${{M.l}}" x2="${{W - M.r}}" y1="${{y(panel.zero_line).toFixed(1)}}" y2="${{y(panel.zero_line).toFixed(1)}}"/>`;
  }}
  svg += `<line class="axis-line" x1="${{M.l}}" x2="${{M.l}}" y1="${{M.t}}" y2="${{H - M.b}}"/>`;
  svg += `<line class="axis-line" x1="${{M.l}}" x2="${{W - M.r}}" y1="${{H - M.b}}" y2="${{H - M.b}}"/>`;
  for (let i = 0; i < n; i += xstep) {{
    svg += `<text class="tick" x="${{x(i).toFixed(1)}}" y="${{H - M.b + 16}}" text-anchor="middle">${{i}}</text>`;
  }}
  svg += `<text class="tick" x="${{(M.l + (W - M.r)) / 2}}" y="${{H - 2}}" text-anchor="middle">epoch</text>`;

  runs.forEach((run, ri) => {{
    const color = `var(--s${{(colorIndex + ri) % 3 + 1}})`;
    const pts = run.values.map((v, i) => `${{x(i).toFixed(1)}},${{y(v).toFixed(1)}}`).join(' ');
    svg += `<polyline points="${{pts}}" fill="none" stroke="${{color}}" stroke-width="2"
             stroke-linejoin="round" stroke-linecap="round"/>`;
    run.values.forEach((v, i) => {{
      svg += `<circle cx="${{x(i).toFixed(1)}}" cy="${{y(v).toFixed(1)}}" r="4"
               fill="${{color}}" stroke="var(--surface-1)" stroke-width="2"/>`;
    }});
    const last = run.values.length - 1;
    svg += `<text class="end-label" x="${{(x(last) + 10).toFixed(1)}}" y="${{(y(run.values[last]) + 4).toFixed(1)}}">${{fmt(run.values[last], panel.decimals, panel.unit)}}</text>`;
  }});

  svg += `<line class="crosshair" id="ch-${{panel.key}}" y1="${{M.t}}" y2="${{H - M.b}}"/>`;
  svg += `<rect class="hit" x="${{M.l}}" y="0" width="${{W - M.l - M.r}}" height="${{H}}"
           data-key="${{panel.key}}" data-n="${{n}}"/>`;
  svg += `</svg>`;

  const legend = runs.length > 1
    ? `<p class="cap">` + runs.map((r, ri) =>
        `<span style="color:var(--s${{(colorIndex + ri) % 3 + 1}})">&#9632;</span> ${{r.name}}`).join('&nbsp;&nbsp;') + `</p>`
    : '';

  return `<div class="panel">
      <h2>${{panel.title}}</h2>
      <p class="cap">${{panel.subtitle}}</p>
      ${{legend}}
      <div class="chart-scroll">${{svg}}</div>
    </div>`;
}}

// stat tiles from the last epoch of the first run
const first = DATA.panels;
let tiles = '';
const lossPanel = first.find(p => p.key === 'loss');
if (lossPanel) {{
  const v = lossPanel.runs[0].values;
  tiles += `<div class="tile"><div class="label">epochs</div><div class="value">${{v.length}}</div></div>`;
  tiles += `<div class="tile"><div class="label">loss</div><div class="value">${{v[v.length - 1].toFixed(3)}}
            <small>from ${{v[0].toFixed(3)}}</small></div></div>`;
}}
const cPanel = first.find(p => p.key === 'contrast_dB');
if (cPanel) {{
  const v = cPanel.runs[0].values;
  tiles += `<div class="tile"><div class="label">contrast</div><div class="value">${{v[v.length - 1].toFixed(1)}}
            <small>dB, from ${{v[0].toFixed(1)}}</small></div></div>`;
}}
const tPanel = first.find(p => p.key === 'epoch_seconds');
if (tPanel) {{
  // Median, not mean: a single epoch that lost the CPU to something else
  // should not become the headline number for how long an epoch takes.
  const v = [...tPanel.runs[0].values].sort((a, b) => a - b);
  const mid = v.length % 2 ? v[(v.length - 1) / 2] : (v[v.length / 2 - 1] + v[v.length / 2]) / 2;
  tiles += `<div class="tile"><div class="label">per epoch</div><div class="value">${{mid.toFixed(0)}}
            <small>s median</small></div></div>`;
}}
document.getElementById('tiles').innerHTML = tiles;
document.getElementById('panels').innerHTML =
  DATA.panels.map((p, i) => renderPanel(p, i)).join('');

// table view
let rows = '<table><thead><tr><th>epoch</th>' +
  DATA.panels.flatMap(p => p.runs.map(r =>
    `<th>${{p.title}}${{p.runs.length > 1 ? ' &middot; ' + r.name : ''}}</th>`)).join('') +
  '</tr></thead><tbody>';
const maxN = Math.max(...DATA.panels.flatMap(p => p.runs.map(r => r.values.length)));
for (let i = 0; i < maxN; i++) {{
  rows += `<tr><td>${{i}}</td>` + DATA.panels.flatMap(p => p.runs.map(r =>
    `<td>${{r.values[i] === undefined ? '&mdash;' : r.values[i].toFixed(p.decimals) + p.unit}}</td>`)).join('') + '</tr>';
}}
document.getElementById('table').innerHTML = rows + '</tbody></table>';

// hover: crosshair + tooltip
document.querySelectorAll('.hit').forEach(hit => {{
  const svg = hit.ownerSVGElement;
  const key = hit.dataset.key;
  const n = +hit.dataset.n;
  const panel = DATA.panels.find(p => p.key === key);
  const ch = svg.querySelector('#ch-' + key);

  hit.addEventListener('pointermove', ev => {{
    const box = svg.getBoundingClientRect();
    const px = (ev.clientX - box.left) / box.width * W;
    const i = Math.max(0, Math.min(n - 1, Math.round((px - M.l) / ((W - M.l - M.r) / Math.max(n - 1, 1)))));
    const cx = M.l + (n <= 1 ? 0 : i * (W - M.l - M.r) / (n - 1));
    ch.setAttribute('x1', cx); ch.setAttribute('x2', cx); ch.style.opacity = 1;

    const lines = panel.runs.map(r => r.values[i] === undefined ? null :
      `<div><span class="k">${{panel.runs.length > 1 ? r.name + ' ' : ''}}</span>
       <span class="v">${{r.values[i].toFixed(panel.decimals)}}${{panel.unit}}</span></div>`)
      .filter(Boolean).join('');
    tip.innerHTML = `<div class="k">epoch ${{i}}</div>${{lines}}`;
    tip.style.opacity = 1;
    tip.style.left = Math.min(ev.clientX + 14, window.innerWidth - tip.offsetWidth - 8) + 'px';
    tip.style.top = (ev.clientY - 12) + 'px';
  }});
  hit.addEventListener('pointerleave', () => {{ tip.style.opacity = 0; ch.style.opacity = 0; }});
}});
</script>
"""


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
        n = len(run["series"].get("loss", []))
        print(f"  {run['name']}: {n} epochs")


if __name__ == "__main__":
    main()
