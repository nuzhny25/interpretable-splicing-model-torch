"""Interactive SR-balance heatmap over the full MALAT1 alignment matrix.

Writes a self-contained HTML file (no external dependencies) whose grid exactly
mirrors ``alignment_matrix.npy`` (15273 aligned MSA columns x 62 species). Each
populated cell is the raw, unsmoothed SR balance of the 6-nt window anchored at
that aligned column for that species — window k sits at the aligned column of
its first nucleotide, matching how custom_real_train.py scatters the profile.

Instead of embedding per-cell values, the page embeds the raw matrix columns
plus the conv filter weights; JavaScript recomputes every window's activations
at load time and the per-filter breakdown lazily on hover. This keeps the file
small and guarantees the tooltip contributions sum exactly to the cell value.

Page features:
  * Incl - Skip (diverging colormap) / Incl + Skip (sequential) toggle.
  * A filter picker: "all 40 filters" (summed, the SR balance) or any single
    I##/S## filter, in which case cells, scales, and the SD strip show only
    that filter's softplus contribution (skip filters negative in diff mode).
  * A cross-species SD strip under the grid: per aligned column, the standard
    deviation (ddof=0) of the current mode's values over the species populated
    there, on its own orange sequential scale with its own hover tooltip.
  * Hover tooltip: species, 6-mer, aligned-column span (flagging gap-spanning
    windows), cell value, and all 40 per-filter softplus contributions sorted
    largest to smallest. Click pins the tooltip; Esc unpins.
  * Zoom buttons (1-16 px per column), sticky species labels, position ruler.
  * A self-check badge comparing the JS math against reference cells computed
    here with torch via compute_sr_profile.
"""

import json
import os
import sys

import numpy as np
import torch
from numpy.lib.stride_tricks import sliding_window_view

ROOT = os.path.join(os.path.dirname(__file__), "../..")
CUSTOM_DIR = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, ROOT)
# Ahead of ROOT: "custom_model" names both ROOT/custom_model/ (a namespace
# portion) and custom_model/custom_model.py (the module we want). The regular
# module only wins if its directory comes first on sys.path.
sys.path.insert(0, CUSTOM_DIR)
from custom_model import PNASModel
from custom_utils import str_to_vector
from dataset_preparations.maf_processing import SPECIES

MATRIX_PATH = os.path.join(ROOT, "data/multiz100/alignment_matrix.npy")
WEIGHTS_PATH = os.path.join(CUSTOM_DIR, "weights/real_weights_aad.pt")
OUT_PATH = os.path.join(os.path.dirname(__file__), "sr_heatmap.html")


def species_profile(matrix, model, s_idx):
    """Gap-free sequence, aligned columns, and per-window activations."""
    row = matrix[:, s_idx]
    is_nuc = (row != "-") & (row != "N")
    cols = np.nonzero(is_nuc)[0]
    seq = "".join(row[is_nuc]).upper()

    x_seq = torch.tensor(str_to_vector(seq), dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        sr, a_incl, a_skip = model.compute_sr_profile(x_seq, return_activations=True)
    return (
        seq,
        cols,
        sr.squeeze(0).numpy(),
        a_incl.squeeze(0).numpy(),
        a_skip.squeeze(0).numpy(),
    )


def main():
    matrix = np.load(MATRIX_PATH)
    n_aligned, n_species = matrix.shape

    model = PNASModel()
    ckpt = torch.load(WEIGHTS_PATH, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt.get("model_state_dict", ckpt))
    model.eval()

    species_names = list(SPECIES)
    assert len(species_names) == n_species

    payload = {
        "n_aligned": int(n_aligned),
        "species": species_names,
        "columns": ["".join(matrix[:, j]) for j in range(n_species)],
        "w_incl": model.conv_incl.weight.detach().numpy().tolist(),
        "b_incl": model.conv_incl.bias.detach().numpy().tolist(),
        "w_skip": model.conv_skip.weight.detach().numpy().tolist(),
        "b_skip": model.conv_skip.bias.detach().numpy().tolist(),
    }

    # Parity check: the manual conv the JS performs (one-hot lookup-sum over the
    # round-tripped payload weights + softplus) must reproduce the torch path.
    seq, cols, sr, a_incl, a_skip = species_profile(matrix, model, 0)
    w_incl = np.array(payload["w_incl"], dtype=np.float32)
    b_incl = np.array(payload["b_incl"], dtype=np.float32)
    w_skip = np.array(payload["w_skip"], dtype=np.float32)
    b_skip = np.array(payload["b_skip"], dtype=np.float32)
    oh = str_to_vector(seq)  # (4, L)
    win = sliding_window_view(oh, 6, axis=1)  # (4, L-5, 6)
    z_incl = np.einsum("fcp,ckp->fk", w_incl, win) + b_incl[:, None]
    z_skip = np.einsum("fcp,ckp->fk", w_skip, win) + b_skip[:, None]
    a_incl_manual = np.logaddexp(0, z_incl).sum(axis=0)
    a_skip_manual = np.logaddexp(0, z_skip).sum(axis=0)
    assert np.allclose(a_incl_manual, a_incl, atol=1e-4), "incl parity failed"
    assert np.allclose(a_skip_manual, a_skip, atol=1e-4), "skip parity failed"
    assert np.allclose(a_incl_manual - a_skip_manual, sr, atol=1e-4)
    print(f"parity check OK: human ({len(sr)} windows), torch vs payload weights")

    # Reference cells for the in-browser self-check.
    refs = []
    for s_idx, k_choice in [(0, 0), (0, None), (n_species // 2, 0)]:
        seq, cols, sr, a_incl, a_skip = species_profile(matrix, model, s_idx)
        k = len(sr) // 2 if k_choice is None else k_choice
        refs.append(
            {
                "s": int(s_idx),
                "k": int(k),
                "col": int(cols[k]),
                "sixmer": seq[k : k + 6],
                "diff": float(sr[k]),
                "sum": float(a_incl[k] + a_skip[k]),
            }
        )
    payload["refs"] = refs

    # Cross-species SD reference (Incl - Skip mode) at the best-populated
    # column, for the in-browser check of the SD strip.
    col_sum = np.zeros(n_aligned)
    col_sq = np.zeros(n_aligned)
    col_cnt = np.zeros(n_aligned, dtype=int)
    for s_idx in range(n_species):
        _, cols, sr, _, _ = species_profile(matrix, model, s_idx)
        c = cols[: len(sr)]
        np.add.at(col_sum, c, sr)
        np.add.at(col_sq, c, sr.astype(np.float64) ** 2)
        np.add.at(col_cnt, c, 1)
    c_best = int(col_cnt.argmax())
    m = col_sum[c_best] / col_cnt[c_best]
    sd_best = float(np.sqrt(max(0.0, col_sq[c_best] / col_cnt[c_best] - m * m)))
    payload["sd_ref"] = {"col": c_best, "n": int(col_cnt[c_best]), "sd": sd_best}

    data_json = json.dumps(payload, separators=(",", ":")).replace("</", "<\\/")
    html = HTML_TEMPLATE.replace("__DATA_JSON__", data_json)
    with open(OUT_PATH, "w") as f:
        f.write(html)

    print(f"wrote {OUT_PATH} ({os.path.getsize(OUT_PATH) / 1e6:.2f} MB)")
    print("reference cells (species, window, aligned col, 6-mer, Incl-Skip, Incl+Skip):")
    for r in refs:
        print(
            f"  {species_names[r['s']]:<12} k={r['k']:<5} col={r['col']:<6} "
            f"{r['sixmer']}  diff={r['diff']:+.4f}  sum={r['sum']:.4f}"
        )
    sd_ref = payload["sd_ref"]
    print(
        f"SD reference: col={sd_ref['col']} n={sd_ref['n']} "
        f"sd(Incl-Skip)={sd_ref['sd']:.4f}"
    )


HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>MALAT1 SR-balance heatmap</title>
<style>
  :root {
    color-scheme: light;
    --page:       #f9f9f7;
    --surface:    #fcfcfb;
    --ink:        #0b0b0b;
    --ink-2:      #52514e;
    --muted:      #898781;
    --hairline:   #e1e0d9;
    --border:     rgba(11,11,11,0.10);
    --nodata:     #b5b4af;
  }
  * { box-sizing: border-box; margin: 0; }
  body {
    background: var(--page);
    color: var(--ink);
    font-family: system-ui, -apple-system, "Segoe UI", sans-serif;
    font-size: 13px;
    padding: 16px;
  }
  h1 { font-size: 16px; font-weight: 600; margin-bottom: 2px; }
  .sub { color: var(--ink-2); font-size: 12px; margin-bottom: 12px; }
  #controls {
    display: flex; align-items: center; gap: 18px; flex-wrap: wrap;
    margin-bottom: 10px;
  }
  .seg { display: inline-flex; border: 1px solid var(--border); border-radius: 6px; overflow: hidden; }
  .seg button {
    border: 0; background: var(--surface); color: var(--ink-2);
    font: inherit; font-size: 12px; padding: 5px 12px; cursor: pointer;
  }
  .seg button + button { border-left: 1px solid var(--border); }
  .seg button.on { background: var(--ink); color: #fff; }
  select {
    font: inherit; font-size: 12px; padding: 4px 8px;
    background: var(--surface); color: var(--ink-2);
    border: 1px solid var(--border); border-radius: 6px;
  }
  .ctl-label { color: var(--muted); font-size: 11px; margin-right: -10px; }
  #legend, #sdlegend { display: flex; align-items: center; gap: 8px; font-size: 11px; color: var(--ink-2); }
  #legend canvas, #sdlegend canvas { border: 1px solid var(--border); border-radius: 2px; }
  #sdlegend .ctl-label { margin-right: 0; }
  .swatch { display: inline-block; width: 12px; height: 12px; border-radius: 2px;
            background: var(--nodata); vertical-align: -2px; margin-right: 4px; }
  #badge { font-size: 11px; padding: 3px 8px; border-radius: 10px; }
  #badge.ok   { background: #e2f2e2; color: #006300; }
  #badge.fail { background: #f7dcdc; color: #a02020; }
  #badge.wait { background: var(--hairline); color: var(--ink-2); }
  .hint { color: var(--muted); font-size: 11px; }

  #scroll {
    overflow: auto; position: relative;
    background: var(--surface);
    border: 1px solid var(--border); border-radius: 8px;
    max-height: calc(100vh - 150px);
  }
  #ruler {
    position: relative; height: 20px; margin-left: var(--labelw);
    border-bottom: 1px solid var(--hairline);
  }
  #ruler .tick {
    position: absolute; top: 0; height: 100%;
    border-left: 1px solid var(--muted);
    color: var(--muted); font-size: 9px; padding-left: 3px;
    font-variant-numeric: tabular-nums; white-space: nowrap;
  }
  #gridrow { display: flex; align-items: flex-start; }
  #labels {
    position: sticky; left: 0; z-index: 3;
    background: var(--surface); border-right: 1px solid var(--hairline);
    flex: 0 0 auto; width: var(--labelw);
  }
  #labels div {
    height: var(--cellh); line-height: var(--cellh);
    font-size: 9px; color: var(--ink-2);
    text-align: right; padding-right: 6px;
    overflow: hidden; white-space: nowrap;
  }
  #labels div:nth-child(even) { background: var(--page); }
  #wrap { position: relative; flex: 0 0 auto; }
  #hm { display: block; image-rendering: pixelated; cursor: crosshair; }
  #cellbox {
    position: absolute; pointer-events: none; display: none;
    border: 1.5px solid var(--ink); z-index: 2;
  }
  #cellbox.pinned { border-color: #d03b3b; }

  #sdrow { display: flex; align-items: flex-start; border-top: 1px solid var(--hairline); }
  #sdlabel {
    position: sticky; left: 0; z-index: 3;
    background: var(--surface); border-right: 1px solid var(--hairline);
    flex: 0 0 auto; width: var(--labelw);
    height: 14px; line-height: 14px;
    font-size: 9px; color: var(--ink-2);
    text-align: right; padding-right: 6px; white-space: nowrap;
  }
  #sdwrap { position: relative; flex: 0 0 auto; }
  #sd { display: block; image-rendering: pixelated; cursor: crosshair; height: 14px; }

  #tooltip {
    position: fixed; display: none; z-index: 10; pointer-events: none;
    background: var(--surface); border: 1px solid var(--border);
    border-radius: 8px; box-shadow: 0 4px 16px rgba(11,11,11,0.15);
    padding: 10px 12px; font-size: 11px; max-width: 340px;
  }
  #tooltip .tt-species { font-weight: 600; font-size: 12px; }
  #tooltip .tt-meta { color: var(--ink-2); margin: 2px 0 4px; }
  #tooltip .tt-gap { color: #a02020; font-weight: 600; }
  #tooltip .tt-mer { font-family: ui-monospace, Menlo, monospace; font-size: 13px;
                     letter-spacing: 2px; margin: 2px 0; }
  #tooltip .tt-val { margin-bottom: 6px; }
  #tooltip .tt-val b { font-variant-numeric: tabular-nums; }
  #contribs {
    display: grid; grid-template-columns: 1fr 1fr; column-gap: 14px; row-gap: 0;
    font-family: ui-monospace, Menlo, monospace; font-size: 10px;
    color: var(--ink-2); border-top: 1px solid var(--hairline); padding-top: 5px;
  }
  #contribs span { white-space: pre; }
  #contribs .inc { color: #8c2f2b; }
  #contribs .skp { color: #17497f; }
  #contribs .sel { background: #f5edda; font-weight: 700; border-radius: 3px; }

  #splash {
    position: fixed; inset: 0; z-index: 20;
    background: var(--page); display: flex;
    align-items: center; justify-content: center;
    color: var(--ink-2); font-size: 14px;
  }
</style>
</head>
<body>
<h1>MALAT1 SR balance &mdash; per-species, per-window heatmap</h1>
<div class="sub">Rows: 62 species (MSA column order) &middot; Columns: 15273 aligned positions &middot;
Each cell = raw SR value of the 6-nt window anchored at that aligned column. Hover for the window and per-filter contributions; click to pin, Esc to unpin.</div>

<div id="controls">
  <span class="seg" id="modeSeg">
    <button data-mode="diff" class="on">Incl &minus; Skip</button>
    <button data-mode="sum">Incl + Skip</button>
  </span>
  <span class="ctl-label">filter</span>
  <select id="filterPick"></select>
  <span class="ctl-label">zoom</span>
  <span class="seg" id="zoomSeg">
    <button data-z="1">1&times;</button>
    <button data-z="2" class="on">2&times;</button>
    <button data-z="4">4&times;</button>
    <button data-z="8">8&times;</button>
    <button data-z="16">16&times;</button>
  </span>
  <span id="legend">
    <span id="legmin"></span><canvas id="legbar" width="180" height="10"></canvas><span id="legmax"></span>
    <span><span class="swatch"></span>no data</span>
  </span>
  <span id="sdlegend">
    <span class="ctl-label">column SD</span>
    <span>0</span><canvas id="legsd" width="100" height="10"></canvas><span id="legsdmax"></span>
  </span>
  <span id="badge" class="wait">self-check&hellip;</span>
</div>

<div id="scroll">
  <div id="ruler"></div>
  <div id="gridrow">
    <div id="labels"></div>
    <div id="wrap">
      <canvas id="hm"></canvas>
      <div id="cellbox"></div>
    </div>
  </div>
  <div id="sdrow">
    <div id="sdlabel">cross-species SD</div>
    <div id="sdwrap"><canvas id="sd"></canvas></div>
  </div>
</div>

<div id="tooltip"></div>
<div id="splash">computing activations&hellip; <span id="splashpct"></span></div>

<script id="data" type="application/json">__DATA_JSON__</script>
<script>
"use strict";
const DATA = JSON.parse(document.getElementById("data").textContent);
const NA = DATA.n_aligned, NS = DATA.species.length, K = 6, NF = 20;

const CELLH = 10;
let cellW = 2;
let mode = "diff";   // "diff" = Incl - Skip, "sum" = Incl + Skip
document.documentElement.style.setProperty("--cellh", CELLH + "px");
document.documentElement.style.setProperty("--labelw", "150px");

// ---- weights, flattened for the hot loop: w[f*24 + base*6 + pos] ----
function flatten(w) {
  const a = new Float32Array(NF * 4 * K);
  for (let f = 0; f < NF; f++)
    for (let c = 0; c < 4; c++)
      for (let p = 0; p < K; p++)
        a[f * 24 + c * 6 + p] = w[f][c][p];
  return a;
}
const WI = flatten(DATA.w_incl), WS = flatten(DATA.w_skip);
const BI = Float32Array.from(DATA.b_incl), BS = Float32Array.from(DATA.b_skip);
const softplus = z => (z > 30 ? z : Math.log1p(Math.exp(z)));

// ---- per-species parsing: gap-free codes + coordinate maps ----
const CODE = { A: 0, C: 1, G: 2, T: 3 };
const SP = [];  // {cols, seq, codes, winOfCol, nWin, aIncl, aSkip}
for (let s = 0; s < NS; s++) {
  const colStr = DATA.columns[s];
  const cols = [], chars = [];
  for (let i = 0; i < NA; i++) {
    const ch = colStr[i];
    if (ch === "-" || ch === "N") continue;
    cols.push(i);
    chars.push(ch.toUpperCase());
  }
  const seq = chars.join("");
  const L = seq.length, nWin = Math.max(0, L - 5);
  const codes = new Uint8Array(L);
  for (let i = 0; i < L; i++) codes[i] = CODE[seq[i]];
  const winOfCol = new Int32Array(NA).fill(-1);
  for (let k = 0; k < nWin; k++) winOfCol[cols[k]] = k;
  SP.push({ cols: Int32Array.from(cols), seq, codes, winOfCol, nWin,
            aIncl: null, aSkip: null });
}

// Because the input is one-hot, each filter's pre-activation is a plain
// lookup-sum of 6 weights — no multiplications.
function windowActs(codes, k, out) {  // out: Float32Array(40) = [incl0..19, skip0..19]
  for (let f = 0; f < NF; f++) {
    let zI = BI[f], zS = BS[f];
    const base = f * 24;
    for (let p = 0; p < K; p++) {
      const off = base + codes[k + p] * 6 + p;
      zI += WI[off];
      zS += WS[off];
    }
    out[f] = softplus(zI);
    out[NF + f] = softplus(zS);
  }
}

const scratch = new Float32Array(2 * NF);
function computeSpecies(s) {
  const sp = SP[s];
  const aI = new Float32Array(sp.nWin), aS = new Float32Array(sp.nWin);
  for (let k = 0; k < sp.nWin; k++) {
    windowActs(sp.codes, k, scratch);
    let sI = 0, sS = 0;
    for (let f = 0; f < NF; f++) { sI += scratch[f]; sS += scratch[NF + f]; }
    aI[k] = sI;
    aS[k] = sS;
  }
  sp.aIncl = aI;
  sp.aSkip = aS;
}

// ---- colormaps (256-entry LUTs) ----
function hex2rgb(h) { return [1, 3, 5].map(i => parseInt(h.slice(i, i + 2), 16)); }
function buildLUT(stops) {  // stops: [[t, "#hex"], ...] with t in [0,1]
  const lut = new Uint8ClampedArray(256 * 3);
  for (let i = 0; i < 256; i++) {
    const t = i / 255;
    let a = 0;
    while (a < stops.length - 2 && t > stops[a + 1][0]) a++;
    const [t0, c0] = stops[a], [t1, c1] = stops[a + 1];
    const u = t1 > t0 ? Math.min(1, Math.max(0, (t - t0) / (t1 - t0))) : 0;
    const r0 = hex2rgb(c0), r1 = hex2rgb(c1);
    for (let c = 0; c < 3; c++) lut[i * 3 + c] = r0[c] + u * (r1[c] - r0[c]);
  }
  return lut;
}
const LUT_DIFF = buildLUT([[0, "#1c5cab"], [0.5, "#f0efec"], [1, "#b23b36"]]);
const LUT_SUM  = buildLUT([[0, "#cde2fb"], [0.5, "#3987e5"], [1, "#0d366b"]]);
// Second sequential context (the SD strip) takes the next categorical hue:
// orange, so it can't be misread on the main colormap's scale.
const LUT_SD   = buildLUT([[0, "#fde3d7"], [0.5, "#eb6834"], [1, "#7a2a0e"]]);
const NODATA = hex2rgb("#b5b4af");

// ---- filter selection: -1 = all 40 summed, 0..19 = I##, 20..39 = S## ----
let filterSel = -1;
const filterName = fi => (fi < NF ? "I" : "S") + String(fi % NF).padStart(2, "0");

// Per-window softplus activation of one filter, for every species (sp.fAct).
function computeFilterActs(fi) {
  const W = fi < NF ? WI : WS, B = fi < NF ? BI : BS;
  const f = fi % NF, base = f * 24;
  for (const sp of SP) {
    const a = new Float32Array(sp.nWin);
    const codes = sp.codes;
    for (let k = 0; k < sp.nWin; k++) {
      let z = B[f];
      for (let p = 0; p < K; p++) z += W[base + codes[k + p] * 6 + p];
      a[k] = softplus(z);
    }
    sp.fAct = a;
  }
}

// Cell values under the current filter selection. A skip filter's
// contribution is negative in diff mode, positive in sum mode.
function valDiff(sp, k) {
  if (filterSel < 0) return sp.aIncl[k] - sp.aSkip[k];
  return filterSel < NF ? sp.fAct[k] : -sp.fAct[k];
}
function valSum(sp, k) {
  return filterSel < 0 ? sp.aIncl[k] + sp.aSkip[k] : sp.fAct[k];
}

let vmaxDiff = 1, vmaxSum = 1;  // 98th percentile for the current selection
function percentileScales() {
  let n = 0;
  for (const sp of SP) n += sp.nWin;
  const dv = new Float32Array(n), sv = new Float32Array(n);
  let i = 0;
  for (const sp of SP)
    for (let k = 0; k < sp.nWin; k++, i++) {
      dv[i] = Math.abs(valDiff(sp, k));
      sv[i] = valSum(sp, k);
    }
  dv.sort(); sv.sort();
  const q = Math.min(n - 1, Math.floor(0.98 * n));
  vmaxDiff = dv[q] || 1;
  vmaxSum = sv[q] || 1;
}

// ---- rendering ----
const hm = document.getElementById("hm");
hm.width = NA;
hm.height = NS;
const ctx = hm.getContext("2d");

function renderHeatmap() {
  const img = ctx.createImageData(NA, NS);
  const px = img.data;
  const lut = mode === "diff" ? LUT_DIFF : LUT_SUM;
  for (let s = 0; s < NS; s++) {
    const rowOff = s * NA * 4;
    for (let i = 0; i < NA; i++) {
      const o = rowOff + i * 4;
      px[o] = NODATA[0]; px[o + 1] = NODATA[1]; px[o + 2] = NODATA[2]; px[o + 3] = 255;
    }
    const sp = SP[s];
    for (let k = 0; k < sp.nWin; k++) {
      let t;
      if (mode === "diff") {
        t = (valDiff(sp, k) / vmaxDiff + 1) / 2;
      } else {
        t = valSum(sp, k) / vmaxSum;
      }
      const li = 3 * Math.round(255 * Math.min(1, Math.max(0, t)));
      const o = rowOff + sp.cols[k] * 4;
      px[o] = lut[li]; px[o + 1] = lut[li + 1]; px[o + 2] = lut[li + 2];
    }
  }
  ctx.putImageData(img, 0, 0);
}

// ---- per-column cross-species SD strip ----
const sdCanvas = document.getElementById("sd");
sdCanvas.width = NA;
sdCanvas.height = 1;
const sdCtx = sdCanvas.getContext("2d");
const sdVals = new Float32Array(NA);
const sdCnt = new Int32Array(NA);
let sdMax = 1;
const SD_MIN_SPECIES = 2;  // columns with fewer populated cells render as no data

function computeSD() {
  const sum = new Float64Array(NA), sq = new Float64Array(NA);
  sdCnt.fill(0);
  for (const sp of SP)
    for (let k = 0; k < sp.nWin; k++) {
      const v = mode === "diff" ? valDiff(sp, k) : valSum(sp, k);
      const c = sp.cols[k];
      sum[c] += v;
      sq[c] += v * v;
      sdCnt[c]++;
    }
  const seen = [];
  for (let i = 0; i < NA; i++) {
    if (sdCnt[i] >= SD_MIN_SPECIES) {
      const m = sum[i] / sdCnt[i];
      sdVals[i] = Math.sqrt(Math.max(0, sq[i] / sdCnt[i] - m * m));
      seen.push(sdVals[i]);
    } else sdVals[i] = NaN;
  }
  seen.sort((a, b) => a - b);
  sdMax = seen[Math.min(seen.length - 1, Math.floor(0.98 * seen.length))] || 1;
}

function renderSD() {
  const img = sdCtx.createImageData(NA, 1);
  const px = img.data;
  for (let i = 0; i < NA; i++) {
    const o = i * 4;
    if (Number.isNaN(sdVals[i])) {
      px[o] = NODATA[0]; px[o + 1] = NODATA[1]; px[o + 2] = NODATA[2];
    } else {
      const t = Math.min(1, Math.max(0, sdVals[i] / sdMax));
      const li = 3 * Math.round(255 * t);
      px[o] = LUT_SD[li]; px[o + 1] = LUT_SD[li + 1]; px[o + 2] = LUT_SD[li + 2];
    }
    px[o + 3] = 255;
  }
  sdCtx.putImageData(img, 0, 0);
}

function renderSDLegend() {
  const bar = document.getElementById("legsd");
  const c = bar.getContext("2d");
  for (let x = 0; x < bar.width; x++) {
    const li = 3 * Math.round(255 * x / (bar.width - 1));
    c.fillStyle = `rgb(${LUT_SD[li]},${LUT_SD[li + 1]},${LUT_SD[li + 2]})`;
    c.fillRect(x, 0, 1, bar.height);
  }
  document.getElementById("legsdmax").textContent = sdMax.toFixed(3);
}

function applyZoom() {
  hm.style.width = NA * cellW + "px";
  hm.style.height = NS * CELLH + "px";
  sdCanvas.style.width = NA * cellW + "px";
  renderRuler();
  hideHover();
}

function renderRuler() {
  const ruler = document.getElementById("ruler");
  ruler.style.width = NA * cellW + "px";
  ruler.innerHTML = "";
  const minor = cellW >= 4 ? 100 : 500;
  const major = cellW >= 4 ? 500 : 1000;
  for (let p = 0; p < NA; p += minor) {
    const d = document.createElement("div");
    d.className = "tick";
    d.style.left = p * cellW + "px";
    if (p % major === 0) d.textContent = p;
    else d.style.height = "35%";
    ruler.appendChild(d);
  }
}

function renderLabels() {
  const el = document.getElementById("labels");
  el.innerHTML = "";
  for (const name of DATA.species) {
    const d = document.createElement("div");
    d.textContent = name.replace(/_/g, " ");
    d.title = name;
    el.appendChild(d);
  }
}

function renderLegend() {
  const bar = document.getElementById("legbar");
  const c = bar.getContext("2d");
  const lut = mode === "diff" ? LUT_DIFF : LUT_SUM;
  for (let x = 0; x < bar.width; x++) {
    const li = 3 * Math.round(255 * x / (bar.width - 1));
    c.fillStyle = `rgb(${lut[li]},${lut[li + 1]},${lut[li + 2]})`;
    c.fillRect(x, 0, 1, bar.height);
  }
  const mn = document.getElementById("legmin"), mx = document.getElementById("legmax");
  if (mode === "diff") {
    mn.textContent = (-vmaxDiff).toFixed(2);
    mx.textContent = "+" + vmaxDiff.toFixed(2);
  } else {
    mn.textContent = "0";
    mx.textContent = vmaxSum.toFixed(2);
  }
}

// ---- tooltip ----
const tooltip = document.getElementById("tooltip");
const cellbox = document.getElementById("cellbox");
let pinned = false;

function showTooltip(s, col, clientX, clientY) {
  const sp = SP[s];
  const k = sp.winOfCol[col];
  if (k < 0) { hideHover(); return; }

  windowActs(sp.codes, k, scratch);
  const contribs = [];
  for (let f = 0; f < NF; f++) {
    contribs.push({ label: "I" + String(f).padStart(2, "0"),
                    cls: "inc" + (filterSel === f ? " sel" : ""), v: scratch[f] });
    contribs.push({ label: "S" + String(f).padStart(2, "0"),
                    cls: "skp" + (filterSel === NF + f ? " sel" : ""),
                    v: mode === "diff" ? -scratch[NF + f] : scratch[NF + f] });
  }
  contribs.sort((a, b) => b.v - a.v);
  const value = mode === "diff" ? valDiff(sp, k) : valSum(sp, k);

  const c0 = sp.cols[k], c5 = sp.cols[k + 5];
  const gap = c5 - c0 !== 5;
  const modeLabel = filterSel < 0
    ? (mode === "diff" ? "Incl − Skip" : "Incl + Skip")
    : filterName(filterSel) + " contribution";
  tooltip.innerHTML =
    `<div class="tt-species">${DATA.species[s].replace(/_/g, " ")}</div>` +
    `<div class="tt-meta">window ${k} &middot; aligned cols ${c0}–${c5}` +
    (gap ? ` <span class="tt-gap">spans gaps</span>` : "") + `</div>` +
    `<div class="tt-mer">${sp.seq.slice(k, k + 6)}</div>` +
    `<div class="tt-val">${modeLabel} = <b>${value.toFixed(4)}</b></div>` +
    `<div id="contribs">` +
    contribs.map(c =>
      `<span class="${c.cls}">${c.label} ${(c.v >= 0 ? "+" : "−") +
        Math.abs(c.v).toFixed(3)}</span>`).join("") +
    `</div>`;

  tooltip.style.display = "block";
  const tw = tooltip.offsetWidth, th = tooltip.offsetHeight;
  let x = clientX + 14, y = clientY + 14;
  if (x + tw > window.innerWidth - 8) x = clientX - tw - 14;
  if (y + th > window.innerHeight - 8) y = Math.max(8, clientY - th - 14);
  tooltip.style.left = x + "px";
  tooltip.style.top = y + "px";

  cellbox.style.display = "block";
  cellbox.style.left = col * cellW - 1 + "px";
  cellbox.style.top = s * CELLH - 1 + "px";
  cellbox.style.width = cellW + 2 + "px";
  cellbox.style.height = CELLH + 2 + "px";
}

function hideHover() {
  if (pinned) return;
  tooltip.style.display = "none";
  cellbox.style.display = "none";
}

function showSDTooltip(col, clientX, clientY) {
  const modeLabel = filterSel < 0
    ? (mode === "diff" ? "Incl − Skip" : "Incl + Skip")
    : filterName(filterSel) + " contribution";
  tooltip.innerHTML =
    `<div class="tt-species">cross-species SD</div>` +
    `<div class="tt-meta">aligned col ${col} &middot; ${sdCnt[col]} species with a window here</div>` +
    `<div class="tt-val">SD of ${modeLabel} = <b>${sdVals[col].toFixed(4)}</b></div>`;
  tooltip.style.display = "block";
  const tw = tooltip.offsetWidth, th = tooltip.offsetHeight;
  let x = clientX + 14, y = clientY + 14;
  if (x + tw > window.innerWidth - 8) x = clientX - tw - 14;
  if (y + th > window.innerHeight - 8) y = Math.max(8, clientY - th - 14);
  tooltip.style.left = x + "px";
  tooltip.style.top = y + "px";
  cellbox.style.display = "none";
}

sdCanvas.addEventListener("mousemove", e => {
  if (pinned) return;
  const rect = sdCanvas.getBoundingClientRect();
  const col = Math.floor((e.clientX - rect.left) / cellW);
  if (col < 0 || col >= NA || Number.isNaN(sdVals[col])) { hideHover(); return; }
  showSDTooltip(col, e.clientX, e.clientY);
});
sdCanvas.addEventListener("mouseleave", hideHover);
sdCanvas.addEventListener("click", () => {
  if (pinned) { pinned = false; cellbox.classList.remove("pinned"); hideHover(); return; }
  if (tooltip.style.display === "block") pinned = true;
});

hm.addEventListener("mousemove", e => {
  if (pinned) return;
  const rect = hm.getBoundingClientRect();
  const col = Math.floor((e.clientX - rect.left) / cellW);
  const row = Math.floor((e.clientY - rect.top) / CELLH);
  if (col < 0 || col >= NA || row < 0 || row >= NS) { hideHover(); return; }
  showTooltip(row, col, e.clientX, e.clientY);
});
hm.addEventListener("mouseleave", hideHover);
hm.addEventListener("click", e => {
  if (pinned) { pinned = false; cellbox.classList.remove("pinned"); hideHover(); return; }
  if (tooltip.style.display === "block") { pinned = true; cellbox.classList.add("pinned"); }
});
document.addEventListener("keydown", e => {
  if (e.key === "Escape" && pinned) {
    pinned = false;
    cellbox.classList.remove("pinned");
    hideHover();
  }
});

// ---- controls ----
document.getElementById("modeSeg").addEventListener("click", e => {
  const b = e.target.closest("button");
  if (!b || b.dataset.mode === mode) return;
  mode = b.dataset.mode;
  for (const x of e.currentTarget.children) x.classList.toggle("on", x === b);
  renderHeatmap();
  renderLegend();
  computeSD();
  renderSD();
  renderSDLegend();
  hideHover();
});
const filterPick = document.getElementById("filterPick");
{
  const all = document.createElement("option");
  all.value = "-1";
  all.textContent = "all 40 filters";
  filterPick.appendChild(all);
  for (const [bank, label] of [[0, "inclusion"], [1, "skipping"]]) {
    const og = document.createElement("optgroup");
    og.label = label;
    for (let f = 0; f < NF; f++) {
      const o = document.createElement("option");
      o.value = String(bank * NF + f);
      o.textContent = filterName(bank * NF + f);
      og.appendChild(o);
    }
    filterPick.appendChild(og);
  }
}
filterPick.addEventListener("change", () => {
  filterSel = +filterPick.value;
  if (filterSel >= 0) computeFilterActs(filterSel);
  percentileScales();
  renderHeatmap();
  renderLegend();
  computeSD();
  renderSD();
  renderSDLegend();
  hideHover();
});

document.getElementById("zoomSeg").addEventListener("click", e => {
  const b = e.target.closest("button");
  if (!b) return;
  cellW = +b.dataset.z;
  for (const x of e.currentTarget.children) x.classList.toggle("on", x === b);
  applyZoom();
});

// ---- self-check against torch-computed reference cells ----
function runSelfCheck() {
  const badge = document.getElementById("badge");
  let fails = 0;
  for (const r of DATA.refs) {
    const sp = SP[r.s];
    const diff = sp.aIncl[r.k] - sp.aSkip[r.k];
    const sum = sp.aIncl[r.k] + sp.aSkip[r.k];
    const ok = Math.abs(diff - r.diff) < 1e-3 && Math.abs(sum - r.sum) < 1e-3 &&
               sp.seq.slice(r.k, r.k + 6) === r.sixmer && sp.cols[r.k] === r.col;
    console.assert(ok, "self-check failed", r, { diff, sum });
    if (!ok) fails++;
    // The tooltip contributions must sum exactly to the cell value.
    windowActs(sp.codes, r.k, scratch);
    let sI = 0, sS = 0;
    for (let f = 0; f < NF; f++) { sI += scratch[f]; sS += scratch[NF + f]; }
    console.assert(Math.abs(sI - sS - diff) < 1e-4, "contribution-sum mismatch", r);
  }
  const r2 = DATA.sd_ref;  // computed in Python in diff mode, all filters
  if (r2 && mode === "diff" && filterSel < 0) {
    const ok = sdCnt[r2.col] === r2.n && Math.abs(sdVals[r2.col] - r2.sd) < 1e-3;
    console.assert(ok, "SD self-check failed", r2, { n: sdCnt[r2.col], sd: sdVals[r2.col] });
    if (!ok) fails++;
  }
  badge.textContent = fails ? `self-check: FAIL (${fails})` : "self-check: OK";
  badge.className = fails ? "fail" : "ok";
}

// ---- boot ----
renderLabels();
applyZoom();
const splash = document.getElementById("splash");
const splashpct = document.getElementById("splashpct");
let sNext = 0;
function computeChunk() {
  const t0 = performance.now();
  while (sNext < NS && performance.now() - t0 < 40) computeSpecies(sNext++);
  if (sNext < NS) {
    splashpct.textContent = Math.round(100 * sNext / NS) + "%";
    requestAnimationFrame(computeChunk);
  } else {
    percentileScales();
    renderHeatmap();
    renderLegend();
    computeSD();
    renderSD();
    renderSDLegend();
    runSelfCheck();
    splash.remove();
  }
}
requestAnimationFrame(computeChunk);
</script>
</body>
</html>
"""


if __name__ == "__main__":
    main()
