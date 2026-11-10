"""Interactive view of one commutation loop: hover a tile of the start section and see where both paths send it.

A loop's viewer file (bench/scripts/cross_align.py final, crossloop.view_arrays) holds each section's positions and
organs, a frame transform into the start section's frame, and per start tile the top-k of every kernel: each path's
first step (to its intermediate section) and each path's composition (to the target). loop_view renders it as
self-contained HTML + JS (canvas, no extra dependency), so the hover runs in the browser and works in a saved
notebook too:
- square: start | path 1's intermediate over path 2's | target; triangle: start | the one intermediate | target;
- path 1 (own species first, then across) in orange, path 2 (across first) in blue; on the target both add up, so
  where the paths agree the points turn white;
- hover picks the nearest start tile; the brush slider sums the tiles within a radius (mass-weighted); click pins;
- toggles: organ colours under the highlight, the per-tile commutativity on the start section, and the frame (each
  section's own coordinates, or rotated / reflected into the start's frame along the path couplings).
viewer(out) wraps it in ipywidgets drop-downs over the files in <out>/final/view.
"""
import base64
import html
import json
import uuid
from pathlib import Path

import numpy as np

from mgw.bench import metrics

PATH1, PATH2 = "#ff8c1e", "#2f8cff"


def _b64(a, dtype):
    return base64.b64encode(np.ascontiguousarray(np.asarray(a).astype(dtype)).tobytes()).decode("ascii")


def _palette(organ_names):
    from mgw.bench.report.figures import OTHER, organ_palette
    pal = organ_palette(organ_names)
    return [pal.get(o, OTHER) for o in metrics.ORGANS]


def payload(z):
    """The JSON the page needs, from a viewer file (np.load result or dict)."""
    z = {k: z[k] for k in z.keys()} if not isinstance(z, dict) else z
    names = [str(s) for s in z["sections"]]
    organs = {s: np.asarray(z[f"{s}|organ"]).astype(str) for s in names}
    secs = {}
    for s in names:
        xy = np.asarray(z[f"{s}|xy"], np.float64)
        R, t = np.asarray(z[f"{s}|R"], np.float64), np.asarray(z[f"{s}|t"], np.float64)
        secs[s] = dict(n=len(xy), xy=_b64(xy, "<f4"), xa=_b64(xy @ R + t, "<f4"),
                       organ=_b64(metrics.organ_codes(organs[s]), "<u1"))
    k = int(np.asarray(z["p1a_idx"]).shape[1])
    kern = {}
    for tag in ("p1a", "p1b", "p2a", "p2b"):
        if f"{tag}_idx" in z:
            kern[tag] = dict(idx=_b64(z[f"{tag}_idx"], "<u2"), val=_b64(z[f"{tag}_val"], "<u1"),
                             max=_b64(z[f"{tag}_max"], "<f4"))
    steps = lambda key: [s.split(":")[0] for s in np.asarray(z[key]).astype(str)]
    kept = "; ".join(f"path {t[1]}: " + " / ".join(f"{100 * float(np.mean(z[f'{t}{j}_kept'])):.0f}%"
                                                   for j in "ab" if f"{t}{j}_kept" in z) for t in ("p1", "p2"))
    return dict(title=f"{z['loop']} ({z['kind']}), from {'human' if str(z['direction']) == 'h' else 'mouse'}",
                run=str(z.get("run", "")), temporal=str(z.get("temporal", "")), start=str(z["start"]),
                target=str(z["target"]), inter1=str(z["inter1"]), inter2=str(z["inter2"]), k=k, sections=secs,
                kernels=kern, comm=_b64(z["comm"], "<f4"), w=_b64(z["w"], "<f4"), steps1=steps("steps1"),
                steps2=steps("steps2"), kept=kept, organs=list(metrics.ORGANS), palette=_palette(list(organs.values())),
                path1=PATH1, path2=PATH2)


_TEMPLATE = r"""
<div id="__ID__" class="lv">
<style>
#__ID__ { font: 12px/1.4 system-ui, -apple-system, sans-serif; color: #e8e7e3; background: #141413; padding: 12px;
          border-radius: 8px; max-width: 1180px; }
#__ID__ .lv-head { display: flex; flex-wrap: wrap; gap: 6px 18px; align-items: baseline; margin-bottom: 8px; }
#__ID__ .lv-title { font-size: 14px; font-weight: 600; }
#__ID__ .lv-sub { color: #a3a29d; }
#__ID__ .lv-grid { display: flex; gap: 10px; align-items: center; }
#__ID__ .lv-col { display: flex; flex-direction: column; gap: 10px; }
#__ID__ .lv-panel { position: relative; }
#__ID__ canvas { background: #0b0b0a; border-radius: 6px; display: block; }
#__ID__ .lv-lab { position: absolute; left: 8px; top: 6px; font-size: 11px; color: #c9c8c2; pointer-events: none;
                  text-shadow: 0 0 3px #000; }
#__ID__ .lv-arrow { color: #a3a29d; font-size: 11px; text-align: center; min-width: 38px; }
#__ID__ .lv-arrow b { display: block; font-size: 16px; }
#__ID__ .lv-ctrl { display: flex; flex-wrap: wrap; gap: 6px 16px; align-items: center; margin-top: 10px; }
#__ID__ .lv-ctrl label { display: inline-flex; gap: 5px; align-items: center; cursor: pointer; }
#__ID__ .lv-read { margin-top: 8px; min-height: 3.2em; color: #d6d5d0; }
#__ID__ .sw { display: inline-block; width: 10px; height: 10px; border-radius: 2px; margin-right: 4px;
              vertical-align: -1px; }
</style>
<div class="lv-head"><span class="lv-title"></span><span class="lv-sub"></span></div>
<div class="lv-grid"></div>
<div class="lv-ctrl">
  <span><span class="sw" style="background:__P1__"></span><span class="lv-p1"></span></span>
  <span><span class="sw" style="background:__P2__"></span><span class="lv-p2"></span></span>
  <span class="lv-sub">overlap turns white</span>
  <label><input type="checkbox" class="lv-org" checked> organ colours</label>
  <label><input type="checkbox" class="lv-comm"> commutativity on start</label>
  <label><input type="checkbox" class="lv-frame" checked> common frame</label>
  <label>brush <input type="range" class="lv-brush" min="0" max="40" value="0"> <span class="lv-bv">0</span> px</label>
</div>
<div class="lv-read">Hover over the start section; click to pin a tile.</div>
<script type="application/json" class="lv-data">__DATA__</script>
</div>
<script>
(function () {
  const root = document.getElementById("__ID__");
  const D = JSON.parse(root.querySelector(".lv-data").textContent);
  const dec = (s, T) => { const b = atob(s), u = new Uint8Array(b.length);
    for (let i = 0; i < b.length; i++) u[i] = b.charCodeAt(i); return new T(u.buffer); };
  const hex = h => [parseInt(h.slice(1, 3), 16), parseInt(h.slice(3, 5), 16), parseInt(h.slice(5, 7), 16)];
  const P1 = hex(D.path1), P2 = hex(D.path2), PAL = D.palette.map(hex);
  const S = {};
  for (const [name, s] of Object.entries(D.sections))
    S[name] = { n: s.n, xy: dec(s.xy, Float32Array), xa: dec(s.xa, Float32Array), organ: dec(s.organ, Uint8Array) };
  const K = {};
  for (const [t, k] of Object.entries(D.kernels))
    K[t] = { idx: dec(k.idx, Uint16Array), val: dec(k.val, Uint8Array), max: dec(k.max, Float32Array) };
  const comm = dec(D.comm, Float32Array), w = dec(D.w, Float32Array), k = D.k;
  root.querySelector(".lv-title").textContent = D.title;
  root.querySelector(".lv-sub").textContent = `cross ${D.run} · temporal ${D.temporal}`;
  root.querySelector(".lv-p1").textContent = "path 1: " + D.steps1.join(" → ");
  root.querySelector(".lv-p2").textContent = "path 2: " + D.steps2.join(" → ");
  const square = D.inter1 && D.inter2;
  const big = square ? 330 : 300, small = square ? 230 : 300;
  const grid = root.querySelector(".lv-grid");
  const panels = {};
  function panel(name, role, size, label) {
    const d = document.createElement("div"); d.className = "lv-panel";
    const c = document.createElement("canvas"); const r = window.devicePixelRatio || 1;
    c.width = size * r; c.height = size * r; c.style.width = size + "px"; c.style.height = size + "px";
    const l = document.createElement("div"); l.className = "lv-lab"; l.textContent = label;
    d.appendChild(c); d.appendChild(l);
    panels[role] = { name, role, c, ctx: c.getContext("2d"), size, r };
    return d;
  }
  function arrow(txt) { const a = document.createElement("div"); a.className = "lv-arrow";
    a.innerHTML = "<b>→</b>" + txt; return a; }
  grid.appendChild(panel(D.start, "start", big, "start · " + D.start));
  if (square) {
    const col = document.createElement("div"); col.className = "lv-col";
    col.appendChild(panel(D.inter1, "i1", small, "path 1 · " + D.inter1));
    col.appendChild(panel(D.inter2, "i2", small, "path 2 · " + D.inter2));
    const a1 = document.createElement("div"); a1.className = "lv-col";
    a1.appendChild(arrow(D.steps1[0])); a1.appendChild(arrow(D.steps2[0]));
    grid.appendChild(a1); grid.appendChild(col);
    const a2 = document.createElement("div"); a2.className = "lv-col";
    a2.appendChild(arrow(D.steps1[1])); a2.appendChild(arrow(D.steps2[1]));
    grid.appendChild(a2);
  } else {
    const role = D.inter1 ? "i1" : "i2", nm = D.inter1 || D.inter2;
    grid.appendChild(arrow(D.inter1 ? D.steps1[0] : D.steps2[0]));
    grid.appendChild(panel(nm, role, small, (D.inter1 ? "path 1 · " : "path 2 · ") + nm));
    grid.appendChild(arrow(D.inter1 ? D.steps1[1] + " / " + D.steps2[0] : D.steps2[1] + " / " + D.steps1[0]));
  }
  grid.appendChild(panel(D.target, "target", big, "target · " + D.target));

  const ui = { org: root.querySelector(".lv-org"), comm: root.querySelector(".lv-comm"),
               frame: root.querySelector(".lv-frame"), brush: root.querySelector(".lv-brush"),
               bv: root.querySelector(".lv-bv"), read: root.querySelector(".lv-read") };
  // pixel transform of a section in the current frame: equal aspect, padded
  function layout(p) {
    const s = S[p.name], xy = ui.frame.checked ? s.xa : s.xy;
    let x0 = Infinity, x1 = -Infinity, y0 = Infinity, y1 = -Infinity;
    for (let i = 0; i < s.n; i++) { const x = xy[2 * i], y = xy[2 * i + 1];
      if (x < x0) x0 = x; if (x > x1) x1 = x; if (y < y0) y0 = y; if (y > y1) y1 = y; }
    const W = p.size * p.r, pad = 10 * p.r, sc = (W - 2 * pad) / Math.max(x1 - x0, y1 - y0, 1e-9);
    const ox = (W - sc * (x1 - x0)) / 2, oy = (W - sc * (y1 - y0)) / 2;
    p.px = new Float32Array(2 * s.n);
    for (let i = 0; i < s.n; i++) { p.px[2 * i] = ox + sc * (xy[2 * i] - x0); p.px[2 * i + 1] = oy + sc * (xy[2 * i + 1] - y0); }
    p.dot = Math.max(1.5 * p.r, Math.min(7 * p.r, 0.95 * Math.sqrt((sc * (x1 - x0)) * (sc * (y1 - y0)) * 0.55 / s.n)));
    if (p.role === "start") {               // grid lookup for hover
      const g = Math.max(4 * p.r, p.dot * 2); p.g = g; p.cells = new Map();
      for (let i = 0; i < s.n; i++) { const key = Math.floor(p.px[2 * i] / g) + "," + Math.floor(p.px[2 * i + 1] / g);
        if (!p.cells.has(key)) p.cells.set(key, []); p.cells.get(key).push(i); }
    }
  }
  const commMax = (() => { const a = Array.from(comm).sort((x, y) => x - y); return a[Math.floor(0.97 * (a.length - 1))] || 1; })();
  function ramp(t) {  // dark blue -> teal -> yellow
    const st = [[40, 30, 110], [30, 140, 160], [120, 200, 90], [250, 230, 60]];
    t = Math.max(0, Math.min(1, t)) * (st.length - 1); const i = Math.min(st.length - 2, Math.floor(t)), f = t - i;
    return st[i].map((v, j) => Math.round(v + f * (st[i + 1][j] - v)));
  }
  function base(p, dim) {
    const s = S[p.name], ctx = p.ctx, d = p.dot, fa = dim ? 0.13 : 0.3;
    ctx.globalCompositeOperation = "source-over"; ctx.clearRect(0, 0, p.c.width, p.c.height);
    for (let i = 0; i < s.n; i++) {
      let col;
      if (p.role === "start" && ui.comm.checked) { const c = ramp(comm[i] / commMax); col = `rgba(${c[0]},${c[1]},${c[2]},0.95)`; }
      else if (ui.org.checked) { const c = PAL[s.organ[i]]; col = `rgba(${c[0]},${c[1]},${c[2]},${fa})`; }
      else col = `rgba(120,120,115,${fa})`;
      ctx.fillStyle = col; ctx.fillRect(p.px[2 * i] - d / 2, p.px[2 * i + 1] - d / 2, d, d);
    }
  }
  function glow(p, acc, rgb) {
    let mx = 0; for (let j = 0; j < acc.length; j++) if (acc[j] > mx) mx = acc[j];
    if (mx <= 0) return;
    const ctx = p.ctx, d = p.dot * 1.5; ctx.globalCompositeOperation = "lighter";
    for (let j = 0; j < acc.length; j++) { const f = acc[j] / mx; if (f < 0.01) continue;
      const a = 0.25 + 0.75 * Math.sqrt(f);
      ctx.fillStyle = `rgba(${rgb[0]},${rgb[1]},${rgb[2]},${a.toFixed(3)})`;
      ctx.fillRect(p.px[2 * j] - d / 2, p.px[2 * j + 1] - d / 2, d, d); }
    ctx.globalCompositeOperation = "source-over";
  }
  function accumulate(kr, sel, n) {
    const acc = new Float32Array(n);
    for (const [i, wi] of sel) for (let j = 0; j < k; j++) {
      const v = kr.val[i * k + j]; if (v) acc[kr.idx[i * k + j]] += wi * v / 255 * kr.max[i]; }
    return acc;
  }
  // a path from the selection: [mass on its intermediate section or null, mass on the target]
  function pathMass(tag, inter, sel) {
    const a = K[tag + "a"], b = K[tag + "b"], nT = S[D.target].n;
    if (!b) return [null, accumulate(a, sel, nT)];
    const mid = accumulate(a, sel, S[inter].n), out = new Float32Array(nT);
    for (let j = 0; j < mid.length; j++) { const m = mid[j]; if (!m) continue;
      for (let q = 0; q < k; q++) { const v = b.val[j * k + q]; if (v) out[b.idx[j * k + q]] += m * v / 255 * b.max[j]; } }
    return [mid, out];
  }
  function organMix(acc, s) {
    const m = new Float64Array(D.organs.length); let tot = 0;
    for (let j = 0; j < acc.length; j++) { m[s.organ[j]] += acc[j]; tot += acc[j]; }
    return Array.from(m.keys()).sort((a, b) => m[b] - m[a]).slice(0, 3).filter(o => m[o] > 0)
      .map(o => `${D.organs[o]} ${(100 * m[o] / tot).toFixed(0)}%`).join(", ");
  }
  let sel = [], pinned = false, cursor = null;
  function draw() {
    for (const p of Object.values(panels)) base(p, sel.length > 0 && !(p.role === "start" && ui.comm.checked));
    const ps = panels.start;
    if (sel.length) {
      const tS = S[D.target];
      const [m1, a1] = pathMass("p1", D.inter1, sel), [m2, a2] = pathMass("p2", D.inter2, sel);
      glow(panels.target, a1, P1); glow(panels.target, a2, P2);
      if (panels.i1 && m1) glow(panels.i1, m1, P1);
      if (panels.i2 && m2) glow(panels.i2, m2, P2);
      const ctx = ps.ctx, d = ps.dot * 1.3; ctx.fillStyle = "#ffffff";
      for (const [i] of sel) ctx.fillRect(ps.px[2 * i] - d / 2, ps.px[2 * i + 1] - d / 2, d, d);
      if (sel.length === 1) { const i = sel[0][0]; ctx.strokeStyle = "#ffffff"; ctx.lineWidth = 1.5 * ps.r;
        ctx.beginPath(); ctx.arc(ps.px[2 * i], ps.px[2 * i + 1], 4 * ps.dot, 0, 2 * Math.PI); ctx.stroke(); }
      const i0 = sel[0][0], so = S[D.start];
      let cm = 0, ws = 0; for (const [i, wi] of sel) { cm += wi * comm[i]; ws += wi; }
      ui.read.innerHTML = `<b>${sel.length > 1 ? sel.length + " tiles around " : "tile "}${i0}</b> · ${D.organs[so.organ[i0]]}` +
        ` · commutativity ${(cm / ws).toFixed(3)} (target RMS radii)${pinned ? " · pinned" : ""}<br>` +
        `<span class="sw" style="background:${D.path1}"></span>path 1 reaches: ${organMix(a1, tS)}<br>` +
        `<span class="sw" style="background:${D.path2}"></span>path 2 reaches: ${organMix(a2, tS)}` +
        `<br><span class="lv-sub">top-${k} per step keeps ${D.kept}</span>`;
    }
    if (cursor && +ui.brush.value > 0) { const ctx = ps.ctx; ctx.strokeStyle = "rgba(255,255,255,0.6)";
      ctx.lineWidth = ps.r; ctx.beginPath(); ctx.arc(cursor[0], cursor[1], +ui.brush.value * ps.r, 0, 2 * Math.PI); ctx.stroke(); }
  }
  function pick(ev) {
    const p = panels.start, b = p.c.getBoundingClientRect();
    const x = (ev.clientX - b.left) * p.r, y = (ev.clientY - b.top) * p.r; cursor = [x, y];
    const rad = Math.max(+ui.brush.value * p.r, 0), reach = Math.max(rad, p.g);
    const gx = Math.floor(x / p.g), gy = Math.floor(y / p.g), span = Math.ceil(reach / p.g);
    let best = -1, bd = Infinity; const out = [];
    for (let a = gx - span; a <= gx + span; a++) for (let c = gy - span; c <= gy + span; c++) {
      const cell = p.cells.get(a + "," + c); if (!cell) continue;
      for (const i of cell) { const dx = p.px[2 * i] - x, dy = p.px[2 * i + 1] - y, dd = Math.hypot(dx, dy);
        if (dd < bd) { bd = dd; best = i; } if (rad > 0 && dd <= rad) out.push([i, w[i]]); } }
    if (rad > 0 && out.length) return out;
    return best >= 0 && bd < 6 * p.dot ? [[best, 1]] : [];
  }
  const startCanvas = panels.start.c;
  startCanvas.addEventListener("mousemove", ev => { if (pinned) return; sel = pick(ev); draw(); });
  startCanvas.addEventListener("mouseleave", () => { if (pinned) return; cursor = null; draw(); });
  startCanvas.addEventListener("click", ev => { pinned = !pinned; if (!pinned) sel = pick(ev); draw(); });
  function relayout() { for (const p of Object.values(panels)) layout(p); draw(); }
  ui.org.addEventListener("change", draw); ui.comm.addEventListener("change", draw);
  ui.frame.addEventListener("change", () => { sel = []; pinned = false; relayout(); });
  ui.brush.addEventListener("input", () => { ui.bv.textContent = ui.brush.value; });
  relayout();
  if (D.select !== null && D.select !== undefined) { sel = [[D.select, 1]]; pinned = true; draw(); }
})();
</script>
"""


def loop_view(z, select=None):
    """Self-contained HTML of one viewer file (path, np.load result or dict); `select`: a start tile to pin."""
    if isinstance(z, (str, Path)):
        z = np.load(z, allow_pickle=False)
    data = json.dumps(dict(payload(z), select=None if select is None else int(select))).replace("</", "<\\/")
    return (_TEMPLATE.replace("__ID__", "lv" + uuid.uuid4().hex[:10]).replace("__P1__", PATH1)
            .replace("__P2__", PATH2).replace("__DATA__", data))


def save_page(z, path, title="Loop viewer", select=None):
    """A standalone HTML page of one viewer file (for a browser outside the notebook)."""
    page = (f"<!doctype html><html><head><meta charset='utf-8'><title>{html.escape(title)}</title></head>"
            f"<body style='background:#141413;margin:16px'>{loop_view(z, select)}</body></html>")
    Path(path).write_text(page)
    return Path(path)


def view_files(out):
    """{(loop, direction, cross run, temporal run): file} of <out>/final/view."""
    files = {}
    for f in sorted((Path(out) / "final" / "view").glob("*/*.npz")):
        loop, direction = f.parent.name.rsplit("_", 1)
        run, temporal = f.stem.split("__", 1)
        files[loop, direction, run, temporal] = f
    return files


def viewer(out, loop=None, direction="h", run=None, temporal=None):
    """Drop-downs (loop, direction, cross run, temporal run) over the viewer files; the page below them."""
    import ipywidgets as W
    from IPython.display import HTML, display
    files = view_files(out)
    if not files:
        raise FileNotFoundError(f"no viewer files in {Path(out) / 'final' / 'view'} (cross_align.py final)")
    keys = list(files)
    opts = [sorted({k[i] for k in keys}, key=str) for i in range(4)]
    loops_sorted = sorted(opts[0], key=lambda s: [int(x[1:]) for x in s.split("-")])
    dd = [W.Dropdown(options=loops_sorted, value=loop if loop in opts[0] else loops_sorted[0], description="loop"),
          W.Dropdown(options=[("from human", "h"), ("from mouse", "m")], value=direction, description="start"),
          W.Dropdown(options=opts[2], value=run if run in opts[2] else opts[2][0], description="cross"),
          W.Dropdown(options=opts[3], value=temporal if temporal in opts[3] else opts[3][0], description="temporal")]
    display(W.HBox(dd[:2]), W.HBox(dd[2:]))
    page = display(HTML(""), display_id=True)          # a display handle, not an Output widget: its updates
                                                       # re-render the page (scripts included) in any frontend

    def show(*_):
        key = tuple(d.value for d in dd)
        page.update(HTML(loop_view(files[key])) if key in files else HTML(f"<p>no viewer file for {key}</p>"))
    for d in dd:
        d.observe(show, names="value")
    show()
