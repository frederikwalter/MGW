"""The alignment viewer: one self-contained HTML file per (pair, seed), no server and no CDN.

It shows the query and the target side by side, coloured by organ. Clicking or brushing query spots paints the
target by where their mass goes (and the reverse); the query can be coloured by the transferred organs, the
confidence or the fold-overs. Below are the organ x organ matrix C of the shown coupling (click a cell to highlight
its spots) and the candidate grid of the direction (kappa_bal per query x target, selected and oracle marked), so the
regret is visible at a glance.

Embedded couplings: A_fgw selected / oracle / L_partner, as the top-k entries per row and per
column (k_view of the stored top-64, renormalized, quantized to 8 bits). Every other candidate carries its matrix C
and its scores only.
"""
import base64
import json
from pathlib import Path

import numpy as np
import pandas as pd

from mgw.bench import metrics, prepare, store
from mgw.bench.report.figures import OTHER, SLOTS, organ_palette

TEMPLATE = Path(__file__).with_name("viewer_template.html")
METRIC_KEYS = ["kappa_bal", "oms", "kappa", "K", "objective", "acc_hard", "foldover", "sharpness", "coexpr_r",
               "coverage", "mass_unscored", "kappa_bal_red"]


def b64(a):
    return base64.b64encode(np.ascontiguousarray(a).tobytes()).decode()


def _xy(xy):
    xy = np.asarray(xy, dtype=np.float64)
    xy = xy - xy.mean(0)
    xy = xy / (np.sqrt((xy ** 2).sum(1).mean()) or 1)
    return b64(np.round(xy * 8000).astype(np.int16))


def _labels(lab):
    return b64(metrics.organ_codes(lab).astype(np.uint8))


def _topk(idx, val, k):
    """First k (by value) of stored top-k, renormalized per row and quantized to 8 bits."""
    idx, val = np.asarray(idx), np.asarray(val, dtype=np.float32)
    order = np.argsort(-val, axis=1)[:, :k]
    idx, val = np.take_along_axis(idx, order, 1), np.take_along_axis(val, order, 1)
    val = val / np.maximum(val.sum(1, keepdims=True), 1e-12)
    return b64(idx.astype(np.uint16)), b64(np.round(val * 255).astype(np.uint8)), int(idx.shape[1])


def _metrics(r):
    out = {}
    for k in METRIC_KEYS:
        v = r.get(k)
        if v is not None and not (isinstance(v, float) and not np.isfinite(v)):
            out[k] = round(float(v), 4)
    return out


def collect(cfg, pair, seed, root, k_view=8):
    sh = cfg.section_hash()
    data = dict(meta=dict(pair=pair.id, kind=pair.kind, seed=seed, note=pair.note, stage=pair.stage,
                          overlap=pair.overlap, organs=list(metrics.ORGANS), k=k_view, pipelines={}),
                sections={}, targets={}, directions=[], grid={}, couplings={}, cmat={})
    labels = []
    for side, e in (("a", pair.a), ("b", pair.b)):
        for s in e.sections:
            if s in data["sections"]:
                continue
            sec = prepare.load_section(root, s, seed, sh)
            data["sections"][s] = dict(side=side, embryo=e.name, n=sec.n, xy=_xy(sec.xy), label=_labels(sec.label),
                                       L=None if not np.isfinite(sec.L) else round(sec.L, 3))
            labels.append(sec.label)
    pal = organ_palette(labels)
    data["meta"]["palette"] = pal
    data["meta"]["slots"], data["meta"]["other"] = SLOTS, OTHER
    for direction in pair.directions:
        qe, te = pair.sides(direction)
        data["directions"].append(dict(id=direction, label=pair.direction_label(direction),
                                       queries=list(qe.sections), targets=list(te.sections)))
    for pipe in ("A_fgw",):
        if pipe not in cfg.raw["pipelines"] or pair.kind not in cfg.raw["pipelines"][pipe]["kinds"]:
            continue
        d = store.result_dir(root, cfg, pipe, pair.id, seed)
        if not store.is_done(d, cfg, pipe):
            continue
        data["meta"]["pipelines"][pipe] = cfg.pipeline_hash(pipe)
        df = pd.read_csv(d / "scores.csv")
        cand = np.load(d / "candidates.npz")
        if (d / "targets.npz").exists():
            tz = np.load(d / "targets.npz")
            for kk in tz.files:
                if kk.endswith("/xy"):
                    direction, name, _ = kk.split("/")
                    data["targets"][f"{pipe}|{direction}|{name}"] = dict(
                        n=len(tz[kk]), xy=_xy(tz[kk]), label=_labels(tz[f"{direction}/{name}/label"]))
        for direction in pair.directions:
            dd = df[df.direction == direction]
            if pipe == "A_fgw":
                g = dd[dd.role == "candidate"]
                cells = {}
                for r in g.to_dict("records"):
                    cells.setdefault(r["query"], {})[r["target"]] = _metrics(r)
                    ck = store.key(direction, r["query"], r["target"], "C_full")
                    data["cmat"][f"{pipe}|{direction}|{r['query']}|{r['target']}"] = np.round(
                        cand[ck].astype(np.float64), 2).tolist()
                roles = {}
                for r in dd[dd.role.isin(["selected", "oracle", "L_partner"])].to_dict("records"):
                    roles.setdefault(r["query"], {})[r["role"]] = r["target"]
                data["grid"][f"{pipe}|{direction}"] = dict(cells=cells, roles=roles)
            for r in dd[dd.role.isin(["selected", "oracle", "L_partner"])].to_dict("records"):
                key = f"{pipe}|{direction}|{r['query']}|{r['target']}"
                kk = lambda f: store.key(direction, r["query"], r["target"], f)
                if key in data["couplings"]:
                    data["couplings"][key]["roles"].append(r["role"])
                    continue
                if kk("top_row_idx") not in cand.files:
                    continue
                ri, rv, k = _topk(cand[kk("top_row_idx")], cand[kk("top_row_val")], k_view)
                ci, cv, kc = _topk(cand[kk("top_col_idx")], cand[kk("top_col_val")], k_view)
                tname = r["target"] if pipe == "A_fgw" else f"{pipe}|{direction}|{r['target']}"
                data["couplings"][key] = dict(
                    pipeline=pipe, direction=direction, query=r["query"], target=r["target"], target_ref=tname,
                    roles=[r["role"]], metrics=_metrics(r), k_row=k, k_col=kc, row_idx=ri, row_val=rv, col_idx=ci,
                    col_val=cv, pred=b64(cand[kk("pred")]), conf=b64(np.round(cand[kk("conf")].astype(
                        np.float32) * 255).astype(np.uint8)), folded=b64(cand[kk("folded")]),
                    C=np.round(cand[kk("C_full")].astype(np.float64), 2).tolist())
    return data


def build(cfg, pair, seed, root, out=None, k_view=8):
    data = collect(cfg, pair, seed, root, k_view)
    out = Path(out) if out else Path(root) / "reports" / "viewers" / f"{store.safe(pair.id)}_seed{seed}.html"
    out.parent.mkdir(parents=True, exist_ok=True)
    html = TEMPLATE.read_text().replace("/*__DATA__*/null", json.dumps(data, separators=(",", ":")))
    html = html.replace("__TITLE__", f"{pair.id} alignments")
    out.write_text(html)
    return out
