"""PNG contact sheets: one per (pair, direction, pipeline), one row per query.

Columns: query organs | target organs | organs transferred to the query (arg-max of P) | confidence of the transfer |
the organ x organ matrix C (rows normalized: the share of each query organ's mass that lands on each target organ).
Organs: the eight most frequent of the pair in fixed categorical slots, the rest folded into a grey "Other".
"""
from pathlib import Path

import numpy as np

from mgw.bench import metrics, prepare, store

# validated categorical slots (light surface), in fixed order; "Other" is grey
SLOTS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
OTHER = "#c9c8c2"
BLUES = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]


def organ_palette(label_sets, n=8):
    """The n most frequent scored organs over all label sets -> colors; everything else is Other."""
    counts = {}
    for labels in label_sets:
        for o, c in zip(*np.unique(np.asarray(labels, dtype=str), return_counts=True)):
            if metrics.SCORED[metrics.ORGAN_INDEX.get(o, metrics.ORGAN_INDEX["Other"])]:
                counts[o] = counts.get(o, 0) + c
    top = sorted(counts, key=lambda o: -counts[o])[:n]
    return {o: SLOTS[i] for i, o in enumerate(top)}


def _colors(labels, pal):
    return [pal.get(l, OTHER) for l in np.asarray(labels, dtype=str)]


def contact_sheet(pair, pipeline, direction, d, rows, sections, targets, pal, out, role="selected"):
    """rows: the scores rows (one per query) of `role`; sections / targets: name -> (xy, labels)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    from matplotlib.lines import Line2D

    cand = np.load(d / "candidates.npz")
    blues = LinearSegmentedColormap.from_list("blues", BLUES)
    n = len(rows)
    fig, axes = plt.subplots(n, 5, figsize=(19, 3.4 * n + 0.8), squeeze=False,
                             gridspec_kw=dict(width_ratios=[1, 1, 1, 1, 1.15]))
    for ax_row, r in zip(axes, rows):
        q, t = r["query"], r["target"]
        xq, lq = sections[q]
        xt, lt = targets.get(t) or sections[t]
        k = lambda f: store.key(direction, q, t, f)
        pred = np.asarray(metrics.ORGANS)[cand[k("pred")]]
        conf = cand[k("conf")].astype(np.float32)
        C = cand[k("C_full")].astype(np.float64)
        s = float(np.clip(9000 / len(xq), 1.5, 14))
        for ax, (xy, c, title) in zip(ax_row[:3], [(xq, _colors(lq, pal), f"{q} (query)"),
                                                   (xt, _colors(lt, pal), f"{t} (target)"),
                                                   (xq, _colors(pred, pal), "transferred")]):
            ax.scatter(*xy.T, c=c, s=s, linewidths=0)
            ax.set_title(title, fontsize=10, color="#52514e")
        ax_row[3].scatter(*xq.T, c=conf, cmap=blues, vmin=0, vmax=1, s=s, linewidths=0)
        ax_row[3].set_title(f"confidence (mean {conf.mean():.2f})", fontsize=10, color="#52514e")
        for ax in ax_row[:4]:
            ax.set_aspect("equal"); ax.axis("off")
        # scored organs with mass: rows = query organs, columns = target organs
        rr = np.flatnonzero(metrics.SCORED & (C.sum(1) > 1e-9))
        cc = np.flatnonzero(metrics.SCORED & (C.sum(0) > 1e-9))
        Cs = C[np.ix_(rr, cc)]
        Cs = Cs / np.maximum(Cs.sum(1, keepdims=True), 1e-12)
        ax = ax_row[4]
        ax.imshow(Cs, cmap=blues, vmin=0, vmax=1, aspect="auto")
        ax.set_yticks(range(len(rr)), [metrics.ORGANS[i] for i in rr], fontsize=7)
        ax.set_xticks(range(len(cc)), [metrics.ORGANS[i] for i in cc], fontsize=7, rotation=60, ha="right")
        ax.set_title(f"κ_bal {r['kappa_bal']:.3f} · OMS {r['oms']:.3f} · K {int(r['K'])}", fontsize=10,
                     color="#0b0b0b")
        for i, o in enumerate(rr):
            if o in set(cc):
                j = list(cc).index(o)
                ax.add_patch(matplotlib.patches.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, lw=1.2,
                                                          ec="#0b0b0b"))
    handles = [Line2D([0], [0], marker="o", ls="", color=c, label=o) for o, c in pal.items()]
    handles.append(Line2D([0], [0], marker="o", ls="", color=OTHER, label="Other organs"))
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), frameon=False, fontsize=9)
    fig.suptitle(f"{pair.id} · {pipeline} · {pair.direction_label(direction)} · {role}", fontsize=12,
                 color="#0b0b0b")
    fig.tight_layout(rect=(0, 0.04, 1, 0.97))
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=110)
    plt.close(fig)
    return out


def sheets(cfg, pair, seed, root, outdir, pipelines=("A_fgw", "V_fgw", "PC_fgw")):
    """All contact sheets of one pair and seed (a volume target is drawn as its in-plane projection). Returns the
    files written."""
    sh = cfg.section_hash()
    sections, labels = {}, []
    for e in pair.embryos:
        for s in e.sections:
            sec = prepare.load_section(root, s, seed, sh)
            sections[s] = (sec.xy - sec.xy.mean(0), sec.label)
            labels.append(sec.label)
    pal = organ_palette(labels)
    files = []
    for pipe in pipelines:
        if pipe not in cfg.raw["pipelines"] or pair.kind not in cfg.raw["pipelines"][pipe]["kinds"]:
            continue
        d = store.result_dir(root, cfg, pipe, pair.id, seed)
        if not store.is_done(d, cfg, pipe):
            continue
        import pandas as pd
        df = pd.read_csv(d / "scores.csv")
        targets = {}
        if (d / "targets.npz").exists():
            tz = np.load(d / "targets.npz")
            for kk in tz.files:
                if kk.endswith("/xy"):
                    direction, name, _ = kk.split("/")
                    targets[(direction, name)] = (tz[kk] - tz[kk].mean(0), tz[f"{direction}/{name}/label"])
        for direction in pair.directions:
            rows = df[(df.role == "selected") & (df.direction == direction)].to_dict("records")
            if not rows:
                continue
            tg = {name: v for (dd, name), v in targets.items() if dd == direction}
            out = Path(outdir) / f"{store.safe(pair.id)}_seed{seed}_{pipe}_{direction}.png"
            files.append(contact_sheet(pair, pipe, direction, d, rows, sections, tg, pal, out))
    return files
