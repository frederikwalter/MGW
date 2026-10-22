"""The label-transfer figure: one PNG per (pair, direction).

One row per query section (file order), at the section's own spot positions. Columns: the annotated organs, then the
organs transferred along each given pipeline's alignment (the arg-max of the selected coupling's row, stored in
transfer.npz), each titled with the section's kappa_bal (and, for 2D, the chosen target section).
Organs: the eight most frequent of the pair in fixed categorical slots, the rest folded into a grey "Other".
"""
from pathlib import Path

import numpy as np
import pandas as pd

from mgw.bench import metrics, prepare, store

# validated categorical slots (light surface), in fixed order; "Other" is grey
SLOTS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
OTHER = "#c9c8c2"


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


def transfer_figure(cfg, pair, direction, seed, root, pipes, out):
    """Annotated | transferred by each of `pipes` for every query section of a direction. Returns the file, or None
    if none of the pipelines has results."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    res = {}
    for p in pipes:
        d = store.result_dir(root, cfg, p, pair.id, seed)
        if store.is_done(d, cfg, p) and (d / "transfer.npz").exists():
            sc = pd.read_csv(d / "scores.csv")
            sc = sc[(sc.role == "selected") & (sc.direction == direction)].set_index("query")
            res[p] = (np.load(d / "transfer.npz"), sc)
    if not res:
        return None
    sh = cfg.section_hash()
    q_emb, t_emb = pair.sides(direction)
    secs = {s: prepare.load_section(root, s, seed, sh) for s in q_emb.sections}
    pal = organ_palette([s.label for s in secs.values()] +
                        [prepare.load_section(root, s, seed, sh).label for s in t_emb.sections])
    names = np.asarray(metrics.ORGANS)
    n, m = len(secs), 1 + len(res)
    fig, axes = plt.subplots(n, m, figsize=(3.4 * m, 3.4 * n + 0.9), squeeze=False)
    for ax_row, (name, sec) in zip(axes, secs.items()):
        xy = sec.xy - sec.xy.mean(0)
        s = float(np.clip(9000 / len(xy), 1.5, 14))
        panels = [(sec.label, f"{name}\nannotated")]
        for p, (tz, sc) in res.items():
            k = store.key(direction, name)
            if k not in tz.files or name not in sc.index:
                panels.append((None, f"{p}\nno result"))
                continue
            r = sc.loc[name]
            via = f" → {r['target']}" if r["target"] not in ("volume", "points") else ""
            panels.append((names[tz[k]], f"{p}{via}\nκ_bal {r['kappa_bal']:.2f}"))
        for ax, (labels, title) in zip(ax_row, panels):
            if labels is not None:
                ax.scatter(*xy.T, c=_colors(labels, pal), s=s, linewidths=0)
            ax.set_title(title, fontsize=9, color="#0b0b0b")
            ax.set_aspect("equal"); ax.axis("off")
    handles = [Line2D([], [], marker="o", ls="", color=c, label=o) for o, c in pal.items()]
    handles.append(Line2D([], [], marker="o", ls="", color=OTHER, label="Other"))
    fig.legend(handles=handles, loc="upper center", ncol=min(len(handles), 5), frameon=False, fontsize=9,
               bbox_to_anchor=(0.5, 1.0))
    fig.suptitle(f"{pair.id} seed {seed}: {q_emb.name} ({q_emb.species}) labelled from {t_emb.name} "
                 f"({t_emb.species})", y=1 - 0.55 / fig.get_figheight(), fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.9 / fig.get_figheight()))
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=110)
    plt.close(fig)
    return out
