"""The report of one tier: reports/<name>/{index.html, table.md, 2d.csv, within.csv, 3d.csv, png/*.png}.

index.html is self-contained apart from the PNGs it links by relative paths, so the whole folder can be copied as is.
"""
import html
import time
from pathlib import Path

import numpy as np

from mgw.bench.report import figures, table

TITLES = {
    "2d": ("1. 2D ↔ 2D: which tool aligns real sections best?",
           "Per pair: κ_bal of the selected target section (lowest objective) for every query section, the mean of "
           "both directions. Last row: the median over the pairs."),
    "within": ("2a. 3D, within embryo: which tool couples neighbouring sections best?",
               "Per embryo: κ_bal of the kept coupling of every pair of neighbouring sections, both ways. Last row: "
               "the mean over the embryos."),
    "3d": ("2b. 3D, between embryos: which tool aligns the volumes best, and by how much does it beat 2D?",
           "Per pair: κ_bal on the real query spots, the mean of both directions. Δ: κ_bal minus that of the best 2D "
           "tool on the same pair. Last row: the median over the pairs."),
}


def _cells(t):
    """Column names and rows of a table with mean and _sd columns merged into "mean ± sd"."""
    cols = [c for c in t.columns if not c.endswith("_sd")]
    rows = []
    for _, r in t.iterrows():
        row = []
        for c in cols:
            v, sd = r[c], r.get(f"{c}_sd", np.nan)
            if isinstance(v, (float, np.floating)) and np.isfinite(v):
                s = f"{v:+.3f}" if c.startswith("Δ") else f"{v:.3f}"
                row.append(s + (f" ± {sd:.3f}" if isinstance(sd, (float, np.floating)) and np.isfinite(sd) else ""))
            else:
                row.append("–" if not isinstance(v, str) else v)
        rows.append(row)
    return cols, rows


def to_markdown(t, best=None):
    if t.empty:
        return "_no results_\n"
    cols, rows = _cells(t)
    head = [f"**{c}**" if c == best else c for c in cols]
    lines = ["| " + " | ".join(head) + " |", "| " + " | ".join("---" for _ in cols) + " |"]
    lines += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(lines) + "\n"


def to_html(t, best=None):
    if t.empty:
        return "<p class='muted'>no results</p>"
    cols, rows = _cells(t)
    head = "".join(f"<th class='{'best' if c == best else ''}'>{html.escape(c)}{' ★' if c == best else ''}</th>"
                   for c in cols)
    body = "".join("<tr>" + "".join(
        f"<td class='{'num' if j else ''}{' best' if cols[j] == best else ''}'>{html.escape(v)}</td>"
        for j, v in enumerate(r)) + "</tr>" for r in rows)
    return f"<div class='scroll'><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>"


PAGE = """<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>{title}</title><style>
:root {{ color-scheme: light; --bg: #f5f5f3; --card: #fcfcfb; --border: #e2e1dc; --text: #0b0b0b; --text2: #52514e;
  --muted: #8a8984; --accent: #256abf; }}
@media (prefers-color-scheme: dark) {{ :root:not([data-theme="light"]) {{ color-scheme: dark; --bg: #121211;
  --card: #1a1a19; --border: #383835; --text: #f0efec; --text2: #c3c2b7; --muted: #8f8e86; --accent: #6da7ec; }} }}
:root[data-theme="dark"] {{ color-scheme: dark; --bg: #121211; --card: #1a1a19; --border: #383835; --text: #f0efec;
  --text2: #c3c2b7; --muted: #8f8e86; --accent: #6da7ec; }}
body {{ margin: 0; background: var(--bg); color: var(--text); font: 14px/1.5 system-ui, -apple-system, sans-serif; }}
main {{ max-width: 1200px; margin: 0 auto; padding: 16px; }}
h1 {{ font-size: 22px; margin: 4px 0; }} h2 {{ font-size: 16px; margin: 28px 0 6px; }} h3 {{ font-size: 14px; }}
p {{ color: var(--text2); }} a {{ color: var(--accent); }}
.card {{ background: var(--card); border: 1px solid var(--border); border-radius: 10px; padding: 12px; }}
.scroll {{ overflow-x: auto; }} table {{ border-collapse: collapse; font-variant-numeric: tabular-nums; }}
th, td {{ padding: 4px 12px; border-bottom: 1px solid var(--border); text-align: left; white-space: nowrap; }}
th {{ color: var(--text2); font-weight: 500; }} td.num {{ text-align: right; }} .best {{ color: var(--accent);
  font-weight: 600; }} .muted {{ color: var(--muted); }}
img {{ max-width: 100%; height: auto; background: #fff; border-radius: 6px; }}
details {{ margin: 8px 0; }} summary {{ cursor: pointer; color: var(--text2); }}
</style></head><body><main>{body}</main></body></html>"""


def report(cfg, ser, root, tier="dev", name=None, seeds=None, figures_on=True, log=print):
    root = Path(root)
    name = name or tier
    out = root / "reports" / name
    out.mkdir(parents=True, exist_ok=True)
    df, tabs = table.tables(cfg, ser, root, tier, seeds)
    for key, (t, _, _) in tabs.items():
        t.to_csv(out / f"{key}.csv", index=False)
    b2, bw, b3 = tabs["2d"][2], tabs["within"][2], tabs["3d"][2]
    coupling = cfg.raw["stack"]["coupling"]
    notes = {"2d": f"Best: <b>{b2 or '–'}</b>.",
             "within": f"Best: <b>{bw or '–'}</b>. The 3D pipelines below use <code>stack.coupling = {coupling}</code>"
                       + ("" if not bw or cfg.raw["pipelines"][bw]["params"]["coupling"] == coupling else " (not the best: set it to the winner and rerun "
                                                                    "the 3D pipelines)") + ".",
             "3d": f"Best: <b>{b3 or '–'}</b>, against the best 2D tool <b>{b2 or '–'}</b>."}
    # figures: annotated | best 2D | best 3D, per pair and direction, for the first seed run
    figs = []
    pipes = [p for p in (b2, b3) if p]
    if figures_on and len(df) and pipes:
        s0 = int(sorted(df.seed.unique())[0])
        for pid in [p for p in ser.tier(tier) if p in set(df.pair)]:
            pair = ser.pairs[pid]
            for direction in pair.directions:
                try:
                    f = figures.transfer_figure(cfg, pair, direction, s0, root, pipes,
                                                out / "png" / f"transfer_{pid}_{direction}_seed{s0}.png")
                    if f:
                        figs.append((pair, direction, f))
                except Exception as e:   # noqa: BLE001
                    log(f"  figure {pid} {direction} failed: {type(e).__name__}: {e}")
    stamp = time.strftime("%Y-%m-%d %H:%M")
    seeds_run = sorted(int(s) for s in df.seed.unique()) if len(df) else []
    md = [f"# Alignment benchmark · tier {tier}\n",
          f"Generated {stamp}. Metric: organ-balanced Cohen's κ (κ_bal; 0 = chance, 1 = every organ to the same "
          f"organ). Seeds {seeds_run} (mean ± SD over seeds where there are several).\n"]
    body = [f"<h1>Alignment benchmark · tier {html.escape(tier)}</h1>",
            f"<p>Generated {stamp} from {html.escape(str(root))}. One metric: organ-balanced Cohen's κ (κ_bal), "
            "averaged over organs: 0 is chance, 1 sends every organ's tissue to the same organ in the other embryo. "
            f"Seeds {seeds_run}; mean ± SD over seeds where there are several. ★ marks the best tool.</p>"]
    for key, best in (("2d", b2), ("within", bw), ("3d", b3)):
        title, expl = TITLES[key]
        t = tabs[key][0]
        md += [f"\n## {title}\n", expl + "\n", to_markdown(t, best), notes[key].replace("<b>", "**").replace(
            "</b>", "**").replace("<code>", "`").replace("</code>", "`") + "\n"]
        body += [f"<h2>{html.escape(title)}</h2><p>{html.escape(expl)}</p><div class='card'>", to_html(t, best),
                 f"<p>{notes[key]}</p></div>"]
    body.append("<h2>Label transfer</h2><p>Every query section: the annotated organs, then the organs transferred "
                f"along the best 2D tool ({html.escape(b2 or '–')}) and the best 3D tool ({html.escape(b3 or '–')}). "
                "Each spot takes the organ that receives most of its mass.</p>")
    md.append("\n## Label transfer\n")
    for i, (pair, direction, f) in enumerate(figs):
        rel = f.relative_to(out).as_posix()
        q, t = pair.sides(direction)
        label = f"{pair.id}: {q.name} ({q.species} queries) from {t.name}"
        body.append(f"<details{' open' if i == 0 else ''}><summary>{html.escape(label)}</summary>"
                    f"<img src='{rel}' alt='{html.escape(label)}' loading='lazy'></details>")
        md.append(f"- [{label}]({rel})")
    if not figs:
        body.append("<p class='muted'>no figures</p>")
    (out / "table.md").write_text("\n".join(md) + "\n")
    (out / "index.html").write_text(PAGE.format(title=f"Alignment benchmark {tier}", body="\n".join(body)))
    log(f"report -> {out / 'index.html'}")
    return out / "index.html"
