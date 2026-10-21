"""The report of one tier: reports/<name>/{table.csv, table.md, ceilings.csv, ranking.csv, selection.csv,
noise.json, index.html, viewers/*.html, png/*.png}.

index.html is self-contained (tables inline) and links the viewers and sheets by relative paths, so the whole folder
can be copied or published as is.
"""
import html
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from mgw.bench import config as bconfig
from mgw.bench.report import figures, table, viewer

FMT3 = ["K", "A", "oracle", "regret", "rand_section", "L_partner", "rigid", "feat_OT",
        "space_GW", "SCOT", "prior", "ceiling", "ceiling_share", "W_human", "W_mouse", "OMS", "kappa", "A_sd",
        "delta_ref", "V", "V_feat_OT", "V_GW", "PC", "dV", "dPC"]


def resolve_reference(cfg, reference=None):
    """The reference entry: --reference <pipeline> | history:<i>, else reference.yaml's current."""
    path = cfg.reference_path()
    ref = yaml.safe_load(path.read_text()) if path.exists() else {"current": {"pipeline": "A_fgw"}, "history": []}
    if reference and reference.startswith("history:"):
        entry = ref["history"][int(reference.split(":", 1)[1])]
    elif reference:
        entry = dict(pipeline=reference)
    else:
        entry = ref["current"]
    entry = dict(entry)
    if not entry.get("hash"):
        # overrides of other pipelines (e.g. --set pipelines.V_fgw.params...) do not exist in the reference's config
        ov = [o for o in (entry.get("overrides") or list(cfg.overrides))
              if not o.startswith("pipelines.") or o.startswith(f"pipelines.{entry['pipeline']}.")]
        rc = bconfig.load(entry.get("config") or cfg.path, ov)
        entry["hash"] = rc.pipeline_hash(entry["pipeline"])
    return entry


def to_markdown(df, floats=3):
    if df.empty:
        return "_no results_\n"
    cols = list(df.columns)
    fmt = lambda v: ("" if v is None or (isinstance(v, float) and np.isnan(v)) else
                     f"{v:.{floats}f}" if isinstance(v, (float, np.floating)) else
                     ("✓" if v is True else "✗" if v is False else str(v)))
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join("---" for _ in cols) + " |"]
    lines += ["| " + " | ".join(fmt(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join(lines) + "\n"


def to_html(df, floats=3, links=None):
    if df.empty:
        return "<p class='muted'>no results</p>"
    def cell(c, v):
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return "<td class='na'>–</td>"
        if isinstance(v, (bool, np.bool_)):
            return f"<td class='{'ok' if v else 'bad'}'>{'✓ pass' if v else '✗ fail'}</td>"
        if isinstance(v, (float, np.floating)):
            return f"<td class='num'>{v:.{floats}f}</td>"
        s = html.escape(str(v))
        if links and c == "pair" and str(v) in links:
            s = f"<a href='{links[str(v)]}'>{s}</a>"
        return f"<td>{s}</td>"
    head = "".join(f"<th>{html.escape(str(c))}</th>" for c in df.columns)
    body = "".join("<tr>" + "".join(cell(c, v) for c, v in zip(df.columns, row)) + "</tr>"
                   for row in df.itertuples(index=False))
    return f"<div class='scroll'><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>"


PAGE = """<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>{title}</title><style>
:root {{ color-scheme: light; --bg: #f5f5f3; --card: #fcfcfb; --border: #e2e1dc; --text: #0b0b0b; --text2: #52514e;
  --muted: #8a8984; --good: #0ca30c; --critical: #d03b3b; --link: #256abf; }}
@media (prefers-color-scheme: dark) {{ :root:not([data-theme="light"]) {{ color-scheme: dark; --bg: #121211;
  --card: #1a1a19; --border: #383835; --text: #f0efec; --text2: #c3c2b7; --muted: #8f8e86; --link: #6da7ec; }} }}
:root[data-theme="dark"] {{ color-scheme: dark; --bg: #121211; --card: #1a1a19; --border: #383835; --text: #f0efec;
  --text2: #c3c2b7; --muted: #8f8e86; --link: #6da7ec; }}
body {{ margin: 0; background: var(--bg); color: var(--text); font: 14px/1.5 system-ui, -apple-system, sans-serif; }}
main {{ max-width: 1400px; margin: 0 auto; padding: 16px; }}
h1 {{ font-size: 22px; margin: 4px 0; }} h2 {{ font-size: 16px; margin: 24px 0 8px; }}
p, li {{ color: var(--text2); }} a {{ color: var(--link); }}
.card {{ background: var(--card); border: 1px solid var(--border); border-radius: 10px; padding: 12px; }}
.scroll {{ overflow-x: auto; }} table {{ border-collapse: collapse; font-variant-numeric: tabular-nums; }}
th, td {{ padding: 4px 10px; border-bottom: 1px solid var(--border); text-align: left; white-space: nowrap; }}
th {{ color: var(--text2); font-weight: 500; }} td.num {{ text-align: right; }} td.na {{ color: var(--muted); }}
td.ok {{ color: var(--good); }} td.bad {{ color: var(--critical); font-weight: 600; }}
.muted {{ color: var(--muted); }} ul.files {{ columns: 3; }} @media (max-width: 760px) {{ ul.files {{ columns: 1; }} }}
</style></head><body><main>{body}</main></body></html>"""


def report(cfg, ser, root, tier="screen", name=None, reference=None, viewers=True, pngs=True, seeds=None,
           sweep=None, sweep_values=None, log=print):
    root = Path(root)
    name = name or f"{cfg.raw['name']}_{tier}"
    out = root / "reports" / name
    out.mkdir(parents=True, exist_ok=True)
    ref = resolve_reference(cfg, reference)
    noise = table.read_noise(cfg, root) or table.noise_margin(cfg, ser, root, write=False)
    margin = noise.get("margin") if noise else None
    go, ceil, rank, df = table.go_table(cfg, ser, root, tier, reference=ref, seeds=seeds, margin=margin)
    sel = table.selection_quality(df) if len(df) else pd.DataFrame()
    vol = table.volume_table(cfg, ser, df, tier)
    for f, d in (("table.csv", go), ("ceilings.csv", ceil), ("ranking.csv", rank), ("selection.csv", sel),
                 ("volumes.csv", vol)):
        d.to_csv(out / f, index=False)
    (out / "noise.json").write_text(json.dumps(noise, indent=1))
    sw = None
    if sweep and sweep_values:
        sw = table.sweep_table(lambda o: bconfig.load(cfg.path, list(cfg.overrides) + o), sweep, sweep_values, ser,
                               root, "dev" if tier in ("smoke", "dev", "screen") else tier)
        sw.to_csv(out / "sweep.csv", index=False)
    show = go[[c for c in go.columns if c not in ("stage", "overlap")]] if len(go) else go
    md = [f"# {cfg.raw['name']} · tier {tier}\n",
          f"Generated {time.strftime('%Y-%m-%d %H:%M')}. Reference: {ref['pipeline']} ({ref['hash']}). "
          f"Noise margin: {margin if margin is None else round(margin, 4)} ({noise.get('tier') if noise else ''}).\n",
          "## Go table (κ_bal, full resolution, mean of both directions)\n", to_markdown(show),
          "\n## Ceilings (A on same-species pairs)\n", to_markdown(ceil),
          "\n## Pipelines against the reference\n", to_markdown(rank),
          "\n## Selection: Spearman(objective, κ_bal) over each query's candidates\n", to_markdown(sel)]
    if len(vol):
        md += ["\n## Volume pipelines per direction (rejected mass, depth reached, consistency)\n", to_markdown(vol)]
    if sw is not None:
        md += [f"\n## Sweep over {sweep}\n", to_markdown(sw)]
    (out / "table.md").write_text("\n".join(md))
    links, files = {}, []
    if len(df):
        pairs_run = [p for p in ser.tier(tier) if p in set(df.pair)]
        seeds_run = sorted(df.seed.unique())
        s0 = int(seeds_run[0])
        for pid in pairs_run:
            pair = ser.pairs[pid]
            if viewers:
                try:
                    f = viewer.build(cfg, pair, s0, root, out / "viewers" / f"{pid.replace(':', '_')}_seed{s0}.html")
                    links[pid] = f.relative_to(out).as_posix()
                    files.append(f)
                    log(f"  viewer {pid}: {f.stat().st_size / 1e6:.1f} MB")
                except Exception as e:   # noqa: BLE001
                    log(f"  viewer {pid} failed: {type(e).__name__}: {e}")
            if pngs:
                try:
                    files += figures.sheets(cfg, pair, s0, root, out / "png")
                except Exception as e:   # noqa: BLE001
                    log(f"  sheets {pid} failed: {type(e).__name__}: {e}")
        if sw is not None and len(sw) and "A" in sw:
            files.append(_sweep_plot(sw, sweep, out / "png" / "sweep.png"))
    body = [f"<h1>{html.escape(cfg.raw['name'])} · tier {tier}</h1>",
            f"<p>Generated {time.strftime('%Y-%m-%d %H:%M')} from {html.escape(str(root))}. κ_bal at full resolution, "
            f"the mean of both directions (each the mean over its queries). Reference: <b>{ref['pipeline']}</b> "
            f"({ref['hash']}). Noise margin (2 × seed SD on {noise.get('tier') if noise else '–'}): "
            f"<b>{'–' if margin is None else f'{margin:.4f}'}</b>.</p>",
            "<h2>Go table</h2><div class='card'>", to_html(show, links=links), "</div>",
            "<p class='muted'>K: organs of the query present in the target embryo · "
            "A: best real section by the FGW objective · oracle: best real section by the labels · "
            "rand_section: expectation over random real sections"
            " · L_partner: closest laterality · rigid: rigid placement only · feat_OT: features "
            "only · space_GW: geometry only · SCOT: feature geometry only · prior: organ frequencies (0 by "
            "construction) · ceiling: between embryos where run, else the mean of both W · g_*: guardrails.</p>",
            "<h2>Ceilings</h2><div class='card'>", to_html(ceil, links=links), "</div>",
            "<h2>Pipelines against the reference</h2><div class='card'>", to_html(rank), "</div>",
            "<h2>Selection quality</h2><div class='card'>", to_html(sel, links=links), "</div>"]
    if len(vol):
        body += ["<h2>Volume pipelines</h2><div class='card'>", to_html(vol, links=links),
                 "<p class='muted'>V: field-sampled volumes, FGW · V_feat_OT / V_GW: features only / geometry only · "
                 "PC: the registered stack points with slab masses (no field) · delta_A: κ_bal minus A's · "
                 "rejected_*: mass the unbalanced solve leaves unmatched · L_reached: mass-weighted L of the target "
                 "spots reached (L_reached_rel: 0 at the target's lowest L, 1 at its highest) · normal_dev_deg, "
                 "order_rho_abs: the query's parallel sections in the target (§2d) · g_end: X6 guardrail.</p></div>"]
    if sw is not None:
        body += [f"<h2>Sweep over {html.escape(sweep)}</h2><div class='card'>", to_html(sw),
                 "<p><img src='png/sweep.png' alt='kappa_bal against the swept value' style='max-width:100%'></p></div>"]
    body += ["<h2>Files</h2><ul class='files'>"] + [
        f"<li><a href='{Path(f).relative_to(out).as_posix()}'>{Path(f).name}</a></li>" for f in files] + ["</ul>"]
    (out / "index.html").write_text(PAGE.format(title=f"{cfg.raw['name']} {tier} report", body="\n".join(body)))
    log(f"report -> {out / 'index.html'}")
    return out / "index.html"


def _sweep_plot(sw, key, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    out.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
    for n, (pid, g) in enumerate(sw.dropna(subset=["A"]).groupby("pair")):
        g = g.sort_values("value")
        c = figures.SLOTS[n % len(figures.SLOTS)]
        ax[0].plot(g.value, g.A, "-o", color=c, lw=2, ms=6, label=pid)
        ax[1].plot(g.value, g.seconds_per_solve, "-o", color=c, lw=2, ms=6, label=pid)
    ax[0].set_xlabel(key); ax[0].set_ylabel("κ_bal of A (selected)")
    ax[1].set_xlabel(key); ax[1].set_ylabel("seconds per FGW solve")
    for a in ax:
        a.spines[["top", "right"]].set_visible(False); a.grid(alpha=0.3)
    ax[0].legend(frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig(out, dpi=110); plt.close(fig)
    return out
