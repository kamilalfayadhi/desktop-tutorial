"""Self-contained HTML report (figures embedded) summarising the whole run."""
from __future__ import annotations

import base64
import html
from datetime import datetime
from pathlib import Path

import pandas as pd

from .environment import describe

CSS = """
:root{--bg:#fcfcfb;--ink:#0b0b0b;--ink2:#52514e;--line:#e4e3df;--accent:#1f6b35}
@media (prefers-color-scheme: dark){:root{--bg:#1a1a19;--ink:#fff;--ink2:#c3c2b7;--line:#3a3936;--accent:#6fbf73}
 img{background:#fcfcfb;border-radius:6px}}
body{background:var(--bg);color:var(--ink);font:15px/1.55 system-ui,-apple-system,Segoe UI,Roboto,sans-serif;
 max-width:980px;margin:0 auto;padding:24px 16px 64px}
h1{font-size:26px;margin:0 0 4px}h2{font-size:19px;margin:36px 0 10px;border-bottom:1px solid var(--line);padding-bottom:6px}
.sub{color:var(--ink2);margin:0 0 20px}
table{border-collapse:collapse;width:100%;margin:8px 0 16px;font-size:14px;display:block;overflow-x:auto}
th,td{text-align:left;padding:6px 10px;border-bottom:1px solid var(--line);white-space:nowrap}
th{color:var(--ink2);font-weight:600}
td.num{text-align:right;font-variant-numeric:tabular-nums}
img{max-width:100%;height:auto;display:block;margin:8px 0 18px}
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:12px;margin:12px 0}
.tile{border:1px solid var(--line);border-radius:10px;padding:12px 14px}
.tile b{display:block;font-size:24px;color:var(--accent)}.tile span{color:var(--ink2);font-size:13px}
.note{color:var(--ink2);font-size:13.5px}
"""

METRIC_LABELS = {
    "training_auc": "Training AUC", "max_tss": "Max TSS (true skill statistic)",
    "continuous_boyce_index": "Continuous Boyce index", "cv_test_auc": "Spatial-CV test AUC (mean)",
    "cv_test_auc_sd": "Spatial-CV test AUC (SD)", "cv_auc_diff": "AUC difference train − test (overfitting)",
    "cv_omission_rate_10pct": "Omission rate at 10th percentile (CV)",
}

SELECTION_RULES = {
    "test_auc": "The model with the highest mean test AUC was selected.",
    "auc_diff": "The model with the least overfitting (smallest train − test AUC difference) was selected.",
    "auc_then_or10": ("Among models within {tol} of the best mean test AUC, the one with the lowest 10th-percentile "
                      "omission rate (then the strongest regularization) was selected."),
}


def _img(path: Path) -> str:
    data = base64.b64encode(path.read_bytes()).decode()
    return f'<img alt="{html.escape(path.stem)}" src="data:image/png;base64,{data}">'


def _table(df: pd.DataFrame, floatfmt="{:.3f}") -> str:
    head = "".join(f"<th>{html.escape(str(c))}</th>" for c in df.columns)
    rows = []
    for _, r in df.iterrows():
        cells = []
        for v in r:
            if isinstance(v, float):
                cells.append(f'<td class="num">{floatfmt.format(v)}</td>')
            elif isinstance(v, int):
                cells.append(f'<td class="num">{v:,}</td>')
            else:
                cells.append(f"<td>{html.escape(str(v))}</td>")
        rows.append("<tr>" + "".join(cells) + "</tr>")
    return f"<table><thead><tr>{head}</tr></thead><tbody>{''.join(rows)}</tbody></table>"


def build(out: Path, ctx: dict) -> Path:
    fig = out / "figures"
    cfg = ctx["cfg"]
    best = ctx["best"]
    m = ctx["metrics"]
    tiles = [
        (f"{ctx['n_presences']}", "presences used (after thinning)"),
        (f"{best['test_auc']:.3f}", "mean spatial-CV test AUC"),
        (f"{m['training_auc']:.3f}", "training AUC (final model)"),
        (f"{m['max_tss']:.3f}", "max TSS"),
        (f"{m['continuous_boyce_index']:.2f}", "continuous Boyce index"),
        (f"{ctx['suitable_area_km2']:,.0f} km²", "suitable area (max-TSS threshold)"),
    ]
    tiles_html = "".join(f'<div class="tile"><b>{v}</b><span>{html.escape(t)}</span></div>' for v, t in tiles)

    cleaning = pd.DataFrame(list(ctx["cleaning_steps"].items()), columns=["Step", "Records remaining"])
    variables = pd.DataFrame({"Variable": ctx["variables"], "Description": [describe(v) for v in ctx["variables"]]})
    removed = pd.DataFrame(ctx["removed"]) if ctx["removed"] else pd.DataFrame(columns=["variable", "reason"])
    tuning = ctx["tuning"].sort_values("test_auc", ascending=False).head(10).copy()
    tuning["rm"] = tuning["rm"].astype(float)
    imp = ctx["importance"].reset_index().rename(columns={"index": "Variable"})
    imp["Description"] = imp["Variable"].map(describe)
    imp = imp[["Variable", "Description", "importance_pct", "auc_drop"]]
    metrics_table = pd.DataFrame([(METRIC_LABELS.get(k, k), float(v)) for k, v in m.items()], columns=["Metric", "Value"])
    thresholds = pd.DataFrame(list(ctx["thresholds"].items()), columns=["Threshold rule", "Cloglog value"])

    future_html = ""
    if ctx.get("future"):
        fut = ctx["future"]
        table = pd.DataFrame({
            "Scenario": [f["label"] for f in fut],
            "Suitable area (km²)": [round(f["suitable_area_km2"]) for f in fut],
            "Change vs current (%)": [f["area_change_pct"] for f in fut],
            "Novel climate, MESS < 0 (% of cells)": [f["novel_climate_pct"] for f in fut],
        })
        sections = "".join(
            f"<h3>{html.escape(f['label'])}</h3>"
            + "".join(_img(fig / f"{kind}_{f['tag']}.png") for kind in ("suitability", "change", "mess"))
            for f in fut)
        future_html = f"""
<h2>7. Future projections</h2>
<p>Current suitable area: <b>{ctx['suitable_area_km2']:,.0f} km²</b> (max-TSS threshold, applied unchanged to every scenario).
Areas with negative MESS have climates outside the range of current training conditions; predictions there are extrapolations.</p>
{_table(table, floatfmt="{:+.1f}")}
{sections}"""

    src = cfg["environment"]
    env_desc = (f"WorldClim v2.1 bioclimatic variables (1970–2000), {src['resolution']} resolution"
                if src["source"] == "worldclim" else f"Local rasters from <code>{html.escape(str(src['local_dir']))}</code>")

    doc = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>MaxEnt – {html.escape(cfg['species']['name'])}</title><style>{CSS}</style></head><body>
<h1>Habitat suitability model – <i>{html.escape(cfg['species']['name'])}</i></h1>
<p class="sub">MaxEnt species distribution model · generated {datetime.now():%Y-%m-%d %H:%M} ·
study area {cfg['study_area']['bbox']} · occurrences: {html.escape(ctx['occ_source'])}</p>
<div class="tiles">{tiles_html}</div>

<h2>1. Occurrence data</h2>
{_table(cleaning)}
<p class="note">Records were cleaned (invalid/zero coordinates, outside study area, old or imprecise records,
duplicates) and spatially thinned to {cfg['occurrences']['thin_km']} km to reduce sampling bias.
{ctx['n_background']:,} background points were sampled
{"within " + str(cfg['background']['buffer_km']) + " km of presences" if cfg['background']['buffer_km'] else "across the study area"}.</p>
{_img(fig / 'occurrences.png')}

<h2>2. Environmental variables</h2>
<p>{env_desc}. Collinear variables were removed (VIF &lt; {src['vif_threshold']}, then |r| ≤ {src['correlation_threshold']}).</p>
<p><b>Retained:</b></p>{_table(variables)}
<p><b>Removed:</b></p>{_table(removed)}
{_img(fig / 'correlation.png')}

<h2>3. Model tuning</h2>
<p>{len(ctx['tuning'])} candidate models (feature classes × regularization multipliers) were evaluated with
{"4-fold spatial block" if cfg['model']['cv_method'] == 'block' else "5-fold random"} cross-validation.
{SELECTION_RULES.get(cfg['model'].get('selection_metric', 'test_auc'), '').format(tol=cfg['model'].get('auc_tolerance', 0.005))}
Selected: <b>features = {html.escape(best['features'])}, RM = {best['rm']}</b>
(test AUC {best['test_auc']:.3f} ± {best['test_auc_sd']:.3f}, AUC diff {best['auc_diff']:.3f}, OR10 {best['or10']:.3f}).</p>
{_table(tuning)}
{_img(fig / 'tuning.png')}

<h2>4. Final model evaluation</h2>
{_table(metrics_table)}
<p class="note">AUC &gt; 0.7 is usually considered useful, &gt; 0.8 good, &gt; 0.9 excellent. Spatial-CV test AUC is the most
honest measure of transferability. Boyce index ranges −1 to 1; positive values mean predictions match presences better than random.</p>
{_img(fig / 'roc.png')}
<p><b>Thresholds</b> (used for binary maps):</p>{_table(thresholds)}

<h2>5. Variable contributions</h2>
{_table(imp)}
{_img(fig / 'importance.png')}
{_img(fig / 'response_curves.png')}

<h2>6. Habitat suitability – current climate</h2>
{_img(fig / 'suitability_current.png')}
{_img(fig / 'binary_current.png')}
{future_html}

<h2>Files</h2>
<p class="note">GeoTIFFs for GIS are in <code>rasters/</code>; tables (cleaned occurrences, background, tuning results,
importance, response curves, metrics) in <code>tables/</code>; the fitted model in <code>model/</code>.</p>
<h2>Citation / caveats</h2>
<p class="note">Fick &amp; Hijmans (2017) WorldClim 2. Phillips et al. (2006, 2017) MaxEnt; model fitted with the
<code>elapid</code> Python implementation (Anderson 2023). If occurrences came from GBIF, cite the GBIF records used
(create a download DOI at gbif.org for publication). Presence-background models estimate relative suitability,
not probability of occurrence; sampling bias in the occurrence data can bias the result.</p>
</body></html>"""
    path = out / "report.html"
    path.write_text(doc, encoding="utf-8")
    return path
