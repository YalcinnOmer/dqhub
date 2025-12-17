from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.graph_objects as go
from plotly.utils import PlotlyJSONEncoder


ROOT = Path(__file__).resolve().parents[2]
REPORTS_DIR = ROOT / "reports"
DATA_OUTPUT_DIR = ROOT / "data" / "output"

DQ_SUMMARY_CSV = REPORTS_DIR / "dq_summary.csv"
DQ_ISSUES_CSV = REPORTS_DIR / "dq_issues.csv"
CLEAN_CSV = DATA_OUTPUT_DIR / "clean.csv"

OUT_HTML = REPORTS_DIR / "DQ_Report.html"
OUT_XLSX = REPORTS_DIR / "DQ_Report.xlsx"

# Thresholds
THRESH_GOOD = 99.0
THRESH_WARN = 97.0  # below is critical


def _is_nan(x: Any) -> bool:
    try:
        return x is None or (isinstance(x, float) and math.isnan(x))
    except Exception:
        return False


def _safe_float(x: Any) -> float | None:
    if _is_nan(x):
        return None
    try:
        return float(x)
    except Exception:
        return None


def _safe_int(x: Any) -> int | None:
    if _is_nan(x):
        return None
    try:
        return int(float(x))
    except Exception:
        return None


def _pct_str(x: Any) -> str:
    v = _safe_float(x)
    return "N/A" if v is None else f"{v:.2f}%"


def _kpi_delta(curr: float | None, prev: float | None) -> str:
    if curr is None or prev is None:
        return "—"
    d = curr - prev
    sign = "+" if d >= 0 else ""
    return f"{sign}{d:.2f} pp"


def _level_for_score(score: float | None) -> str:
    if score is None:
        return "unknown"
    if score >= THRESH_GOOD:
        return "good"
    if score >= THRESH_WARN:
        return "warn"
    return "crit"


def _tight_pct_range(values: list[float | None], floor: float = 90.0, pad: float = 0.35) -> tuple[float, float]:
    nums = [float(v) for v in values if v is not None and not _is_nan(v)]
    if not nums:
        return (0.0, 100.0)

    lo = min(nums) - pad
    hi = max(nums) + pad

    hi = min(100.0, hi)
    lo = max(floor, lo)

    if hi - lo < 1.0:
        lo = max(floor, hi - 1.0)

    return (lo, hi)


def _read_csv_if_exists(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    try:
        return pd.read_csv(path)
    except Exception:
        return None


def _parse_run_ts(hist: pd.DataFrame) -> pd.DataFrame:
    # Optional ordering if timestamp exists
    if "run_ts_utc" in hist.columns:
        hist["run_ts_utc"] = pd.to_datetime(hist["run_ts_utc"], utc=True, errors="coerce")
        hist = hist.dropna(subset=["run_ts_utc"]).sort_values("run_ts_utc")
    return hist


def _prev_run(hist: pd.DataFrame) -> dict[str, Any] | None:
    if len(hist) < 2:
        return None
    return hist.iloc[-2].to_dict()


@dataclass(frozen=True)
class RunRow:
    run_id: str
    run_label: str
    rows_raw: int | None
    completeness_pct: float | None
    validity_pct: float | None
    uniqueness_pct: float | None
    timeliness_pct: float | None
    overall_pct: float | None
    rules_version: str | None


def _runs_with_labels(hist: pd.DataFrame) -> list[RunRow]:
    h = hist.reset_index(drop=True).copy()
    rows: list[RunRow] = []

    for i in range(len(h)):
        d = h.iloc[i].to_dict()
        rows.append(
            RunRow(
                run_id=str(d.get("run_id") or f"run_{i+1}"),
                run_label=f"Run {i+1}",
                rows_raw=_safe_int(d.get("rows_raw")),
                completeness_pct=_safe_float(d.get("completeness_pct")),
                validity_pct=_safe_float(d.get("validity_pct")),
                uniqueness_pct=_safe_float(d.get("uniqueness_pct")),
                timeliness_pct=_safe_float(d.get("timeliness_pct")),
                overall_pct=_safe_float(d.get("overall_pct")),
                rules_version=str(d.get("rules_version") or "").strip() or None,
            )
        )
    return rows


def _find_primary_key(df: pd.DataFrame) -> str | None:
    candidates = ["order_id", "id", "transaction_id", "invoice_id"]
    for c in candidates:
        if c in df.columns:
            return c
    for c in df.columns:
        if isinstance(c, str) and c.lower().endswith("_id"):
            return c
    return None


def _compute_uniqueness_from_df(df: pd.DataFrame) -> tuple[float | None, dict[str, Any]]:
    key = _find_primary_key(df)
    if key is None or df.empty:
        return None, {"key": None, "duplicates": None, "examples": [], "total": len(df)}

    dup_mask = df.duplicated(subset=[key], keep="first")
    dup_count = int(dup_mask.sum())
    total = int(len(df))
    if total <= 0:
        return None, {"key": key, "duplicates": dup_count, "examples": [], "total": total}

    score = 100.0 * (1.0 - (dup_count / total))
    score = max(0.0, min(100.0, score))

    examples: list[str] = []
    if dup_count > 0:
        examples = (
            df.loc[df.duplicated(subset=[key], keep=False), key]
            .astype(str)
            .head(10)
            .tolist()
        )

    return float(score), {"key": key, "duplicates": dup_count, "examples": examples, "total": total}


def _diagnostics_completeness(df: pd.DataFrame) -> dict[str, Any]:
    if df.empty:
        return {"available": False, "summary": "Dataset is empty.", "top_missing": []}

    miss = df.isna().sum().sort_values(ascending=False)
    total = len(df)

    top = []
    for col, cnt in miss.head(10).items():
        if int(cnt) == 0:
            continue
        pct = 100.0 * (int(cnt) / (total if total else 1))
        top.append({"column": str(col), "missing_count": int(cnt), "missing_pct": round(pct, 4)})

    total_missing_cells = int(miss.sum())
    total_cells = int(df.shape[0] * df.shape[1]) if df.shape[1] else 0
    missing_pct = 100.0 * (total_missing_cells / total_cells) if total_cells else 0.0

    return {
        "available": True,
        "summary": f"Missing cells: {total_missing_cells} ({missing_pct:.2f}% of all cells).",
        "top_missing": top,
    }


_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


def _diagnostics_validity(df: pd.DataFrame) -> dict[str, Any]:
    if df.empty:
        return {"available": False, "summary": "Dataset is empty.", "checks": []}

    checks: list[dict[str, Any]] = []
    total = len(df)

    email_cols = [c for c in df.columns if isinstance(c, str) and "email" in c.lower()]
    for c in email_cols[:2]:
        s = df[c].astype(str).fillna("")
        invalid = int((~s.str.match(_EMAIL_RE)).sum())
        pct = 100.0 * (invalid / (total if total else 1))
        checks.append({"check": f"{c}: invalid email format", "invalid_count": invalid, "invalid_pct": round(pct, 4)})

    country_cols = [c for c in df.columns if isinstance(c, str) and "country" in c.lower()]
    for c in country_cols[:2]:
        s = df[c].astype(str).fillna("")
        invalid = int((~s.str.match(r"^[A-Za-z]{2}$")).sum())
        pct = 100.0 * (invalid / (total if total else 1))
        checks.append({"check": f"{c}: invalid country code", "invalid_count": invalid, "invalid_pct": round(pct, 4)})

    currency_cols = [c for c in df.columns if isinstance(c, str) and "currency" in c.lower()]
    for c in currency_cols[:1]:
        s = df[c].astype(str).fillna("")
        invalid = int((~s.str.match(r"^[A-Za-z]{3}$")).sum())
        pct = 100.0 * (invalid / (total if total else 1))
        checks.append({"check": f"{c}: invalid currency code", "invalid_count": invalid, "invalid_pct": round(pct, 4)})

    amt_cols = [c for c in df.columns if isinstance(c, str) and any(k in c.lower() for k in ["amount", "total", "price"])]
    for c in amt_cols[:2]:
        s = pd.to_numeric(df[c], errors="coerce")
        invalid = int((s.isna() | (s < 0)).sum())
        pct = 100.0 * (invalid / (total if total else 1))
        checks.append({"check": f"{c}: invalid numeric amount", "invalid_count": invalid, "invalid_pct": round(pct, 4)})

    if not checks:
        return {"available": True, "summary": "No built-in validity diagnostics matched dataset columns.", "checks": []}

    worst = max(checks, key=lambda x: x["invalid_pct"])
    return {
        "available": True,
        "summary": f"Top invalidity: {worst['check']} ({worst['invalid_pct']:.2f}%).",
        "checks": sorted(checks, key=lambda x: x["invalid_pct"], reverse=True)[:10],
    }


def _diagnostics_timeliness(df: pd.DataFrame) -> dict[str, Any]:
    if df.empty:
        return {"available": False, "summary": "Dataset is empty.", "stats": {}}

    order_cols = [c for c in df.columns if isinstance(c, str) and "order_ts" in c.lower()]
    ingest_cols = [c for c in df.columns if isinstance(c, str) and any(k in c.lower() for k in ["ingested_ts", "ingest_ts"])]

    if not order_cols or not ingest_cols:
        return {"available": False, "summary": "No timestamp pair found (order_ts / ingested_ts).", "stats": {}}

    oc = order_cols[0]
    ic = ingest_cols[0]

    o = pd.to_datetime(df[oc], utc=True, errors="coerce")
    i = pd.to_datetime(df[ic], utc=True, errors="coerce")
    mask = o.notna() & i.notna()
    if mask.sum() == 0:
        return {"available": False, "summary": "Timestamp columns exist but could not be parsed.", "stats": {}}

    delay_hours = (i[mask] - o[mask]).dt.total_seconds() / 3600.0
    p50 = float(delay_hours.quantile(0.50))
    p95 = float(delay_hours.quantile(0.95))
    mx = float(delay_hours.max())
    late = int((delay_hours > 24.0).sum())
    pct_late = 100.0 * (late / (len(delay_hours) if len(delay_hours) else 1))

    return {
        "available": True,
        "summary": f"Latency p95: {p95:.2f}h, late>24h: {late} ({pct_late:.2f}%).",
        "stats": {
            "p50_h": round(p50, 4),
            "p95_h": round(p95, 4),
            "max_h": round(mx, 4),
            "late_24h_count": late,
            "late_24h_pct": round(pct_late, 4),
        },
        "columns": {"order_ts": oc, "ingested_ts": ic},
    }


def _build_details_for_clean(clean_path: Path) -> dict[str, Any]:
    df = _read_csv_if_exists(clean_path)
    if df is None:
        return {"available": False, "summary": f"Missing dataset: {clean_path.name}", "dimensions": {}}

    uniq_score, uniq_meta = _compute_uniqueness_from_df(df)

    return {
        "available": True,
        "summary": f"Dataset loaded: {clean_path.name} ({len(df)} rows, {df.shape[1]} columns).",
        "dimensions": {
            "Completeness": _diagnostics_completeness(df),
            "Validity": _diagnostics_validity(df),
            "Uniqueness": {
                "available": True,
                "summary": (
                    "No key column found for uniqueness diagnostics."
                    if uniq_score is None
                    else f"Key: {uniq_meta.get('key')}, duplicates: {uniq_meta.get('duplicates')} of {uniq_meta.get('total')}."
                ),
                "meta": uniq_meta,
                "score": uniq_score,
            },
            "Timeliness": _diagnostics_timeliness(df),
        },
    }


def _issues_state(issues: pd.DataFrame | None, run_id: str) -> tuple[str, int | None, pd.DataFrame | None]:
    if issues is None:
        return ("missing", None, None)

    df = issues.copy()
    if "run_id" in df.columns:
        df = df[df["run_id"].astype(str) == str(run_id)]

    if df.empty:
        return ("none", 0, df)

    return ("some", int(len(df)), df)


def _lowest_dimension(cur: RunRow) -> tuple[str, str]:
    dims = [
        ("Completeness", cur.completeness_pct),
        ("Validity", cur.validity_pct),
        ("Uniqueness", cur.uniqueness_pct),
        ("Timeliness", cur.timeliness_pct),
    ]
    available = [(n, v) for (n, v) in dims if v is not None]
    missing = [n for (n, v) in dims if v is None]

    if not available:
        return ("N/A", "No dimension scores available.")

    vals = [v for _, v in available]
    lo = min(vals)
    hi = max(vals)

    if abs(hi - lo) < 0.001:
        sub = ("Missing: " + ", ".join(missing)) if missing else "All metrics available."
        return (f"All dimensions tied ({lo:.2f}%)", sub)

    worst = [n for (n, v) in available if abs(v - lo) < 0.001]
    sub = ("Missing: " + ", ".join(missing)) if missing else "All metrics available."
    return (f"{', '.join(worst)} ({lo:.2f}%)", sub)


def _fig_score_bar_for_run(run: RunRow) -> go.Figure:
    labels = ["Completeness", "Validity", "Uniqueness", "Timeliness", "Overall"]
    values = [run.completeness_pct, run.validity_pct, run.uniqueness_pct, run.timeliness_pct, run.overall_pct]

    y: list[float] = []
    text: list[str] = []
    colors: list[str] = []

    for v in values:
        if v is None:
            y.append(0.0)
            text.append("")
            colors.append("#9ca3af")
        else:
            y.append(float(v))
            text.append(f"{float(v):.2f}%")
            lvl = _level_for_score(float(v))
            colors.append("#16a34a" if lvl == "good" else "#f59e0b" if lvl == "warn" else "#dc2626")

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=labels,
            y=y,
            marker=dict(color=colors),
            text=text,
            textposition="inside",
            hovertemplate="Metric: %{x}<br>Score: %{y:.2f}%<extra></extra>",
        )
    )

    for lab, v in zip(labels, values):
        if v is None:
            fig.add_annotation(x=lab, y=3, text="N/A", showarrow=False, font=dict(size=12), yanchor="bottom")

    fig.update_layout(
        template="plotly_white",
        height=320,
        margin=dict(l=40, r=20, t=10, b=40),
        showlegend=False,
        paper_bgcolor="white",
        plot_bgcolor="white",
        font=dict(size=12),
    )
    fig.update_yaxes(title_text="%", range=[0, 100])
    return fig


def _fig_trend(runs: list[RunRow]) -> go.Figure | None:
    if len(runs) < 2:
        return None

    labels = [r.run_label for r in runs]

    def series(col: str) -> list[float | None]:
        return [getattr(r, col) for r in runs]

    all_vals: list[float | None] = []
    fig = go.Figure()

    for name, col, default_visible in [
        ("Overall", "overall_pct", True),
        ("Completeness", "completeness_pct", False),
        ("Validity", "validity_pct", False),
        ("Uniqueness", "uniqueness_pct", False),
        ("Timeliness", "timeliness_pct", False),
    ]:
        y = series(col)
        all_vals.extend(y)
        fig.add_trace(
            go.Scatter(
                x=labels,
                y=y,
                mode="lines+markers",
                name=name,
                visible=True if default_visible else "legendonly",
                hovertemplate=f"{name}: %{{y:.2f}}%<br>Run: %{{x}}<extra></extra>",
            )
        )

    ylo, yhi = _tight_pct_range(all_vals, floor=90.0, pad=0.35)

    fig.update_layout(
        template="plotly_white",
        height=320,
        margin=dict(l=40, r=20, t=40, b=45),
        paper_bgcolor="white",
        plot_bgcolor="white",
        font=dict(size=12),
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0.0, font=dict(size=11)),
    )
    fig.update_yaxes(title_text="%", range=[ylo, yhi], automargin=True)
    fig.update_xaxes(title_text="", type="category", automargin=True)
    return fig


def _write_xlsx(hist: pd.DataFrame, issues: pd.DataFrame | None, details: dict[str, Any]) -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    try:
        with pd.ExcelWriter(OUT_XLSX, engine="openpyxl") as xw:
            hist.to_excel(xw, sheet_name="dq_summary", index=False)
            if issues is not None:
                issues.to_excel(xw, sheet_name="dq_issues", index=False)

            diag_rows: list[dict[str, Any]] = []
            dims = details.get("dimensions", {})
            for dim, payload in dims.items():
                diag_rows.append({"dimension": dim, "summary": payload.get("summary", "")})
            pd.DataFrame(diag_rows).to_excel(xw, sheet_name="diagnostics", index=False)
    except Exception:
        # Best-effort
        pass


_HTML_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Data Quality Dashboard</title>
  <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
  <style>
    :root {
      --text:#111827;
      --muted:#6b7280;
      --border:#e5e7eb;
      --card:#ffffff;
      --bg:#f8fafc;
      --good:#16a34a;
      --warn:#f59e0b;
      --crit:#dc2626;
      --shadow: 0 1px 2px rgba(0,0,0,.04);
    }
    [data-theme="dark"] {
      --text:#e5e7eb;
      --muted:#9ca3af;
      --border:#1f2937;
      --card:#0b1220;
      --bg:#060b16;
      --shadow: 0 1px 2px rgba(0,0,0,.35);
    }
    body {
      margin:0;
      background:var(--bg);
      color:var(--text);
      font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Arial;
    }
    .container {
      max-width: 1280px;
      margin: 26px auto;
      padding: 0 18px 40px;
    }
    header {
      display:flex;
      align-items:flex-end;
      justify-content:space-between;
      gap:16px;
      margin-bottom: 14px;
    }
    h1 {
      margin: 0;
      font-size: 30px;
      letter-spacing: -0.02em;
    }
    .sub {
      color: var(--muted);
      font-size: 12.5px;
      margin-top: 6px;
    }
    .toolbar {
      display:flex;
      gap:10px;
      align-items:center;
      justify-content:flex-end;
      flex-wrap: wrap;
    }
    .pill {
      border:1px solid var(--border);
      background:var(--card);
      border-radius:999px;
      padding:7px 10px;
      font-size:12px;
      color:var(--muted);
      box-shadow: var(--shadow);
      display:flex;
      gap:8px;
      align-items:center;
    }
    select, button {
      border:1px solid var(--border);
      background:var(--card);
      color:var(--text);
      border-radius:10px;
      padding:8px 10px;
      font-size:12px;
      cursor:pointer;
      box-shadow: var(--shadow);
    }
    button.secondary { color: var(--muted); }
    button:hover { filter: brightness(0.98); }

    .kpi-grid {
      display:grid;
      grid-template-columns: 1fr;
      gap: 12px;
      margin: 14px 0 18px;
    }
    @media (min-width: 900px) {
      .kpi-grid { grid-template-columns: repeat(4, 1fr); }
    }
    .kpi {
      border:1px solid var(--border);
      border-radius: 14px;
      background: var(--card);
      padding: 12px 12px;
      box-shadow: var(--shadow);
      min-height: 90px;
      position: relative;
    }
    .kpi-label {
      color: var(--muted);
      font-size: 12px;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: .04em;
    }
    .kpi-value {
      font-size: 20px;
      font-weight: 800;
      margin-top: 8px;
    }
    .kpi-sub {
      color: var(--muted);
      font-size: 12px;
      margin-top: 6px;
      line-height: 1.35;
    }
    .badge {
      position:absolute;
      top:10px;
      right:10px;
      font-size:11px;
      font-weight:800;
      padding:4px 8px;
      border-radius:999px;
      border:1px solid var(--border);
    }
    .badge.good { color: var(--good); }
    .badge.warn { color: var(--warn); }
    .badge.crit { color: var(--crit); }

    .grid {
      display:grid;
      grid-template-columns: 1fr;
      gap: 14px;
      align-items: start;
    }
    @media (min-width: 1050px) {
      .grid { grid-template-columns: 1fr 1fr; }
    }

    .card {
      border:1px solid var(--border);
      border-radius: 14px;
      background: var(--card);
      padding: 12px;
      box-shadow: var(--shadow);
    }
    .card h2 {
      margin: 2px 0 10px;
      font-size: 14px;
      font-weight: 900;
      letter-spacing: -0.01em;
      display:flex;
      align-items:center;
      justify-content:space-between;
      gap: 10px;
    }
    .muted {
      color: var(--muted);
      font-size: 12.5px;
      line-height: 1.45;
    }

    .filters {
      display:flex;
      flex-wrap:wrap;
      gap:10px;
      align-items:center;
      margin-top: 6px;
    }
    .chk {
      display:flex;
      gap:6px;
      align-items:center;
      font-size:12px;
      color: var(--muted);
    }

    .details-grid {
      display:grid;
      grid-template-columns: 1fr;
      gap: 12px;
    }
    @media (min-width: 1050px) {
      .details-grid { grid-template-columns: 1.1fr 0.9fr; }
    }

    .table-wrap {
      overflow:auto;
      border: 1px solid var(--border);
      border-radius: 10px;
    }
    table {
      width:100%;
      border-collapse: collapse;
      font-size: 12px;
      min-width: 720px;
    }
    thead th {
      text-align:left;
      padding: 10px 10px;
      background: rgba(249,250,251,.8);
      border-bottom: 1px solid var(--border);
      color: var(--muted);
      font-weight: 900;
      white-space: nowrap;
    }
    [data-theme="dark"] thead th {
      background: rgba(17,24,39,.55);
    }
    tbody td {
      padding: 10px 10px;
      border-bottom: 1px solid var(--border);
      white-space: nowrap;
      color: var(--text);
    }
    .hint {
      font-size: 12px;
      color: var(--muted);
      margin-top: 8px;
    }
    a.nolink { text-decoration:none; }
  </style>
</head>
<body>
  <div class="container">
    <header>
      <div>
        <h1>Data Quality Dashboard</h1>
        <div class="sub">Click a metric bar to open drill-down details. Toggle trend dimensions using filters or legend.</div>
      </div>

      <div class="toolbar">
        <div class="pill">
          <span>Run</span>
          <select id="runSelect"></select>
        </div>

        <button id="themeToggle" class="secondary" type="button">Toggle Dark Mode</button>

        <div class="pill">
          <span>Export</span>
          <button id="btnPrint" class="secondary" type="button">Print (PDF)</button>
          <button id="btnPng" class="secondary" type="button">Chart PNG</button>
          <a id="lnkXlsx" class="nolink" href="__XLSX__" download><button class="secondary" type="button">Excel</button></a>
          <a id="lnkSummary" class="nolink" href="__SUMMARY__" download><button class="secondary" type="button">dq_summary.csv</button></a>
          <a id="lnkIssues" class="nolink" href="__ISSUES__" download><button class="secondary" type="button">dq_issues.csv</button></a>
        </div>
      </div>
    </header>

    <div class="kpi-grid">
      <div class="kpi">
        <div class="kpi-label">Overall DQ</div>
        <div id="kpiOverall" class="kpi-value">__KPI_OVERALL__</div>
        <div id="kpiOverallSub" class="kpi-sub">Δ vs previous: __KPI_DELTA__</div>
        <div id="kpiOverallBadge" class="badge __KPI_LEVEL__">__KPI_LEVEL_TXT__</div>
      </div>

      <div class="kpi">
        <div class="kpi-label">Lowest Dimension</div>
        <div id="kpiLowest" class="kpi-value">__KPI_LOW_TITLE__</div>
        <div id="kpiLowestSub" class="kpi-sub">__KPI_LOW_SUB__</div>
      </div>

      <div class="kpi">
        <div class="kpi-label">Rows (raw)</div>
        <div id="kpiRows" class="kpi-value">__KPI_ROWS__</div>
        <div id="kpiRules" class="kpi-sub">Rules: __KPI_RULES__</div>
      </div>

      <div class="kpi">
        <div class="kpi-label">Issues (current run)</div>
        <div id="kpiIssues" class="kpi-value">__KPI_ISSUES__</div>
        <div id="kpiIssuesSub" class="kpi-sub">__KPI_ISSUES_SUB__</div>
      </div>
    </div>

    <div class="grid">
      <div class="card">
        <h2>
          <span>Current DQ Score</span>
          <span class="muted">Click bars for details</span>
        </h2>
        <div id="scoreChart"></div>
      </div>

      <div class="card">
        <h2>
          <span>Trend</span>
          <span class="muted">Default: Overall</span>
        </h2>

        <div class="filters">
          <label class="chk"><input type="checkbox" data-trace="Overall" checked /> Overall</label>
          <label class="chk"><input type="checkbox" data-trace="Completeness" /> Completeness</label>
          <label class="chk"><input type="checkbox" data-trace="Validity" /> Validity</label>
          <label class="chk"><input type="checkbox" data-trace="Uniqueness" /> Uniqueness</label>
          <label class="chk"><input type="checkbox" data-trace="Timeliness" /> Timeliness</label>
        </div>

        <div id="trendChart"></div>
        <div id="trendHint" class="hint"></div>
      </div>
    </div>

    <div style="height:14px"></div>

    <div class="card">
      <h2>
        <span>Metric Details (Drill-down)</span>
        <span class="muted" id="detailsMeta">Select a metric</span>
      </h2>

      <div class="details-grid">
        <div>
          <div id="detailsSummary" class="muted">Click a bar (Completeness / Validity / Uniqueness / Timeliness / Overall) to view diagnostics.</div>
          <div id="detailsExtra" class="hint"></div>
        </div>
        <div>
          <div id="detailsTableWrap" class="table-wrap" style="display:none;"></div>
        </div>
      </div>
    </div>

    <div style="height:14px"></div>

    <div class="card">
      <h2>
        <span>Run History</span>
        <span class="muted">Run labels only (no timestamps)</span>
      </h2>
      <div class="table-wrap">
        <table id="histTable"></table>
      </div>
    </div>
  </div>

<script>
  const SCORE_FIG = __SCORE_JSON__;
  const TREND_FIG = __TREND_JSON__;
  const RUNS = __RUNS_JSON__;
  const DETAILS = __DETAILS_JSON__;

  const THRESH_GOOD = __THRESH_GOOD__;
  const THRESH_WARN = __THRESH_WARN__;

  const cfg = {
    displaylogo: false,
    responsive: true,
    displayModeBar: false
  };

  function setTheme(theme) {
    document.documentElement.setAttribute("data-theme", theme);
    localStorage.setItem("dq_theme", theme);
  }

  function initTheme() {
    const saved = localStorage.getItem("dq_theme");
    if (saved === "dark" || saved === "light") setTheme(saved);
    else setTheme("light");
  }

  function pct(x) {
    if (x === null || x === undefined || Number.isNaN(x)) return "N/A";
    return `${Number(x).toFixed(2)}%`;
  }

  function level(score) {
    if (score === null || score === undefined || Number.isNaN(score)) return "unknown";
    if (score >= THRESH_GOOD) return "good";
    if (score >= THRESH_WARN) return "warn";
    return "crit";
  }

  function lowestDimension(run) {
    const dims = [
      ["Completeness", run.completeness_pct],
      ["Validity", run.validity_pct],
      ["Uniqueness", run.uniqueness_pct],
      ["Timeliness", run.timeliness_pct],
    ];

    const available = dims.filter(([_, v]) => v !== null && v !== undefined && !Number.isNaN(v));
    const missing = dims.filter(([_, v]) => v === null || v === undefined || Number.isNaN(v)).map(([n]) => n);

    if (!available.length) return ["N/A", "No dimension scores available."];

    const vals = available.map(([_, v]) => Number(v));
    const lo = Math.min(...vals);
    const hi = Math.max(...vals);

    if (Math.abs(hi - lo) < 0.001) {
      const sub = missing.length ? `Missing: ${missing.join(", ")}` : "All metrics available.";
      return [`All dimensions tied (${lo.toFixed(2)}%)`, sub];
    }

    const worst = available.filter(([_, v]) => Math.abs(Number(v) - lo) < 0.001).map(([n]) => n);
    const sub = missing.length ? `Missing: ${missing.join(", ")}` : "All metrics available.";
    return [`${worst.join(", ")} (${lo.toFixed(2)}%)`, sub];
  }

  function renderRunSelect() {
    const sel = document.getElementById("runSelect");
    sel.innerHTML = "";
    RUNS.forEach((r, idx) => {
      const opt = document.createElement("option");
      opt.value = r.run_id;
      opt.textContent = r.run_label + (idx === RUNS.length - 1 ? " (latest)" : "");
      sel.appendChild(opt);
    });
    sel.value = RUNS[RUNS.length - 1].run_id;
  }

  function renderHistoryTable() {
    const t = document.getElementById("histTable");
    const cols = ["run_label", "rows_raw", "overall_pct", "completeness_pct", "validity_pct", "uniqueness_pct", "timeliness_pct"];
    const thead = `<thead><tr>${cols.map(c => `<th>${c}</th>`).join("")}</tr></thead>`;
    const tbody = `<tbody>${RUNS.map(r => {
      return `<tr>${
        cols.map(c => {
          const v = r[c];
          if (c.endsWith("_pct")) return `<td>${pct(v)}</td>`;
          return `<td>${v === null || v === undefined ? "N/A" : v}</td>`;
        }).join("")
      }</tr>`;
    }).join("")}</tbody>`;
    t.innerHTML = thead + tbody;
  }

  function plotScore(figDict) {
    Plotly.newPlot("scoreChart", figDict.data, figDict.layout, cfg);
  }

  function plotTrend(figDict) {
    const hint = document.getElementById("trendHint");
    if (!figDict) {
      hint.textContent = "Trend appears after 2+ runs.";
      return;
    }
    Plotly.newPlot("trendChart", figDict.data, figDict.layout, cfg);
    hint.textContent = RUNS.length < 5 ? `Limited history: ${RUNS.length} runs.` : "";
  }

  function applyTrendFilters() {
    const checks = Array.from(document.querySelectorAll('input[type="checkbox"][data-trace]'));
    const visible = new Set(checks.filter(c => c.checked).map(c => c.getAttribute("data-trace")));

    const gd = document.getElementById("trendChart");
    if (!gd || !gd.data) return;

    gd.data.forEach((trace, idx) => {
      const isOn = visible.has(trace.name);
      Plotly.restyle(gd, { visible: isOn ? true : "legendonly" }, [idx]);
    });
  }

  function detailsFor(runId, metricName) {
    const d = DETAILS[runId] || null;
    if (!d || !d.available) {
      return {
        title: metricName,
        summary: "Diagnostics are only available for the latest run (clean dataset snapshot required).",
        table: null,
        extra: ""
      };
    }

    const dims = (d.dimensions || {});
    if (metricName === "Overall") {
      return {
        title: "Overall",
        summary: "Overall is an aggregate score. Use dimension drill-down to identify root causes.",
        table: null,
        extra: d.summary || ""
      };
    }

    const payload = dims[metricName] || null;
    if (!payload) {
      return { title: metricName, summary: "No diagnostics available for this metric.", table: null, extra: d.summary || "" };
    }

    if (metricName === "Completeness") {
      const rows = payload.top_missing || [];
      return {
        title: metricName,
        summary: payload.summary || "",
        extra: d.summary || "",
        table: rows.length ? { columns: ["column", "missing_count", "missing_pct"], rows } : null
      };
    }

    if (metricName === "Validity") {
      const rows = payload.checks || [];
      return {
        title: metricName,
        summary: payload.summary || "",
        extra: d.summary || "",
        table: rows.length ? { columns: ["check", "invalid_count", "invalid_pct"], rows } : null
      };
    }

    if (metricName === "Uniqueness") {
      const meta = payload.meta || {};
      const ex = meta.examples || [];
      const rows = ex.map(x => ({ duplicate_key: x }));
      return {
        title: metricName,
        summary: payload.summary || "",
        extra: meta.key ? `Key column: ${meta.key}` : "",
        table: rows.length ? { columns: ["duplicate_key"], rows } : null
      };
    }

    if (metricName === "Timeliness") {
      const stats = payload.stats || {};
      const rows = Object.keys(stats).map(k => ({ metric: k, value: stats[k] }));
      return {
        title: metricName,
        summary: payload.summary || "",
        extra: payload.columns ? `Columns: ${payload.columns.order_ts} → ${payload.columns.ingested_ts}` : "",
        table: rows.length ? { columns: ["metric", "value"], rows } : null
      };
    }

    return { title: metricName, summary: payload.summary || "", table: null, extra: d.summary || "" };
  }

  function renderDetails(runId, metricName) {
    const meta = document.getElementById("detailsMeta");
    const summary = document.getElementById("detailsSummary");
    const extra = document.getElementById("detailsExtra");
    const wrap = document.getElementById("detailsTableWrap");

    const d = detailsFor(runId, metricName);

    const runLabel = (RUNS.find(r => r.run_id === runId)?.run_label || "");
    meta.textContent = `${metricName} · ${runLabel}`;
    summary.textContent = d.summary || "";
    extra.textContent = d.extra || "";

    if (!d.table) {
      wrap.style.display = "none";
      wrap.innerHTML = "";
      return;
    }

    const cols = d.table.columns;
    const rows = d.table.rows;

    const thead = `<thead><tr>${cols.map(c => `<th>${c}</th>`).join("")}</tr></thead>`;
    const tbody = `<tbody>${rows.map(r => `<tr>${cols.map(c => `<td>${r[c]}</td>`).join("")}</tr>`).join("")}</tbody>`;
    wrap.innerHTML = `<table>${thead + tbody}</table>`;
    wrap.style.display = "block";
  }

  function updateKpis(run) {
    document.getElementById("kpiOverall").textContent = pct(run.overall_pct);
    document.getElementById("kpiOverallSub").textContent = "Δ vs previous: —";

    const lvl = level(run.overall_pct);
    const badge = document.getElementById("kpiOverallBadge");
    badge.className = `badge ${lvl}`;
    badge.textContent = lvl.toUpperCase();

    const [lt, ls] = lowestDimension(run);
    document.getElementById("kpiLowest").textContent = lt;
    document.getElementById("kpiLowestSub").textContent = ls;

    document.getElementById("kpiRows").textContent =
      (run.rows_raw === null || run.rows_raw === undefined) ? "N/A" : run.rows_raw;
  }

  function updateScoreChart(run) {
    const fig = JSON.parse(JSON.stringify(SCORE_FIG));
    const labels = ["Completeness", "Validity", "Uniqueness", "Timeliness", "Overall"];
    const values = [run.completeness_pct, run.validity_pct, run.uniqueness_pct, run.timeliness_pct, run.overall_pct];

    const y = [];
    const text = [];
    const colors = [];

    for (const v of values) {
      if (v === null || v === undefined || Number.isNaN(v)) {
        y.push(0);
        text.push("");
        colors.push("#9ca3af");
      } else {
        y.push(Number(v));
        text.push(`${Number(v).toFixed(2)}%`);
        const lvl = level(Number(v));
        colors.push(lvl === "good" ? "#16a34a" : (lvl === "warn" ? "#f59e0b" : "#dc2626"));
      }
    }

    fig.data[0].x = labels;
    fig.data[0].y = y;
    fig.data[0].text = text;
    fig.data[0].marker = { color: colors };

    Plotly.react("scoreChart", fig.data, fig.layout, cfg);
  }

  function bindScoreClick() {
    const gd = document.getElementById("scoreChart");
    gd.on("plotly_click", (ev) => {
      const runId = document.getElementById("runSelect").value;
      const metric = ev?.points?.[0]?.x;
      if (metric) renderDetails(runId, metric);
    });
  }

  function bindTrendFilters() {
    const checks = Array.from(document.querySelectorAll('input[type="checkbox"][data-trace]'));
    checks.forEach(c => c.addEventListener("change", applyTrendFilters));
  }

  function bindRunSelect() {
    const sel = document.getElementById("runSelect");
    sel.addEventListener("change", () => {
      const runId = sel.value;
      const run = RUNS.find(r => r.run_id === runId) || RUNS[RUNS.length - 1];
      updateKpis(run);
      updateScoreChart(run);
      renderDetails(runId, "Overall");
    });
  }

  function bindExport() {
    document.getElementById("btnPrint").addEventListener("click", () => window.print());
    document.getElementById("btnPng").addEventListener("click", async () => {
      const scoreDiv = document.getElementById("scoreChart");
      const trendDiv = document.getElementById("trendChart");
      try { await Plotly.downloadImage(scoreDiv, {format: "png", filename: "dq_score"}); } catch (e) {}
      try { await Plotly.downloadImage(trendDiv, {format: "png", filename: "dq_trend"}); } catch (e) {}
    });
  }

  function bindTheme() {
    document.getElementById("themeToggle").addEventListener("click", () => {
      const cur = document.documentElement.getAttribute("data-theme") || "light";
      setTheme(cur === "dark" ? "light" : "dark");
    });
  }

  function init() {
    initTheme();
    renderRunSelect();
    renderHistoryTable();

    plotScore(SCORE_FIG);
    plotTrend(TREND_FIG);

    bindScoreClick();
    bindTrendFilters();
    bindRunSelect();
    bindExport();
    bindTheme();

    applyTrendFilters();

    const latestId = RUNS[RUNS.length - 1].run_id;
    renderDetails(latestId, "Overall");
  }

  init();
</script>
</body>
</html>
"""


def write_dashboard_html(
    runs: list[RunRow],
    current: RunRow,
    prev: dict[str, Any] | None,
    issues: pd.DataFrame | None,
    details_by_run_id: dict[str, Any],
) -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    prev_overall = _safe_float(prev.get("overall_pct")) if prev else None
    overall_delta = _kpi_delta(current.overall_pct, prev_overall)
    overall_level = _level_for_score(current.overall_pct)

    low_title, low_sub = _lowest_dimension(current)

    state, issue_count, _ = _issues_state(issues, current.run_id)
    if state == "missing":
        issues_value = "Missing"
        issues_sub = "Issue log not available."
    elif state == "none":
        issues_value = "0"
        issues_sub = "No issues detected."
    else:
        issues_value = str(issue_count)
        issues_sub = "Issues detected."

    score_fig = _fig_score_bar_for_run(current)
    trend_fig = _fig_trend(runs)

    score_json = json.dumps(score_fig.to_dict(), cls=PlotlyJSONEncoder)
    trend_json = "null" if trend_fig is None else json.dumps(trend_fig.to_dict(), cls=PlotlyJSONEncoder)

    runs_json = json.dumps([r.__dict__ for r in runs], cls=PlotlyJSONEncoder)
    details_json = json.dumps(details_by_run_id, cls=PlotlyJSONEncoder)

    xlsx_href = OUT_XLSX.name
    summary_href = DQ_SUMMARY_CSV.name
    issues_href = DQ_ISSUES_CSV.name if DQ_ISSUES_CSV.exists() else "#"

    html = (
        _HTML_TEMPLATE
        .replace("__XLSX__", xlsx_href)
        .replace("__SUMMARY__", summary_href)
        .replace("__ISSUES__", issues_href)
        .replace("__KPI_OVERALL__", _pct_str(current.overall_pct))
        .replace("__KPI_DELTA__", overall_delta)
        .replace("__KPI_LEVEL__", overall_level)
        .replace("__KPI_LEVEL_TXT__", overall_level.upper())
        .replace("__KPI_LOW_TITLE__", low_title)
        .replace("__KPI_LOW_SUB__", low_sub)
        .replace("__KPI_ROWS__", str(current.rows_raw if current.rows_raw is not None else "N/A"))
        .replace("__KPI_RULES__", str(current.rules_version or "—"))
        .replace("__KPI_ISSUES__", issues_value)
        .replace("__KPI_ISSUES_SUB__", issues_sub)
        .replace("__SCORE_JSON__", score_json)
        .replace("__TREND_JSON__", trend_json)
        .replace("__RUNS_JSON__", runs_json)
        .replace("__DETAILS_JSON__", details_json)
        .replace("__THRESH_GOOD__", str(THRESH_GOOD))
        .replace("__THRESH_WARN__", str(THRESH_WARN))
    )

    OUT_HTML.write_text(html, encoding="utf-8")
    print(f"Wrote: {OUT_HTML} | exists: {OUT_HTML.exists()}")


def main() -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    hist = _read_csv_if_exists(DQ_SUMMARY_CSV)
    if hist is None or hist.empty:
        raise FileNotFoundError(f"Missing or empty: {DQ_SUMMARY_CSV}")

    hist = _parse_run_ts(hist)
    runs = _runs_with_labels(hist)
    current = runs[-1]
    prev = _prev_run(hist)

    details_by_run_id: dict[str, Any] = {}
    current_details = _build_details_for_clean(CLEAN_CSV)
    details_by_run_id[current.run_id] = current_details

    # Backfill uniqueness_pct for current run if missing (computed from clean.csv)
    uniq_score = current_details.get("dimensions", {}).get("Uniqueness", {}).get("score")
    if current.uniqueness_pct is None and uniq_score is not None:
        backfill = float(uniq_score)
        runs[-1] = RunRow(
            run_id=current.run_id,
            run_label=current.run_label,
            rows_raw=current.rows_raw,
            completeness_pct=current.completeness_pct,
            validity_pct=current.validity_pct,
            uniqueness_pct=backfill,
            timeliness_pct=current.timeliness_pct,
            overall_pct=current.overall_pct,
            rules_version=current.rules_version,
        )
        current = runs[-1]
        if "uniqueness_pct" in hist.columns:
            hist.loc[hist.index[-1], "uniqueness_pct"] = backfill

    issues = _read_csv_if_exists(DQ_ISSUES_CSV)

    try:
        _write_xlsx(hist, issues, details_by_run_id.get(current.run_id, {}))
    except Exception:
        pass

    write_dashboard_html(
        runs=runs,
        current=current,
        prev=prev,
        issues=issues,
        details_by_run_id=details_by_run_id,
    )


if __name__ == "__main__":
    main()
