from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
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

THRESH_GOOD = 99.0
THRESH_WARN = 97.0


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


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


def _level(score: float | None) -> str:
    if score is None:
        return "unknown"
    if score >= THRESH_GOOD:
        return "good"
    if score >= THRESH_WARN:
        return "warn"
    return "crit"


def _read_csv(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    try:
        return pd.read_csv(path)
    except Exception:
        return None


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


def _build_runs(hist: pd.DataFrame) -> list[RunRow]:
    h = hist.reset_index(drop=True).copy()
    runs: list[RunRow] = []
    for i in range(len(h)):
        d = h.iloc[i].to_dict()
        runs.append(
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
    return runs


def _score_bar(run: RunRow) -> go.Figure:
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
            lvl = _level(float(v))
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


def _trend(runs: list[RunRow]) -> go.Figure | None:
    if len(runs) < 2:
        return None

    x = [r.run_label for r in runs]

    def series(attr: str) -> list[float | None]:
        return [getattr(r, attr) for r in runs]

    fig = go.Figure()
    for name, attr, default_visible in [
        ("Overall", "overall_pct", True),
        ("Completeness", "completeness_pct", False),
        ("Validity", "validity_pct", False),
        ("Uniqueness", "uniqueness_pct", False),
        ("Timeliness", "timeliness_pct", False),
    ]:
        fig.add_trace(
            go.Scatter(
                x=x,
                y=series(attr),
                mode="lines+markers",
                name=name,
                visible=True if default_visible else "legendonly",
                hovertemplate=f"{name}: %{{y:.2f}}%<br>Run: %{{x}}<extra></extra>",
            )
        )

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
    fig.update_yaxes(title_text="%", range=[90, 100], automargin=True)
    fig.update_xaxes(type="category", automargin=True)
    return fig


def _write_xlsx(hist: pd.DataFrame, issues: pd.DataFrame | None) -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(OUT_XLSX, engine="openpyxl") as xw:
        hist.to_excel(xw, sheet_name="dq_summary", index=False)
        if issues is not None:
            issues.to_excel(xw, sheet_name="dq_issues", index=False)


def main() -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    hist = _read_csv(DQ_SUMMARY_CSV)
    if hist is None or hist.empty:
        raise FileNotFoundError(f"Missing or empty: {DQ_SUMMARY_CSV}")

    runs = _build_runs(hist)
    current = runs[-1]

    issues = _read_csv(DQ_ISSUES_CSV)

    # Excel export (best-effort)
    try:
        _write_xlsx(hist, issues)
        print(f"Wrote: {OUT_XLSX} | exists: {OUT_XLSX.exists()}")
    except Exception:
        pass

    score_fig = _score_bar(current)
    trend_fig = _trend(runs)

    score_json = json.dumps(score_fig.to_dict(), cls=PlotlyJSONEncoder)
    trend_json = "null" if trend_fig is None else json.dumps(trend_fig.to_dict(), cls=PlotlyJSONEncoder)

    runs_json = json.dumps([r.__dict__ for r in runs], cls=PlotlyJSONEncoder)

    issues_json = "null"
    if issues is not None and not issues.empty:
        issues_json = json.dumps(issues.to_dict(orient="records"), cls=PlotlyJSONEncoder)

    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Data Quality Dashboard</title>
  <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
  <style>
    :root {{
      --text:#111827; --muted:#6b7280; --border:#e5e7eb; --card:#ffffff; --bg:#f8fafc;
      --good:#16a34a; --warn:#f59e0b; --crit:#dc2626; --shadow: 0 1px 2px rgba(0,0,0,.04);
    }}
    [data-theme="dark"] {{
      --text:#e5e7eb; --muted:#9ca3af; --border:#1f2937; --card:#0b1220; --bg:#060b16;
      --shadow: 0 1px 2px rgba(0,0,0,.35);
    }}
    body {{ margin:0; background:var(--bg); color:var(--text);
      font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Arial; }}
    .container {{ max-width: 1280px; margin: 26px auto; padding: 0 18px 40px; }}
    header {{ display:flex; align-items:flex-end; justify-content:space-between; gap:16px; margin-bottom: 14px; }}
    h1 {{ margin:0; font-size: 30px; letter-spacing: -0.02em; }}
    .sub {{ color: var(--muted); font-size: 12.5px; margin-top: 6px; }}
    .toolbar {{ display:flex; gap:10px; align-items:center; justify-content:flex-end; flex-wrap: wrap; }}
    .pill {{ border:1px solid var(--border); background:var(--card); border-radius:999px; padding:7px 10px;
      font-size:12px; color:var(--muted); box-shadow: var(--shadow); display:flex; gap:8px; align-items:center; }}
    select, button {{ border:1px solid var(--border); background:var(--card); color:var(--text);
      border-radius:10px; padding:8px 10px; font-size:12px; cursor:pointer; box-shadow: var(--shadow); }}
    .kpi-grid {{ display:grid; grid-template-columns: 1fr; gap: 12px; margin: 14px 0 18px; }}
    @media (min-width: 900px) {{ .kpi-grid {{ grid-template-columns: repeat(4, 1fr); }} }}
    .kpi {{ border:1px solid var(--border); border-radius: 14px; background: var(--card);
      padding: 12px; box-shadow: var(--shadow); min-height: 90px; position: relative; }}
    .kpi-label {{ color: var(--muted); font-size: 12px; font-weight: 700; text-transform: uppercase; letter-spacing: .04em; }}
    .kpi-value {{ font-size: 20px; font-weight: 800; margin-top: 8px; }}
    .kpi-sub {{ color: var(--muted); font-size: 12px; margin-top: 6px; line-height: 1.35; }}
    .badge {{ position:absolute; top:10px; right:10px; font-size:11px; font-weight:800; padding:4px 8px;
      border-radius:999px; border:1px solid var(--border); }}
    .badge.good {{ color: var(--good); }}
    .badge.warn {{ color: var(--warn); }}
    .badge.crit {{ color: var(--crit); }}
    .grid {{ display:grid; grid-template-columns: 1fr; gap: 14px; align-items: start; }}
    @media (min-width: 1050px) {{ .grid {{ grid-template-columns: 1fr 1fr; }} }}
    .card {{ border:1px solid var(--border); border-radius: 14px; background: var(--card);
      padding: 12px; box-shadow: var(--shadow); }}
    .card h2 {{ margin: 2px 0 10px; font-size: 14px; font-weight: 900; letter-spacing: -0.01em;
      display:flex; align-items:center; justify-content:space-between; gap: 10px; }}
    .muted {{ color: var(--muted); font-size: 12.5px; line-height: 1.45; }}
    .table-wrap {{ overflow:auto; border: 1px solid var(--border); border-radius: 10px; }}
    table {{ width:100%; border-collapse: collapse; font-size: 12px; min-width: 900px; }}
    thead th {{ text-align:left; padding: 10px 10px; background: rgba(249,250,251,.8);
      border-bottom: 1px solid var(--border); color: var(--muted); font-weight: 900; white-space: nowrap; }}
    [data-theme="dark"] thead th {{ background: rgba(17,24,39,.55); }}
    tbody td {{ padding: 10px 10px; border-bottom: 1px solid var(--border); white-space: nowrap; color: var(--text); }}
  </style>
</head>
<body>
  <div class="container">
    <header>
      <div>
        <h1>Data Quality Dashboard</h1>
        <div class="sub">Interactive dashboard generated from dq_summary and dq_issues.</div>
      </div>
      <div class="toolbar">
        <div class="pill">
          <span>Run</span>
          <select id="runSelect"></select>
        </div>
        <button id="themeToggle" type="button">Toggle Dark Mode</button>
        <div class="pill">
          <span>Export</span>
          <a href="{OUT_XLSX.name}" download style="text-decoration:none;"><button type="button">Excel</button></a>
          <button id="btnPrint" type="button">Print (PDF)</button>
        </div>
      </div>
    </header>

    <div class="kpi-grid">
      <div class="kpi">
        <div class="kpi-label">Overall DQ</div>
        <div id="kpiOverall" class="kpi-value">{_pct_str(current.overall_pct)}</div>
        <div id="kpiRules" class="kpi-sub">Rules: {current.rules_version or "—"}</div>
        <div id="kpiBadge" class="badge {_level(current.overall_pct)}">{_level(current.overall_pct).upper()}</div>
      </div>
      <div class="kpi">
        <div class="kpi-label">Rows (raw)</div>
        <div id="kpiRows" class="kpi-value">{current.rows_raw if current.rows_raw is not None else "N/A"}</div>
        <div class="kpi-sub">Source: data/synthetic/input.csv</div>
      </div>
      <div class="kpi">
        <div class="kpi-label">Report</div>
        <div class="kpi-value">HTML + XLSX</div>
        <div class="kpi-sub">{_utc_now().strftime("%Y-%m-%d %H:%M UTC")}</div>
      </div>
      <div class="kpi">
        <div class="kpi-label">Issues</div>
        <div id="kpiIssues" class="kpi-value">—</div>
        <div class="kpi-sub">From dq_issues.csv</div>
      </div>
    </div>

    <div class="grid">
      <div class="card">
        <h2><span>Current DQ Score</span><span class="muted">Hover for details</span></h2>
        <div id="scoreChart"></div>
      </div>
      <div class="card">
        <h2><span>Trend</span><span class="muted">Legend toggles</span></h2>
        <div id="trendChart"></div>
      </div>
    </div>

    <div style="height:14px"></div>

    <div class="card">
      <h2><span>Issue Details (Current Run)</span><span class="muted" id="issueMeta"></span></h2>
      <div class="table-wrap">
        <table id="issuesTable"></table>
      </div>
    </div>

    <div style="height:14px"></div>

    <div class="card">
      <h2><span>Run History</span><span class="muted">dq_summary.csv</span></h2>
      <div class="table-wrap">
        <table id="histTable"></table>
      </div>
    </div>
  </div>

<script>
  const SCORE_FIG = {score_json};
  const TREND_FIG = {trend_json};
  const RUNS = {runs_json};
  const ISSUES = {issues_json};

  const cfg = {{ displaylogo:false, responsive:true, displayModeBar:false }};

  function setTheme(theme) {{
    document.documentElement.setAttribute("data-theme", theme);
    localStorage.setItem("dq_theme", theme);
  }}

  function initTheme() {{
    const saved = localStorage.getItem("dq_theme");
    if (saved === "dark" || saved === "light") setTheme(saved);
    else setTheme("light");
  }}

  function pct(x) {{
    if (x === null || x === undefined || Number.isNaN(x)) return "N/A";
    return `${{Number(x).toFixed(2)}}%`;
  }}

  function renderRunSelect() {{
    const sel = document.getElementById("runSelect");
    sel.innerHTML = "";
    RUNS.forEach((r, idx) => {{
      const opt = document.createElement("option");
      opt.value = r.run_id;
      opt.textContent = r.run_label + (idx === RUNS.length - 1 ? " (latest)" : "");
      sel.appendChild(opt);
    }});
    sel.value = RUNS[RUNS.length - 1].run_id;
  }}

  function renderHistoryTable() {{
    const t = document.getElementById("histTable");
    const cols = ["run_label","rows_raw","overall_pct","completeness_pct","validity_pct","uniqueness_pct","timeliness_pct","rules_version"];
    const thead = `<thead><tr>${{cols.map(c => `<th>${{c}}</th>`).join("")}}</tr></thead>`;
    const tbody = `<tbody>${{RUNS.map(r => {{
      return `<tr>${{cols.map(c => {{
        const v = r[c];
        if (c.endsWith("_pct")) return `<td>${{pct(v)}}</td>`;
        return `<td>${{(v===null||v===undefined) ? "—" : v}}</td>`;
      }}).join("")}}</tr>`;
    }}).join("")}}</tbody>`;
    t.innerHTML = thead + tbody;
  }}

  function plotScore(figDict) {{
    Plotly.newPlot("scoreChart", figDict.data, figDict.layout, cfg);
  }}

  function plotTrend(figDict) {{
    if (!figDict) {{
      document.getElementById("trendChart").innerHTML = "<div class='muted'>Trend appears after 2+ runs.</div>";
      return;
    }}
    Plotly.newPlot("trendChart", figDict.data, figDict.layout, cfg);
  }}

  function issuesForRun(runId) {{
    if (!ISSUES) return [];
    return ISSUES.filter(x => String(x.run_id) === String(runId));
  }}

  function renderIssues(runId) {{
    const rows = issuesForRun(runId);
    document.getElementById("kpiIssues").textContent = String(rows.length);
    document.getElementById("issueMeta").textContent = rows.length ? `run_id: ${{runId}}` : `No issues for run_id: ${{runId}}`;

    const t = document.getElementById("issuesTable");
    const cols = ["dimension","rule","column","failed_count","failed_pct","sample_values"];
    const thead = `<thead><tr>${{cols.map(c => `<th>${{c}}</th>`).join("")}}</tr></thead>`;
    const tbody = `<tbody>${{rows.map(r => {{
      return `<tr>${{cols.map(c => `<td>${{r[c]}}</td>`).join("")}}</tr>`;
    }}).join("")}}</tbody>`;
    t.innerHTML = thead + tbody;
  }}

  function updateScoreForRun(runId) {{
    const run = RUNS.find(r => String(r.run_id) === String(runId)) || RUNS[RUNS.length - 1];
    const labels = ["Completeness","Validity","Uniqueness","Timeliness","Overall"];
    const vals = [run.completeness_pct, run.validity_pct, run.uniqueness_pct, run.timeliness_pct, run.overall_pct];

    const fig = JSON.parse(JSON.stringify(SCORE_FIG));
    const y = [];
    const text = [];
    const colors = [];

    function level(score) {{
      if (score === null || score === undefined || Number.isNaN(score)) return "unknown";
      if (score >= {THRESH_GOOD}) return "good";
      if (score >= {THRESH_WARN}) return "warn";
      return "crit";
    }}

    for (const v of vals) {{
      if (v === null || v === undefined || Number.isNaN(v)) {{
        y.push(0); text.push(""); colors.push("#9ca3af");
      }} else {{
        y.push(Number(v));
        text.push(`${{Number(v).toFixed(2)}}%`);
        const lvl = level(Number(v));
        colors.push(lvl === "good" ? "#16a34a" : (lvl === "warn" ? "#f59e0b" : "#dc2626"));
      }}
    }}

    fig.data[0].x = labels;
    fig.data[0].y = y;
    fig.data[0].text = text;
    fig.data[0].marker = {{ color: colors }};
    Plotly.react("scoreChart", fig.data, fig.layout, cfg);

    document.getElementById("kpiOverall").textContent = pct(run.overall_pct);
    document.getElementById("kpiRows").textContent = (run.rows_raw === null || run.rows_raw === undefined) ? "N/A" : run.rows_raw;
    document.getElementById("kpiRules").textContent = "Rules: " + (run.rules_version || "—");

    const badge = document.getElementById("kpiBadge");
    const lvl = level(run.overall_pct);
    badge.className = "badge " + lvl;
    badge.textContent = lvl.toUpperCase();
  }}

  function bind() {{
    document.getElementById("runSelect").addEventListener("change", (e) => {{
      const runId = e.target.value;
      updateScoreForRun(runId);
      renderIssues(runId);
    }});

    document.getElementById("btnPrint").addEventListener("click", () => window.print());

    document.getElementById("themeToggle").addEventListener("click", () => {{
      const cur = document.documentElement.getAttribute("data-theme") || "light";
      setTheme(cur === "dark" ? "light" : "dark");
    }});
  }}

  function init() {{
    initTheme();
    renderRunSelect();
    renderHistoryTable();
    plotScore(SCORE_FIG);
    plotTrend(TREND_FIG);
    bind();

    const latestId = RUNS[RUNS.length - 1].run_id;
    updateScoreForRun(latestId);
    renderIssues(latestId);
  }}

  init();
</script>
</body>
</html>
"""
    OUT_HTML.write_text(html, encoding="utf-8")
    print(f"Wrote: {OUT_HTML} | exists: {OUT_HTML.exists()}")


if __name__ == "__main__":
    main()
