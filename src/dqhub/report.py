from __future__ import annotations

from pathlib import Path
from datetime import datetime, timezone
from typing import Any
import json

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]

REPORTS_DIR = ROOT / "reports"
OUTPUT_DIR = ROOT / "data" / "output"

SUMMARY_CSV = REPORTS_DIR / "dq_summary.csv"
ISSUES_CSV = REPORTS_DIR / "dq_issues.csv"
CLEAN_CSV = OUTPUT_DIR / "clean.csv"

HTML_OUT = REPORTS_DIR / "DQ_Report.html"
XLSX_OUT = REPORTS_DIR / "DQ_Report.xlsx"


METRICS = [
    ("completeness_pct", "Completeness"),
    ("validity_pct", "Validity"),
    ("uniqueness_pct", "Uniqueness"),
    ("timeliness_pct", "Timeliness"),
    ("overall_pct", "Overall"),
]

DIMS = [
    ("completeness_pct", "Completeness"),
    ("validity_pct", "Validity"),
    ("uniqueness_pct", "Uniqueness"),
    ("timeliness_pct", "Timeliness"),
]


def _now_utc_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _ensure_report_dirs() -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def _read_summary() -> pd.DataFrame:
    if not SUMMARY_CSV.exists():
        raise FileNotFoundError(f"Missing {SUMMARY_CSV}. Run `dqhub clean` first.")
    df = pd.read_csv(SUMMARY_CSV)
    if df.empty:
        raise ValueError("dq_summary.csv is empty.")
    if "run_label" not in df.columns:
        df.insert(0, "run_label", [f"Run {i+1}" for i in range(len(df))])
    return df


def _read_issues() -> pd.DataFrame:
    if not ISSUES_CSV.exists():
        return pd.DataFrame(
            columns=["run_label", "run_id", "dimension", "rule", "column", "failed_count", "failed_pct", "sample_values"]
        )
    df = pd.read_csv(ISSUES_CSV)
    if "run_label" not in df.columns:
        df["run_label"] = ""
    if "run_id" not in df.columns:
        df["run_id"] = ""
    return df


def _dataset_line() -> str:
    if not CLEAN_CSV.exists():
        return "Dataset not loaded."
    try:
        df = pd.read_csv(CLEAN_CSV)
        r, c = df.shape
        return f"Dataset loaded: clean.csv ({r} rows, {c} columns)."
    except Exception:
        return "Dataset loaded: clean.csv."


def _to_records(df: pd.DataFrame) -> list[dict[str, Any]]:
    # Convert NaN to None for clean JSON
    out: list[dict[str, Any]] = []
    for row in df.to_dict(orient="records"):
        clean = {}
        for k, v in row.items():
            if pd.isna(v):
                clean[k] = None
            else:
                clean[k] = v
        out.append(clean)
    return out


def _write_excel(summary: pd.DataFrame, issues: pd.DataFrame) -> None:
    with pd.ExcelWriter(XLSX_OUT, engine="openpyxl") as writer:
        summary.to_excel(writer, index=False, sheet_name="Summary")
        issues.to_excel(writer, index=False, sheet_name="Issues")

        wb = writer.book
        for ws_name in ["Summary", "Issues"]:
            ws = wb[ws_name]
            ws.freeze_panes = "A2"
            # basic width autosize
            for col in ws.columns:
                col_letter = col[0].column_letter
                max_len = 10
                for cell in col[:200]:
                    if cell.value is None:
                        continue
                    max_len = max(max_len, len(str(cell.value)))
                ws.column_dimensions[col_letter].width = min(max_len + 2, 45)


def _build_html(summary_records: list[dict[str, Any]], issues_records: list[dict[str, Any]]) -> str:
    generated = _now_utc_iso()
    dataset_line = _dataset_line()

    # Keep file names relative (works when opening reports/DQ_Report.html locally)
    xlsx_name = XLSX_OUT.name
    summary_name = SUMMARY_CSV.name
    issues_name = ISSUES_CSV.name

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>Data Quality Dashboard</title>
  <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
  <style>
    :root {{
      --bg: #f6f7fb; --panel: #ffffff; --text: #111827; --muted: #6b7280; --border: #e5e7eb;
      --shadow: 0 1px 2px rgba(0,0,0,0.05); --good: #16a34a; --chip: #eef2ff;
    }}
    html.dark {{
      --bg: #0b1220; --panel: #0f172a; --text: #e5e7eb; --muted: #9ca3af; --border: #1f2937;
      --shadow: 0 1px 2px rgba(0,0,0,0.35); --good: #22c55e; --chip: rgba(99,102,241,0.12);
    }}
    body {{
      margin: 0; background: var(--bg); color: var(--text);
      font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial;
    }}
    .container {{ max-width: 1120px; margin: 28px auto 60px; padding: 0 18px; }}
    header {{ display:flex; align-items:flex-start; justify-content:space-between; gap:16px; margin-bottom:14px; }}
    h1 {{ margin:0; font-size:28px; letter-spacing:-0.02em; }}
    .sub {{ margin-top:6px; color:var(--muted); font-size:13px; line-height:1.4; }}
    .toolbar {{ display:flex; gap:10px; align-items:center; flex-wrap:wrap; justify-content:flex-end; }}
    .btn, select {{
      border: 1px solid var(--border); background: var(--panel); color: var(--text);
      padding: 8px 10px; border-radius: 999px; font-size: 12px; box-shadow: var(--shadow); cursor: pointer;
      text-decoration: none;
    }}
    select {{ cursor:pointer; padding-right:28px; }}
    .cards {{ display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:12px; margin-top:16px; }}
    @media (max-width: 980px) {{ .cards {{ grid-template-columns:repeat(2,minmax(0,1fr)); }} }}
    @media (max-width: 520px) {{
      .cards {{ grid-template-columns:1fr; }}
      header {{ flex-direction:column; align-items:stretch; }}
      .toolbar {{ justify-content:flex-start; }}
    }}
    .card {{
      background: var(--panel); border: 1px solid var(--border); border-radius: 12px;
      padding: 14px; box-shadow: var(--shadow);
    }}
    .card .label {{
      font-size: 11px; color: var(--muted); text-transform: uppercase; letter-spacing: 0.06em;
      display:flex; align-items:center; justify-content:space-between; gap:10px;
    }}
    .chip {{
      font-size: 11px; padding: 3px 8px; border-radius: 999px; background: var(--chip);
      border: 1px solid var(--border); color: var(--good); font-weight: 700; text-transform: uppercase; letter-spacing: 0.04em;
    }}
    .value {{ font-size: 20px; font-weight: 800; margin-top: 6px; }}
    .meta {{ font-size: 12px; color: var(--muted); margin-top: 4px; }}
    .grid2 {{ display:grid; grid-template-columns:1.2fr 1fr; gap:12px; margin-top:12px; align-items:stretch; }}
    @media (max-width: 980px) {{ .grid2 {{ grid-template-columns:1fr; }} }}
    .panel {{
      background: var(--panel); border: 1px solid var(--border); border-radius: 12px; padding: 12px; box-shadow: var(--shadow);
    }}
    .panel-header {{ display:flex; align-items:center; justify-content:space-between; gap:12px; margin-bottom:6px; }}
    .panel-title {{ font-weight:800; font-size:13px; }}
    .panel-note {{ font-size:12px; color: var(--muted); }}
    .checks {{ display:flex; gap:10px; flex-wrap:wrap; align-items:center; color:var(--muted); font-size:12px; margin-bottom:6px; }}
    .checks label {{ display:inline-flex; align-items:center; gap:6px; cursor:pointer; user-select:none; }}
    .section {{ margin-top: 14px; }}
    table {{
      width:100%; border-collapse:collapse; font-size:12px; overflow:hidden;
      border-radius:10px; border:1px solid var(--border);
    }}
    th, td {{ padding:10px; border-bottom:1px solid var(--border); vertical-align:top; }}
    th {{
      text-align:left; background: rgba(148,163,184,0.08); color: var(--muted);
      font-weight:700; font-size:11px; text-transform:lowercase;
    }}
    tr:last-child td {{ border-bottom:none; }}
    .right {{ text-align:right; }}
    .hint {{ font-size:12px; color: var(--muted); margin-top:6px; }}
  </style>
</head>
<body>
  <div class="container">
    <header>
      <div>
        <h1>Data Quality Dashboard</h1>
        <div class="sub">
          Click a metric bar to open drill-down details. Toggle trend dimensions using filters or legend.<br/>
          <span style="opacity:0.85">Generated: {generated}</span>
        </div>
      </div>
      <div class="toolbar">
        <button class="btn" id="btnRun">Run</button>
        <select id="runSelect" aria-label="Select run"></select>
        <button class="btn" id="btnDark">Toggle Dark Mode</button>
      </div>
    </header>

    <div class="toolbar" style="justify-content:flex-end; margin-top: 8px;">
      <button class="btn" id="btnExport">Export</button>
      <button class="btn" id="btnPrint">Print (PDF)</button>
      <button class="btn" id="btnPng">Chart PNG</button>
      <a class="btn" href="{xlsx_name}" download>Excel</a>
      <a class="btn" href="{summary_name}" download>dq_summary.csv</a>
      <a class="btn" href="{issues_name}" download>dq_issues.csv</a>
    </div>

    <div class="cards">
      <div class="card">
        <div class="label"><span>Overall DQ</span><span class="chip" id="chipStatus">GOOD</span></div>
        <div class="value" id="overallValue">N/A</div>
        <div class="meta" id="overallDelta">Δ vs previous: N/A</div>
      </div>

      <div class="card">
        <div class="label"><span>Lowest Dimension</span></div>
        <div class="value" id="lowestValue">N/A</div>
        <div class="meta">All metrics available.</div>
      </div>

      <div class="card">
        <div class="label"><span>Rows (Raw)</span></div>
        <div class="value" id="rowsValue">0</div>
        <div class="meta" id="rulesMeta">Rules: N/A</div>
      </div>

      <div class="card">
        <div class="label"><span>Issues (Current Run)</span></div>
        <div class="value" id="issuesValue">0</div>
        <div class="meta" id="issuesMeta">No issues detected.</div>
      </div>
    </div>

    <div class="grid2">
      <div class="panel">
        <div class="panel-header"><div class="panel-title">Current DQ Score</div><div class="panel-note">Click bars for details</div></div>
        <div id="barChart" style="height:320px;"></div>
      </div>

      <div class="panel">
        <div class="panel-header">
          <div><div class="panel-title">Trend</div><div class="checks" id="trendChecks"></div></div>
          <div class="panel-note" id="trendMode">Default: Overall</div>
        </div>
        <div id="trendChart" style="height:320px;"></div>
        <div class="hint" id="historyHint"></div>
      </div>
    </div>

    <div class="section panel">
      <div class="panel-header">
        <div><div class="panel-title">Metric Details (Drill-down)</div><div class="panel-note">Overall is an aggregate score. Use dimension drill-down to identify root causes.</div></div>
        <div class="panel-note" id="drillTitle">Overall -</div>
      </div>
      <div class="panel-note" style="margin-bottom: 8px;">{dataset_line}</div>
      <div id="drillTableWrap"></div>
    </div>

    <div class="section panel">
      <div class="panel-header"><div class="panel-title">Run History</div><div class="panel-note">Run labels only (no timestamps)</div></div>
      <div id="historyTableWrap"></div>
    </div>
  </div>

<script>
  const SUMMARY = {json.dumps(summary_records, ensure_ascii=False)};
  const ISSUES  = {json.dumps(issues_records, ensure_ascii=False)};
  const METRICS = {json.dumps([{"key": k, "label": lbl} for k, lbl in METRICS], ensure_ascii=False)};
  const DIMS    = {json.dumps([{"key": k, "label": lbl} for k, lbl in DIMS], ensure_ascii=False)};

  const els = {{
    runSelect: document.getElementById("runSelect"),
    btnDark: document.getElementById("btnDark"),
    btnPrint: document.getElementById("btnPrint"),
    btnPng: document.getElementById("btnPng"),
    btnExport: document.getElementById("btnExport"),
    btnRun: document.getElementById("btnRun"),
    overallValue: document.getElementById("overallValue"),
    overallDelta: document.getElementById("overallDelta"),
    chipStatus: document.getElementById("chipStatus"),
    lowestValue: document.getElementById("lowestValue"),
    rowsValue: document.getElementById("rowsValue"),
    rulesMeta: document.getElementById("rulesMeta"),
    issuesValue: document.getElementById("issuesValue"),
    issuesMeta: document.getElementById("issuesMeta"),
    trendChecks: document.getElementById("trendChecks"),
    historyHint: document.getElementById("historyHint"),
    drillTitle: document.getElementById("drillTitle"),
    drillTableWrap: document.getElementById("drillTableWrap"),
    historyTableWrap: document.getElementById("historyTableWrap"),
  }};

  function fmtPct(v) {{
    if (v === null || v === undefined || Number.isNaN(Number(v))) return "N/A";
    return Number(v).toFixed(2) + "%";
  }}
  function safeNum(v, fallback=0) {{
    const n = Number(v);
    return Number.isFinite(n) ? n : fallback;
  }}
  function getRunLabels() {{ return SUMMARY.map(r => String(r.run_label)); }}
  function getRunRow(label) {{ return SUMMARY.find(r => String(r.run_label) === String(label)); }}
  function getPrevLabel(label) {{
    const labels = getRunLabels();
    const idx = labels.indexOf(String(label));
    if (idx <= 0) return null;
    return labels[idx - 1];
  }}
  function deltaPP(metricKey, label) {{
    const prev = getPrevLabel(label);
    if (!prev) return null;
    const curRow = getRunRow(label);
    const prevRow = getRunRow(prev);
    if (!curRow || !prevRow) return null;
    const cur = safeNum(curRow[metricKey], NaN);
    const prv = safeNum(prevRow[metricKey], NaN);
    if (!Number.isFinite(cur) || !Number.isFinite(prv)) return null;
    return cur - prv;
  }}
  function lowestDimension(row) {{
    let best = {{ label: "N/A", val: null }};
    DIMS.forEach(d => {{
      const v = safeNum(row[d.key], NaN);
      if (!Number.isFinite(v)) return;
      if (best.val === null || v < best.val) best = {{ label: d.label, val: v }};
    }});
    return best;
  }}
  function issuesForRun(label) {{
    const runIssues = ISSUES.filter(x => String(x.run_label || "") === String(label));
    return runIssues.filter(x => safeNum(x.failed_count, 0) > 0);
  }}
  function buildRunSelect() {{
    els.runSelect.innerHTML = "";
    const labels = getRunLabels();
    labels.forEach((lbl, i) => {{
      const opt = document.createElement("option");
      opt.value = lbl;
      const isLatest = i === labels.length - 1;
      opt.textContent = isLatest ? `${{lbl}} (latest)` : lbl;
      els.runSelect.appendChild(opt);
    }});
    els.runSelect.value = labels[labels.length - 1];
  }}
  function statusChip(overallPct) {{
    if (overallPct >= 95) return {{ text: "GOOD", color: "var(--good)" }};
    if (overallPct >= 85) return {{ text: "OK", color: "#f59e0b" }};
    return {{ text: "RISK", color: "#ef4444" }};
  }}
  function htmlEscape(s) {{
    return String(s).replaceAll("&","&amp;").replaceAll("<","&lt;").replaceAll(">","&gt;")
      .replaceAll('"',"&quot;").replaceAll("'","&#039;");
  }}

  let activeMetric = "overall_pct";

  function renderCards(label) {{
    const row = getRunRow(label);
    const overall = safeNum(row.overall_pct, 0);
    els.overallValue.textContent = fmtPct(overall);

    const d = deltaPP("overall_pct", label);
    els.overallDelta.textContent = "Δ vs previous: " + (d === null ? "N/A" : (d >= 0 ? "+" : "") + d.toFixed(2) + " pp");

    const st = statusChip(overall);
    els.chipStatus.textContent = st.text;
    els.chipStatus.style.color = st.color;

    const low = lowestDimension(row);
    els.lowestValue.textContent = `${{low.label}} (${{fmtPct(low.val)}})`;

    els.rowsValue.textContent = String(row.rows_raw ?? "");
    els.rulesMeta.textContent = "Rules: " + String(row.rules_version ?? "N/A");

    const curIssues = issuesForRun(label);
    els.issuesValue.textContent = String(curIssues.length);
    els.issuesMeta.textContent = curIssues.length === 0 ? "No issues detected." : "See drill-down for details.";
  }}

  function renderBar(label) {{
    const row = getRunRow(label);
    const x = METRICS.map(m => m.label);
    const y = METRICS.map(m => safeNum(row[m.key], 0));

    Plotly.newPlot("barChart",
      [{{ type:"bar", x, y, hovertemplate:"Metric: %{{x}}<br>Score: %{{y:.2f}}%<extra></extra>" }}],
      {{
        margin: {{ l:45, r:15, t:10, b:45 }},
        yaxis: {{ range:[0,100], ticksuffix:"%", gridcolor:"rgba(148,163,184,0.25)" }},
        paper_bgcolor:"rgba(0,0,0,0)", plot_bgcolor:"rgba(0,0,0,0)",
        font: {{ color: getComputedStyle(document.documentElement).getPropertyValue("--text").trim() }},
      }},
      {{ displayModeBar:false, responsive:true }}
    );

    document.getElementById("barChart").on("plotly_click", (ev) => {{
      const metricLabel = ev.points?.[0]?.x;
      const metric = METRICS.find(m => m.label === metricLabel);
      if (!metric) return;
      activeMetric = metric.key;
      renderDrill(label, activeMetric);
    }});
  }}

  function renderTrendChecks() {{
    els.trendChecks.innerHTML = "";
    const make = (key, label, checked=true) => {{
      const id = "chk_" + key;
      const wrap = document.createElement("label");
      wrap.innerHTML = `<input type="checkbox" id="${{id}}" ${{checked ? "checked" : ""}} /> <span>${{label}}</span>`;
      els.trendChecks.appendChild(wrap);
      wrap.querySelector("input").addEventListener("change", () => renderTrend(els.runSelect.value));
    }};
    make("overall_pct", "Overall", true);
    make("completeness_pct", "Completeness", true);
    make("validity_pct", "Validity", true);
    make("uniqueness_pct", "Uniqueness", true);
    make("timeliness_pct", "Timeliness", true);
  }}
  function isChecked(key) {{
    const el = document.getElementById("chk_" + key);
    return el ? el.checked : false;
  }}
  function renderTrend(label) {{
    const labels = getRunLabels();
    const traces = [];
    const add = (key, name) => {{
      if (!isChecked(key)) return;
      traces.push({{
        type:"scatter", mode:"lines+markers", name,
        x: labels,
        y: SUMMARY.map(r => safeNum(r[key], NaN)),
        hovertemplate: `${{name}}: %{{y:.2f}}%<extra></extra>`
      }});
    }};
    add("overall_pct","Overall");
    add("completeness_pct","Completeness");
    add("validity_pct","Validity");
    add("uniqueness_pct","Uniqueness");
    add("timeliness_pct","Timeliness");

    Plotly.newPlot("trendChart", traces,
      {{
        margin: {{ l:45, r:15, t:10, b:45 }},
        yaxis: {{ range:[0,100], ticksuffix:"%", gridcolor:"rgba(148,163,184,0.25)" }},
        paper_bgcolor:"rgba(0,0,0,0)", plot_bgcolor:"rgba(0,0,0,0)",
        font: {{ color: getComputedStyle(document.documentElement).getPropertyValue("--text").trim() }},
        legend: {{ orientation:"h", y: 1.15 }},
      }},
      {{ displayModeBar:false, responsive:true }}
    );
    els.historyHint.textContent = `Limited history: ${{labels.length}} runs.`;
  }}

  function renderHistoryTable() {{
    const cols = ["run_label","rows_raw","overall_pct","completeness_pct","validity_pct","uniqueness_pct","timeliness_pct"];
    const header = cols.map(c => `<th>${{htmlEscape(c)}}</th>`).join("");
    const rows = SUMMARY.map(r => {{
      const tds = cols.map(c => {{
        const v = r[c];
        if (String(c).endsWith("_pct")) return `<td class="right">${{fmtPct(v)}}</td>`;
        return `<td>${{htmlEscape(v ?? "")}}</td>`;
      }}).join("");
      return `<tr>${{tds}}</tr>`;
    }}).join("");
    els.historyTableWrap.innerHTML = `<table><thead><tr>${{header}}</tr></thead><tbody>${{rows}}</tbody></table>`;
  }}

  function renderDrill(label, metricKey) {{
    const metric = METRICS.find(m => m.key === metricKey) || {{ label:"Overall" }};
    els.drillTitle.textContent = `${{metric.label}} - ${{label}}`;

    const runIssues = issuesForRun(label);
    const filtered = metric.label === "Overall"
      ? runIssues
      : runIssues.filter(x => String(x.dimension || "").toLowerCase() === metric.label.toLowerCase());

    if (filtered.length === 0) {{
      els.drillTableWrap.innerHTML = `<div class="panel-note">No issues found for this selection.</div>`;
      return;
    }}

    const cols = ["dimension","rule","column","failed_count","failed_pct","sample_values"];
    const header = cols.map(c => `<th>${{htmlEscape(c)}}</th>`).join("");
    const rows = filtered.slice(0,25).map(x => {{
      const tds = cols.map(c => {{
        const v = x[c];
        if (c === "failed_pct") return `<td class="right">${{fmtPct(v)}}</td>`;
        if (c === "failed_count") return `<td class="right">${{htmlEscape(v ?? 0)}}</td>`;
        return `<td>${{htmlEscape(v ?? "")}}</td>`;
      }}).join("");
      return `<tr>${{tds}}</tr>`;
    }}).join("");

    els.drillTableWrap.innerHTML = `<table><thead><tr>${{header}}</tr></thead><tbody>${{rows}}</tbody></table><div class="hint">Showing up to 25 rows.</div>`;
  }}

  function attachButtons() {{
    els.btnDark.addEventListener("click", () => {{
      document.documentElement.classList.toggle("dark");
      renderAll(els.runSelect.value);
    }});
    els.btnPrint.addEventListener("click", () => window.print());
    els.btnPng.addEventListener("click", async () => {{
      const png = await Plotly.toImage("barChart", {{ format:"png", height:600, width:900 }});
      const a = document.createElement("a");
      a.href = png;
      a.download = "dq_chart.png";
      a.click();
    }});
    els.btnExport.addEventListener("click", () => {{
      const label = els.runSelect.value;
      const metric = METRICS.find(m => m.key === activeMetric) || {{ label:"Overall" }};
      const runIssues = issuesForRun(label);
      const filtered = metric.label === "Overall"
        ? runIssues
        : runIssues.filter(x => String(x.dimension || "").toLowerCase() === metric.label.toLowerCase());

      if (!filtered.length) {{
        alert("Nothing to export for the current drill-down selection.");
        return;
      }}

      const cols = ["dimension","rule","column","failed_count","failed_pct","sample_values"];
      const lines = [
        cols.join(","),
        ...filtered.map(r => cols.map(c => {{
          const v = r[c] ?? "";
          const s = String(v).replaceAll('"','""');
          return `"${{s}}"`;
        }}).join(","))
      ];

      const blob = new Blob([lines.join("\\n")], {{ type:"text/csv;charset=utf-8" }});
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = `dq_drilldown_${{metric.label.toLowerCase()}}_${{label.replaceAll(" ","_")}}.csv`;
      a.click();
      URL.revokeObjectURL(url);
    }});
    els.btnRun.addEventListener("click", () => alert("This HTML is a report viewer. Run the pipeline in the CLI, then regenerate the report."));
    els.runSelect.addEventListener("change", () => renderAll(els.runSelect.value));
  }}

  function renderAll(label) {{
    renderCards(label);
    renderBar(label);
    renderTrend(label);
    renderDrill(label, activeMetric);
  }}

  // bootstrap
  buildRunSelect();
  renderTrendChecks();
  renderHistoryTable();
  attachButtons();
  renderAll(els.runSelect.value);
</script>
</body>
</html>
"""


def main() -> int:
    _ensure_report_dirs()

    summary = _read_summary()
    issues = _read_issues()

    _write_excel(summary=summary, issues=issues)

    html = _build_html(summary_records=_to_records(summary), issues_records=_to_records(issues))
    HTML_OUT.write_text(html, encoding="utf-8")

    print(f"Wrote: {XLSX_OUT} | exists: {XLSX_OUT.exists()}")
    print(f"Wrote: {HTML_OUT} | exists: {HTML_OUT.exists()}")
    print(f"Summary CSV: {SUMMARY_CSV} | exists: {SUMMARY_CSV.exists()}")
    print(f"Issues CSV:  {ISSUES_CSV} | exists: {ISSUES_CSV.exists()}")
    print(f"Clean CSV:   {CLEAN_CSV} | exists: {CLEAN_CSV.exists()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
