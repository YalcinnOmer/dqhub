from __future__ import annotations
from pathlib import Path
from datetime import datetime, timezone
import re, json, yaml
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio

ROOT    = Path(__file__).resolve().parents[1]
DATA    = ROOT / "data"
CLEAN   = DATA / "output" / "clean.csv"
RULES   = ROOT / "rules" / "rules.yaml"
REPORTS = ROOT / "reports"
HIST    = REPORTS / "history"
HTML    = REPORTS / "DQ_Report.html"

EMAIL_RE = re.compile(r"^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}$")
PHONE_RE = re.compile(r"^\+1\d{10}$")

def _safe_pct(n_ok: int, n_all: int) -> float:
    return 100.0 * (n_ok / n_all) if n_all else 100.0

def load_rules() -> dict:
    return yaml.safe_load(RULES.read_text(encoding="utf-8")) or {}

def compute_metrics(df: pd.DataFrame, rules: dict) -> dict:
    cols_cfg   = rules.get("columns", [])
    schema_pk  = rules.get("schema", {}).get("primary_key", [])
    required   = [c["name"] for c in cols_cfg if c.get("required", False)]
    date_rules = [c for c in cols_cfg if c.get("type") == "date" and (c.get("min") or c.get("max"))]

    n = len(df)

    # Completeness (required columns)
    req_ok = 0
    if required:
        mask = pd.Series(True, index=df.index)
        for c in required:
            if c not in df.columns:
                mask &= False
            else:
                mask &= ~(df[c].isna() | (df[c].astype(str).str.strip() == ""))
        req_ok = int(mask.sum())
    completeness = _safe_pct(req_ok, n)

    # Validity (email + phone regex)
    valid_ok = n
    if "email" in df.columns:
        valid_email = df["email"].astype(str).str.fullmatch(EMAIL_RE).fillna(False)
        valid_ok = int(valid_email.sum())
    if "phone" in df.columns:
        valid_phone = df["phone"].astype(str).str.fullmatch(PHONE_RE).fillna(False)
        valid_ok = min(valid_ok, int(valid_phone.sum()))
    validity = _safe_pct(valid_ok, n)

    # Uniqueness (primary key)
    uniqueness = 100.0
    if schema_pk:
        tmp = df.copy()
        dup = tmp.duplicated(subset=schema_pk, keep=False)
        uniqueness = _safe_pct(int((~dup).sum()), n)

    # Timeliness (date min/max compliance)
    time_scores = []
    for c in date_rules:
        name = c["name"]
        if name in df.columns:
            ser = pd.to_datetime(df[name], errors="coerce", utc=True)
            ok_mask = ser.notna()
            if c.get("min"):
                ok_mask &= ser >= pd.to_datetime(c["min"], utc=True)
            if c.get("max"):
                ok_mask &= ser <= pd.to_datetime(c["max"], utc=True)
            time_scores.append(_safe_pct(int(ok_mask.sum()), n))
    timeliness = sum(time_scores)/len(time_scores) if time_scores else 100.0

    # Weighted overall DQ score
    weights = {"completeness": 0.40, "validity": 0.30, "uniqueness": 0.20, "timeliness": 0.10}
    overall = (
        completeness * weights["completeness"] +
        validity     * weights["validity"] +
        uniqueness   * weights["uniqueness"] +
        timeliness   * weights["timeliness"]
    )

    return {
        "rows": n,
        "completeness": round(completeness, 2),
        "validity": round(validity, 2),
        "uniqueness": round(uniqueness, 2),
        "timeliness": round(timeliness, 2),
        "overall": round(overall, 2),
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "rules_version": (rules.get("meta") or {}).get("version", "1.0.0"),
    }

def append_history(metrics: dict) -> pd.DataFrame:
    HIST.mkdir(parents=True, exist_ok=True)
    hist_csv = HIST / "metrics_history.csv"
    row = {
        "ts": metrics["generated_at"],
        "overall": metrics["overall"],
        "completeness": metrics["completeness"],
        "validity": metrics["validity"],
        "uniqueness": metrics["uniqueness"],
        "timeliness": metrics["timeliness"],
        "rows": metrics["rows"],
        "rules_version": metrics["rules_version"],
    }
    if hist_csv.exists():
        dfh = pd.read_csv(hist_csv)
        dfh = pd.concat([dfh, pd.DataFrame([row])], ignore_index=True)
    else:
        dfh = pd.DataFrame([row])
    dfh.to_csv(hist_csv, index=False)
    return dfh

def build_html(metrics: dict, history: pd.DataFrame) -> str:
    # Bar (current run)
    bar = go.Figure()
    bar.add_trace(go.Bar(
        x=["Completeness","Validity","Uniqueness","Timeliness","Overall"],
        y=[metrics["completeness"], metrics["validity"], metrics["uniqueness"], metrics["timeliness"], metrics["overall"]],
        text=[f'{v}%' for v in [metrics["completeness"], metrics["validity"], metrics["uniqueness"], metrics["timeliness"], metrics["overall"]]],
        textposition="auto"
    ))
    bar.update_layout(title="DQ Score (current run)", yaxis_title="%")

    # Trend (history)
    if not history.empty:
        trend = go.Figure()
        trend.add_trace(go.Scatter(x=history["ts"], y=history["overall"], mode="lines+markers", name="Overall"))
        trend.update_layout(title="Overall DQ Trend", xaxis_title="Run", yaxis_title="%")
        trend_html = pio.to_html(trend, full_html=False, include_plotlyjs=False)
    else:
        trend_html = "<p>No history yet.</p>"

    bar_html = pio.to_html(bar, full_html=False, include_plotlyjs="cdn")

    html = f"""<!DOCTYPE html>
<html>
<head><meta charset="utf-8"><title>DQ Report</title></head>
<body style="font-family:system-ui,Segoe UI,Arial;">
<h1>Data Quality Report</h1>
<p><b>Generated at:</b> {metrics["generated_at"]} &nbsp;|&nbsp; <b>Rows:</b> {metrics["rows"]} &nbsp;|&nbsp; <b>Rules:</b> {metrics["rules_version"]}</p>
<div>{bar_html}</div>
<hr/>
<h2>Trend</h2>
<div>{trend_html}</div>
</body>
</html>"""
    return html

def main() -> None:
    if not CLEAN.exists():
        raise SystemExit(f"Missing clean csv: {CLEAN}")
    if not RULES.exists():
        raise SystemExit(f"Missing rules: {RULES}")

    df     = pd.read_csv(CLEAN)
    rules  = load_rules()
    REPORTS.mkdir(parents=True, exist_ok=True)

    metrics = compute_metrics(df, rules)
    history = append_history(metrics)
    html    = build_html(metrics, history)
    HTML.write_text(html, encoding="utf-8")

    (REPORTS / "last_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"Wrote: {HTML} | exists: {HTML.exists()}")

if __name__ == "__main__":
    main()
