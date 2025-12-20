from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional
import json
import re

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = ROOT / "data"
SYNTH_DIR = DATA_DIR / "synthetic"
OUTPUT_DIR = DATA_DIR / "output"
REPORTS_DIR = ROOT / "reports"
HISTORY_DIR = REPORTS_DIR / "history"

INPUT_CSV = SYNTH_DIR / "input.csv"
RAW_CSV = OUTPUT_DIR / "raw.csv"
CLEAN_CSV = OUTPUT_DIR / "clean.csv"

SUMMARY_CSV = REPORTS_DIR / "dq_summary.csv"
ISSUES_CSV = REPORTS_DIR / "dq_issues.csv"


@dataclass(frozen=True)
class Rules:
    version: str = "1.0.0"
    primary_key: str = "order_id"
    required_columns: tuple[str, ...] = ("order_id", "email", "country", "amount", "order_ts_utc", "ingest_ts_utc")
    amount_min: float = 0.0
    max_delay_minutes: int = 12 * 60
    email_regex: str = r"^[^@\s]+@[^@\s]+\.[^@\s]+$"


def _ensure_dirs() -> None:
    SYNTH_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    HISTORY_DIR.mkdir(parents=True, exist_ok=True)


def _run_id(now_utc: datetime) -> str:
    return now_utc.strftime("%Y%m%d_%H%M%S")


def _load_input() -> pd.DataFrame:
    if not INPUT_CSV.exists():
        raise FileNotFoundError(f"Missing dataset: {INPUT_CSV}. Run `dqhub synth` first.")

    df = pd.read_csv(INPUT_CSV)
    df.columns = [c.strip() for c in df.columns]

    # Parse timestamps
    for col in ["order_ts_utc", "ingest_ts_utc"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce", utc=True)

    # Normalize types
    if "order_id" in df.columns:
        df["order_id"] = pd.to_numeric(df["order_id"], errors="coerce").astype("Int64")
    if "amount" in df.columns:
        df["amount"] = pd.to_numeric(df["amount"], errors="coerce")

    # Strip strings
    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = df[col].astype("string").str.strip()

    return df


def _read_summary() -> pd.DataFrame:
    if not SUMMARY_CSV.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(SUMMARY_CSV)
    except Exception:
        return pd.DataFrame()


def _next_run_label(summary: pd.DataFrame) -> str:
    return f"Run {len(summary) + 1}"


def _append_summary_row(row: dict[str, Any]) -> None:
    hist = _read_summary()
    row_df = pd.DataFrame([row])

    if hist.empty:
        out = row_df
    else:
        # align columns
        for c in hist.columns:
            if c not in row_df.columns:
                row_df[c] = pd.NA
        for c in row_df.columns:
            if c not in hist.columns:
                hist[c] = pd.NA
        out = pd.concat([hist, row_df[hist.columns]], ignore_index=True)

    out.to_csv(SUMMARY_CSV, index=False, encoding="utf-8")


def _issue_row(
    *,
    dimension: str,
    rule: str,
    column: str,
    failed_mask: pd.Series,
    df: pd.DataFrame,
    run_label: str,
    run_id: str,
) -> Optional[dict[str, Any]]:
    failed_count = int(failed_mask.sum())
    if failed_count <= 0:
        return None

    rows = int(df.shape[0])
    failed_pct = round(100.0 * failed_count / rows, 2) if rows else 0.0

    sample_vals = []
    try:
        sample_vals = df.loc[failed_mask, column].head(5).tolist() if column in df.columns else []
    except Exception:
        sample_vals = []

    return {
        "run_label": run_label,
        "run_id": run_id,
        "dimension": dimension,
        "rule": rule,
        "column": column,
        "failed_count": failed_count,
        "failed_pct": failed_pct,
        "sample_values": json.dumps(sample_vals),
    }


def main() -> None:
    _ensure_dirs()
    rules = Rules()

    now_utc = datetime.now(timezone.utc)
    rid = _run_id(now_utc)

    summary_hist = _read_summary()
    run_label = _next_run_label(summary_hist)

    df_raw = _load_input()
    rows_raw = int(df_raw.shape[0])

    df_raw.to_csv(RAW_CSV, index=False, encoding="utf-8")
    print(f"Wrote: {RAW_CSV} | exists: {RAW_CSV.exists()}")

    # Build failure masks on RAW (after parsing)
    missing_required = pd.Series([False] * rows_raw)
    for col in rules.required_columns:
        if col not in df_raw.columns:
            missing_required = pd.Series([True] * rows_raw)
            break
        missing_required = missing_required | df_raw[col].isna()

    email_bad = pd.Series([False] * rows_raw)
    if "email" in df_raw.columns:
        rx = re.compile(rules.email_regex)
        email_bad = df_raw["email"].isna() | ~df_raw["email"].astype("string").fillna("").str.match(rx)

    amount_bad = pd.Series([False] * rows_raw)
    if "amount" in df_raw.columns:
        amount_bad = df_raw["amount"].isna() | (df_raw["amount"] < rules.amount_min)

    time_bad = pd.Series([False] * rows_raw)
    if "order_ts_utc" in df_raw.columns and "ingest_ts_utc" in df_raw.columns:
        delay_min = (df_raw["ingest_ts_utc"] - df_raw["order_ts_utc"]).dt.total_seconds() / 60.0
        time_bad = delay_min.isna() | (delay_min < 0) | (delay_min > rules.max_delay_minutes)

    invalid_any = missing_required | email_bad | amount_bad | time_bad
    df_clean = df_raw.loc[~invalid_any].copy()

    df_clean.to_csv(CLEAN_CSV, index=False, encoding="utf-8")
    print(f"Clean CSV: {CLEAN_CSV} | exists: {CLEAN_CSV.exists()}")

    rows_after_hard_clean = int(df_clean.shape[0])


    completeness_pct = 100.0
    validity_pct = 100.0
    timeliness_pct = 100.0

    uniqueness_pct = 0.0
    if rules.primary_key in df_raw.columns and rows_raw > 0:
        unique_count = int(df_raw[rules.primary_key].nunique(dropna=True))
        uniqueness_pct = round(100.0 * unique_count / rows_raw, 2)

    overall_pct = round(100.0 * rows_after_hard_clean / rows_raw, 2) if rows_raw else 0.0

    issues: list[dict[str, Any]] = []
    maybe = _issue_row(
        dimension="Completeness",
        rule="required_columns_present",
        column="(required)",
        failed_mask=missing_required,
        df=df_raw.assign(**{"(required)": ""}),
        run_label=run_label,
        run_id=rid,
    )
    if maybe:
        issues.append(maybe)

    maybe = _issue_row(
        dimension="Validity",
        rule="email_format",
        column="email",
        failed_mask=email_bad,
        df=df_raw,
        run_label=run_label,
        run_id=rid,
    )
    if maybe:
        issues.append(maybe)

    maybe = _issue_row(
        dimension="Validity",
        rule=f"min_amount:{rules.amount_min}",
        column="amount",
        failed_mask=amount_bad,
        df=df_raw,
        run_label=run_label,
        run_id=rid,
    )
    if maybe:
        issues.append(maybe)

    maybe = _issue_row(
        dimension="Timeliness",
        rule=f"max_delay_minutes:{rules.max_delay_minutes}",
        column="ingest_ts_utc",
        failed_mask=time_bad,
        df=df_raw,
        run_label=run_label,
        run_id=rid,
    )
    if maybe:
        issues.append(maybe)

    df_issues = pd.DataFrame(
        issues,
        columns=["run_label", "run_id", "dimension", "rule", "column", "failed_count", "failed_pct", "sample_values"],
    )
    df_issues.to_csv(ISSUES_CSV, index=False, encoding="utf-8")
    print(f"Issues CSV: {ISSUES_CSV} | exists: {ISSUES_CSV.exists()}")

    summary_row = {
        "run_label": run_label,
        "run_id": rid,
        "run_ts_utc": now_utc.isoformat(),
        "rows_raw": rows_raw,
        "rows_after_hard_clean": rows_after_hard_clean,
        "overall_pct": overall_pct,
        "completeness_pct": round(completeness_pct, 2),
        "validity_pct": round(validity_pct, 2),
        "uniqueness_pct": round(uniqueness_pct, 2),
        "timeliness_pct": round(timeliness_pct, 2),
        "rules_version": rules.version,
    }
    _append_summary_row(summary_row)
    print(f"Run History CSV: {SUMMARY_CSV} | exists: {SUMMARY_CSV.exists()}")

    run_dir = HISTORY_DIR / rid
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "raw.csv").write_bytes(RAW_CSV.read_bytes())
    (run_dir / "clean.csv").write_bytes(CLEAN_CSV.read_bytes())
    (run_dir / "dq_issues.csv").write_bytes(ISSUES_CSV.read_bytes())


if __name__ == "__main__":
    main()
