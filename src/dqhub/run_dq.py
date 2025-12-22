from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable
import json
import re
from shutil import copy2

import pandas as pd

try:
    import yaml
except Exception:
    yaml = None


ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
OUTPUT_DIR = DATA_DIR / "output"
REPORTS_DIR = ROOT / "reports"
RULES_YAML = ROOT / "rules" / "rules.yaml"

RAW_CSV = OUTPUT_DIR / "raw.csv"
CLEAN_CSV = OUTPUT_DIR / "clean.csv"

SUMMARY_CSV = REPORTS_DIR / "dq_summary.csv"
ISSUES_CSV = REPORTS_DIR / "dq_issues.csv"

LAST_RUN_JSON = REPORTS_DIR / "last_run.json"
LAST_METRICS_JSON = REPORTS_DIR / "last_metrics.json"


@dataclass(frozen=True)
class Rules:
    version: str = "1.0.0"
    primary_key: str = "order_id"
    required_columns: tuple[str, ...] = (
        "order_id",
        "email",
        "country",
        "amount",
        "order_ts_utc",
        "ingest_ts_utc",
    )
    amount_min: float = 0.0
    max_delay_minutes: int = 12 * 60
    email_regex: str = r"^[^@\s]+@[^@\s]+\.[^@\s]+$"


def _ensure_dirs() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def _run_id(now_utc: datetime) -> str:
    return now_utc.strftime("%Y%m%d_%H%M%S")


def _load_rules() -> Rules:
    if not RULES_YAML.exists():
        return Rules()

    if yaml is None:
        raise RuntimeError(
            "PyYAML is required because rules/rules.yaml exists, but PyYAML is not installed."
        )

    data = yaml.safe_load(RULES_YAML.read_text(encoding="utf-8")) or {}
    return Rules(
        version=str(data.get("version", "1.0.0")),
        primary_key=str(data.get("primary_key", "order_id")),
        required_columns=tuple(data.get("required_columns", Rules.required_columns)),
        amount_min=float(data.get("amount_min", 0.0)),
        max_delay_minutes=int(data.get("max_delay_minutes", 12 * 60)),
        email_regex=str(data.get("email_regex", Rules.email_regex)),
    )


def _next_run_label(summary_hist: pd.DataFrame) -> str:
    if summary_hist is None or summary_hist.empty:
        return "Run 1"
    last = str(summary_hist["run_label"].iloc[-1]) if "run_label" in summary_hist.columns else ""
    m = re.search(r"(\d+)", last)
    if not m:
        return f"Run {len(summary_hist) + 1}"
    return f"Run {int(m.group(1)) + 1}"


def _to_jsonable(x: Any) -> Any:
    if x is pd.NaT:
        return None

    if isinstance(x, pd.Timestamp):
        if pd.isna(x):
            return None
        try:
            return x.isoformat()
        except Exception:
            return str(x)

    if isinstance(x, datetime):
        try:
            return x.isoformat()
        except Exception:
            return str(x)

    try:
        import numpy as np  # type: ignore

        if isinstance(x, (np.integer,)):
            return int(x)
        if isinstance(x, (np.floating,)):
            return None if np.isnan(x) else float(x)
        if isinstance(x, (np.bool_,)):
            return bool(x)
    except Exception:
        pass

    try:
        if pd.isna(x):
            return None
    except Exception:
        pass

    if isinstance(x, dict):
        return {str(k): _to_jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple, set)):
        return [_to_jsonable(v) for v in x]

    return x


def _json_dumps_safe(obj: Any) -> str:
    return json.dumps(_to_jsonable(obj), ensure_ascii=False)


def _read_csv(path: Path) -> pd.DataFrame:
    if path.exists():
        return pd.read_csv(path)

    # If raw.csv is missing but synth exists, copy it for a smoother first-run experience.
    if path.name == "raw.csv":
        fallback = path.parents[1] / "synthetic" / "input.csv"
        if fallback.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            copy2(fallback, path)
            return pd.read_csv(path)

    raise FileNotFoundError(f"Missing required file: {path}")


def _coerce_datetime(df: pd.DataFrame, col: str) -> None:
    if col not in df.columns:
        return
    df[col] = pd.to_datetime(df[col], utc=True, errors="coerce")


def _normalize_strings(df: pd.DataFrame, cols: Iterable[str]) -> None:
    for c in cols:
        if c in df.columns:
            df[c] = df[c].astype("string").str.strip()


def _missing_required_mask(df: pd.DataFrame, required_cols: tuple[str, ...]) -> pd.Series:
    masks: list[pd.Series] = []
    for c in required_cols:
        if c not in df.columns:
            return pd.Series([True] * len(df), index=df.index)
        s = df[c]
        if pd.api.types.is_string_dtype(s) or str(s.dtype).startswith("string"):
            masks.append(s.isna() | (s.astype("string").str.len() == 0))
        else:
            masks.append(s.isna())

    out = masks[0].copy()
    for m in masks[1:]:
        out = out | m
    return out


def _issue_row(
    *,
    run_label: str,
    run_id: str,
    df: pd.DataFrame,
    dimension: str,
    rule: str,
    column: str,
    failed_mask: pd.Series,
) -> dict[str, Any] | None:
    rows = int(df.shape[0])
    failed_count = int(failed_mask.sum())
    if failed_count <= 0 or rows <= 0:
        return None

    failed_pct = round((failed_count / rows) * 100.0, 2)

    try:
        if column in df.columns:
            sample_vals = df.loc[failed_mask, column].head(5).tolist()
        else:
            sample_vals = [""] * min(3, failed_count)
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
        "sample_values": _json_dumps_safe(sample_vals),
    }


def main() -> int:
    _ensure_dirs()
    rules = _load_rules()

    now_utc = datetime.now(timezone.utc).replace(microsecond=0)
    rid = _run_id(now_utc)

    df_raw = _read_csv(RAW_CSV)
    _normalize_strings(df_raw, [c for c in ["email", "country"] if c in df_raw.columns])

    order_col = "order_ts_utc" if "order_ts_utc" in df_raw.columns else "order_ts"
    ingest_col = "ingest_ts_utc" if "ingest_ts_utc" in df_raw.columns else "ingest_ts"

    _coerce_datetime(df_raw, order_col)
    _coerce_datetime(df_raw, ingest_col)

    summary_hist = pd.read_csv(SUMMARY_CSV) if SUMMARY_CSV.exists() else pd.DataFrame()
    run_label = _next_run_label(summary_hist)

    rows_raw = int(df_raw.shape[0])

    # Hard clean: drop rows missing required columns
    missing_required = _missing_required_mask(df_raw, rules.required_columns)
    df_clean = df_raw.loc[~missing_required].copy()

    # normalize expected timestamp column names for downstream consumers
    if order_col != "order_ts_utc" and order_col in df_clean.columns:
        df_clean.rename(columns={order_col: "order_ts_utc"}, inplace=True)
    if ingest_col != "ingest_ts_utc" and ingest_col in df_clean.columns:
        df_clean.rename(columns={ingest_col: "ingest_ts_utc"}, inplace=True)

    CLEAN_CSV.parent.mkdir(parents=True, exist_ok=True)
    df_clean.to_csv(CLEAN_CSV, index=False, encoding="utf-8")
    print(f"Clean CSV: {CLEAN_CSV} | exists: {CLEAN_CSV.exists()}")

    rows_after_global_fixes = rows_raw
    rows_after_hard_clean = int(df_clean.shape[0])

    completeness_bad = missing_required
    completeness_pct = round((1.0 - completeness_bad.mean()) * 100.0, 2) if rows_raw else 0.0

    amount_bad = pd.Series([False] * rows_raw, index=df_raw.index)
    if "amount" in df_raw.columns:
        amt = pd.to_numeric(df_raw["amount"], errors="coerce")
        amount_bad = amt.isna() | (amt < float(rules.amount_min))

    email_bad = pd.Series([False] * rows_raw, index=df_raw.index)
    if "email" in df_raw.columns:
        email_s = df_raw["email"].astype("string")
        email_bad = ~email_s.fillna("").str.match(rules.email_regex)

    invalid_any = amount_bad | email_bad
    validity_pct = round((1.0 - invalid_any.mean()) * 100.0, 2) if rows_raw else 0.0

    dup_mask = pd.Series([False] * rows_raw, index=df_raw.index)
    if rules.primary_key in df_raw.columns:
        dup_mask = df_raw[rules.primary_key].duplicated(keep=False)
    uniqueness_pct = round((1.0 - dup_mask.mean()) * 100.0, 2) if rows_raw else 0.0

    time_bad = pd.Series([False] * rows_raw, index=df_raw.index)
    timeliness_pct = 0.0
    if order_col in df_raw.columns and ingest_col in df_raw.columns:
        order_ts = df_raw[order_col]
        ingest_ts = df_raw[ingest_col]

        time_bad = order_ts.isna() | ingest_ts.isna()
        delay_min = (ingest_ts - order_ts).dt.total_seconds() / 60.0
        delay_bad = delay_min.isna() | (delay_min < 0) | (delay_min > rules.max_delay_minutes)
        time_bad = time_bad | delay_bad

        timeliness_pct = round((1.0 - time_bad.mean()) * 100.0, 2) if rows_raw else 0.0

    overall_pct = round(
        (completeness_pct + validity_pct + uniqueness_pct + timeliness_pct) / 4.0, 2
    )

    issues: list[dict[str, Any]] = []
    for maybe in [
        _issue_row(
            run_label=run_label,
            run_id=rid,
            df=df_raw,
            dimension="Completeness",
            rule="required_columns_present",
            column="(required)",
            failed_mask=completeness_bad,
        ),
        _issue_row(
            run_label=run_label,
            run_id=rid,
            df=df_raw,
            dimension="Validity",
            rule=f"min_amount:{rules.amount_min}",
            column="amount",
            failed_mask=amount_bad,
        ),
        _issue_row(
            run_label=run_label,
            run_id=rid,
            df=df_raw,
            dimension="Validity",
            rule="email_format",
            column="email",
            failed_mask=email_bad,
        ),
        _issue_row(
            run_label=run_label,
            run_id=rid,
            df=df_raw,
            dimension="Uniqueness",
            rule=f"primary_key:{rules.primary_key}",
            column=rules.primary_key,
            failed_mask=dup_mask,
        ),
        _issue_row(
            run_label=run_label,
            run_id=rid,
            df=df_raw,
            dimension="Timeliness",
            rule=f"max_delay_minutes:{rules.max_delay_minutes}",
            column="ingest_ts_utc" if "ingest_ts_utc" in df_raw.columns else ingest_col,
            failed_mask=time_bad,
        ),
    ]:
        if maybe:
            issues.append(maybe)

    summary_row = {
        "run_label": run_label,
        "run_id": rid,
        "run_ts_utc": now_utc.isoformat(),
        "rows_raw": rows_raw,
        "rows_after_global_fixes": rows_after_global_fixes,
        "rows_after_hard_clean": rows_after_hard_clean,
        "completeness_pct": completeness_pct,
        "validity_pct": validity_pct,
        "uniqueness_pct": uniqueness_pct,
        "timeliness_pct": timeliness_pct,
        "overall_pct": overall_pct,
        "rules_version": rules.version,
    }

    summary_out = pd.DataFrame([summary_row])
    if SUMMARY_CSV.exists():
        hist = pd.read_csv(SUMMARY_CSV)
        summary_out = pd.concat([hist, summary_out], ignore_index=True)

    summary_out.to_csv(SUMMARY_CSV, index=False, encoding="utf-8")

    issues_out = pd.DataFrame(issues)
    if ISSUES_CSV.exists():
        hist_i = pd.read_csv(ISSUES_CSV)
        issues_out = pd.concat([hist_i, issues_out], ignore_index=True)

    issues_out.to_csv(ISSUES_CSV, index=False, encoding="utf-8")

    LAST_RUN_JSON.write_text(
        _json_dumps_safe({"run_label": run_label, "run_id": rid, "run_ts_utc": now_utc.isoformat()}),
        encoding="utf-8",
    )
    LAST_METRICS_JSON.write_text(_json_dumps_safe(summary_row), encoding="utf-8")

    print(f"Wrote: {SUMMARY_CSV} | exists: {SUMMARY_CSV.exists()}")
    print(f"Wrote: {ISSUES_CSV} | exists: {ISSUES_CSV.exists()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
