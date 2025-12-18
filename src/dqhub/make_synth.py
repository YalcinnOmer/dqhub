from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]

RULES_YAML = ROOT / "rules" / "rules.yaml"

DATA_SYNTH_INPUT = ROOT / "data" / "synthetic" / "input.csv"
DATA_RAW_FALLBACK = ROOT / "data" / "output" / "raw.csv"
DATA_OUTPUT_DIR = ROOT / "data" / "output"
CLEAN_CSV = DATA_OUTPUT_DIR / "clean.csv"

REPORTS_DIR = ROOT / "reports"
DQ_SUMMARY_CSV = REPORTS_DIR / "dq_summary.csv"
DQ_ISSUES_CSV = REPORTS_DIR / "dq_issues.csv"
HISTORY_DIR = REPORTS_DIR / "history"

THRESH_GOOD = 99.0
THRESH_WARN = 97.0

_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _run_id() -> str:
    return _utc_now().strftime("%Y%m%d_%H%M%S")


def _iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).isoformat()


def _ensure_dirs() -> None:
    DATA_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    HISTORY_DIR.mkdir(parents=True, exist_ok=True)


def _safe_float(x: Any) -> float | None:
    try:
        if x is None:
            return None
        v = float(x)
        if pd.isna(v):
            return None
        return v
    except Exception:
        return None


def _safe_int(x: Any) -> int | None:
    try:
        if x is None:
            return None
        v = int(float(x))
        return v
    except Exception:
        return None


def _pct(x: float | None) -> float | None:
    if x is None:
        return None
    return round(float(x), 4)


def _read_rules_version() -> str | None:
    if not RULES_YAML.exists():
        return None
    try:
        import yaml  # type: ignore

        data = yaml.safe_load(RULES_YAML.read_text(encoding="utf-8")) or {}
        v = data.get("version")
        if isinstance(v, str) and v.strip():
            return v.strip()
        return None
    except Exception:
        return None


def _choose_input_path() -> Path:
    if DATA_SYNTH_INPUT.exists():
        return DATA_SYNTH_INPUT
    if DATA_RAW_FALLBACK.exists():
        return DATA_RAW_FALLBACK
    raise FileNotFoundError(f"Missing input file: {DATA_SYNTH_INPUT} (also missing {DATA_RAW_FALLBACK})")


def _standardize_strings(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    for c in out.columns:
        if out[c].dtype == object:
            out[c] = out[c].astype(str).str.strip()

    if "customer_email" in out.columns:
        out["customer_email"] = out["customer_email"].astype(str).str.lower().str.strip()

    if "country" in out.columns:
        out["country"] = out["country"].astype(str).str.upper().str.strip()

    if "currency" in out.columns:
        out["currency"] = out["currency"].astype(str).str.upper().str.strip()

    return out


def _parse_timestamps(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "order_ts" in out.columns:
        out["order_ts"] = pd.to_datetime(out["order_ts"], utc=True, errors="coerce")
    if "ingested_ts" in out.columns:
        out["ingested_ts"] = pd.to_datetime(out["ingested_ts"], utc=True, errors="coerce")
    return out


def _hard_clean(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    required = [c for c in ["id", "order_ts", "ingested_ts"] if c in out.columns]
    if required:
        out = out.dropna(subset=required)

    if "total_amount" in out.columns:
        out["total_amount"] = pd.to_numeric(out["total_amount"], errors="coerce")
        out = out.dropna(subset=["total_amount"])
        out = out[out["total_amount"] >= 0]

    return out.reset_index(drop=True)


@dataclass(frozen=True)
class IssueRow:
    run_id: str
    run_ts_utc: str
    dimension: str
    rule: str
    column: str
    failed_count: int
    failed_pct: float
    sample_values: str


def _issue(
    run_id: str,
    run_ts: str,
    dimension: str,
    rule: str,
    column: str,
    failed_mask: pd.Series,
    values: Iterable[Any],
) -> IssueRow | None:
    total = int(len(failed_mask))
    failed_count = int(failed_mask.sum())
    if total <= 0 or failed_count <= 0:
        return None

    failed_pct = 100.0 * (failed_count / total)
    samples = []
    for v in values:
        if v is None:
            continue
        s = str(v)
        if s not in samples:
            samples.append(s)
        if len(samples) >= 10:
            break

    return IssueRow(
        run_id=run_id,
        run_ts_utc=run_ts,
        dimension=dimension,
        rule=rule,
        column=column,
        failed_count=failed_count,
        failed_pct=round(failed_pct, 2),
        sample_values=json.dumps(samples),
    )


def _completeness(df: pd.DataFrame) -> float | None:
    if df.empty:
        return None
    total_cells = int(df.shape[0] * df.shape[1])
    if total_cells <= 0:
        return None
    missing = int(df.isna().sum().sum())
    score = 100.0 * (1.0 - (missing / total_cells))
    return max(0.0, min(100.0, score))


def _validity(df: pd.DataFrame, run_id: str, run_ts: str) -> tuple[float | None, list[IssueRow]]:
    if df.empty:
        return None, []
    issues: list[IssueRow] = []

    total = len(df)

    # Email format
    if "customer_email" in df.columns:
        s = df["customer_email"].astype(str).fillna("")
        bad = ~s.str.match(_EMAIL_RE)
        row = _issue(
            run_id=run_id,
            run_ts=run_ts,
            dimension="Validity",
            rule="email_format",
            column="customer_email",
            failed_mask=bad,
            values=s[bad].head(10).tolist(),
        )
        if row:
            issues.append(row)

    # Country code (2-letter)
    if "country" in df.columns:
        s = df["country"].astype(str).fillna("")
        bad = ~s.str.match(r"^[A-Z]{2}$")
        row = _issue(
            run_id=run_id,
            run_ts=run_ts,
            dimension="Validity",
            rule="country_code_alpha2",
            column="country",
            failed_mask=bad,
            values=s[bad].head(10).tolist(),
        )
        if row:
            issues.append(row)

    # Currency code (3-letter)
    if "currency" in df.columns:
        s = df["currency"].astype(str).fillna("")
        bad = ~s.str.match(r"^[A-Z]{3}$")
        row = _issue(
            run_id=run_id,
            run_ts=run_ts,
            dimension="Validity",
            rule="currency_code_alpha3",
            column="currency",
            failed_mask=bad,
            values=s[bad].head(10).tolist(),
        )
        if row:
            issues.append(row)

    # Amount non-negative numeric
    if "total_amount" in df.columns:
        s = pd.to_numeric(df["total_amount"], errors="coerce")
        bad = s.isna() | (s < 0)
        row = _issue(
            run_id=run_id,
            run_ts=run_ts,
            dimension="Validity",
            rule="amount_non_negative",
            column="total_amount",
            failed_mask=bad,
            values=s[bad].head(10).tolist(),
        )
        if row:
            issues.append(row)

    # Validity score: percent of rows passing all implemented checks
    if not issues:
        return 100.0, []

    # Conservative: compute row-level pass rate across checks we ran
    row_bad = pd.Series(False, index=df.index)
    for it in issues:
        if it.rule == "email_format" and "customer_email" in df.columns:
            row_bad |= ~df["customer_email"].astype(str).fillna("").str.match(_EMAIL_RE)
        if it.rule == "country_code_alpha2" and "country" in df.columns:
            row_bad |= ~df["country"].astype(str).fillna("").str.match(r"^[A-Z]{2}$")
        if it.rule == "currency_code_alpha3" and "currency" in df.columns:
            row_bad |= ~df["currency"].astype(str).fillna("").str.match(r"^[A-Z]{3}$")
        if it.rule == "amount_non_negative" and "total_amount" in df.columns:
            s = pd.to_numeric(df["total_amount"], errors="coerce")
            row_bad |= s.isna() | (s < 0)

    bad_count = int(row_bad.sum())
    score = 100.0 * (1.0 - (bad_count / (total if total else 1)))
    score = max(0.0, min(100.0, score))
    return score, issues


def _uniqueness(df: pd.DataFrame, run_id: str, run_ts: str) -> tuple[float | None, list[IssueRow]]:
    if df.empty:
        return None, []

    # Prefer "id", otherwise any *_id
    key = None
    if "id" in df.columns:
        key = "id"
    else:
        for c in df.columns:
            if isinstance(c, str) and c.lower().endswith("_id"):
                key = c
                break

    if key is None:
        return None, []

    dup_mask = df.duplicated(subset=[key], keep="first")
    dup_count = int(dup_mask.sum())
    total = int(len(df))

    if total <= 0:
        return None, []

    score = 100.0 * (1.0 - (dup_count / total))
    score = max(0.0, min(100.0, score))

    issues: list[IssueRow] = []
    if dup_count > 0:
        examples = df.loc[df.duplicated(subset=[key], keep=False), key].astype(str).head(10).tolist()

        row1 = _issue(
            run_id=run_id,
            run_ts=run_ts,
            dimension="Uniqueness",
            rule=f"primary_key:{key}",
            column=key,
            failed_mask=dup_mask,
            values=examples,
        )
        if row1:
            issues.append(row1)

        row2 = _issue(
            run_id=run_id,
            run_ts=run_ts,
            dimension="Uniqueness",
            rule=f"unique:{key}",
            column=key,
            failed_mask=dup_mask,
            values=examples,
        )
        if row2:
            issues.append(row2)

    return score, issues


def _timeliness(df: pd.DataFrame, run_id: str, run_ts: str) -> tuple[float | None, list[IssueRow]]:
    if df.empty:
        return None, []

    if "order_ts" not in df.columns or "ingested_ts" not in df.columns:
        return None, []

    o = pd.to_datetime(df["order_ts"], utc=True, errors="coerce")
    i = pd.to_datetime(df["ingested_ts"], utc=True, errors="coerce")
    ok = o.notna() & i.notna()

    if int(ok.sum()) <= 0:
        return None, []

    delay_h = (i[ok] - o[ok]).dt.total_seconds() / 3600.0
    late = delay_h > 24.0

    total = int(len(delay_h))
    late_count = int(late.sum())

    score = 100.0 * (1.0 - (late_count / (total if total else 1)))
    score = max(0.0, min(100.0, score))

    issues: list[IssueRow] = []
    if late_count > 0:
        examples = delay_h[late].head(10).round(2).astype(str).tolist()
        row = _issue(
            run_id=run_id,
            run_ts=run_ts,
            dimension="Timeliness",
            rule="late_over_24h",
            column="ingested_ts",
            failed_mask=late.reset_index(drop=True),
            values=examples,
        )
        if row:
            issues.append(row)

    return score, issues


def _overall(scores: list[float | None]) -> float | None:
    vals = [float(s) for s in scores if s is not None]
    if not vals:
        return None
    return sum(vals) / len(vals)


def _append_csv(path: Path, df: pd.DataFrame) -> None:
    if path.exists():
        try:
            old = pd.read_csv(path)
            out = pd.concat([old, df], ignore_index=True)
            out.to_csv(path, index=False)
            return
        except Exception:
            pass
    df.to_csv(path, index=False)


def run() -> None:
    _ensure_dirs()

    run_id = _run_id()
    run_ts = _iso(_utc_now())
    rules_version = _read_rules_version()

    input_path = _choose_input_path()

    df_raw = pd.read_csv(input_path)
    rows_raw = int(len(df_raw))

    df_fixed = _standardize_strings(df_raw)
    df_fixed = _parse_timestamps(df_fixed)
    rows_after_global_fixes = int(len(df_fixed))

    df_clean = _hard_clean(df_fixed)
    rows_after_hard_clean = int(len(df_clean))

    df_clean.to_csv(CLEAN_CSV, index=False)

    completeness_pct = _completeness(df_clean)
    validity_pct, validity_issues = _validity(df_clean, run_id, run_ts)
    uniqueness_pct, uniqueness_issues = _uniqueness(df_clean, run_id, run_ts)
    timeliness_pct, timeliness_issues = _timeliness(df_clean, run_id, run_ts)

    overall_pct = _overall([completeness_pct, validity_pct, uniqueness_pct, timeliness_pct])

    summary_row = pd.DataFrame(
        [
            {
                "run_id": run_id,
                "run_ts_utc": run_ts,
                "rows_raw": rows_raw,
                "rows_after_global_fixes": rows_after_global_fixes,
                "rows_after_hard_clean": rows_after_hard_clean,
                "completeness_pct": _pct(completeness_pct),
                "validity_pct": _pct(validity_pct),
                "uniqueness_pct": _pct(uniqueness_pct),
                "timeliness_pct": _pct(timeliness_pct),
                "overall_pct": _pct(overall_pct),
                "rules_version": rules_version or "—",
            }
        ]
    )
    _append_csv(DQ_SUMMARY_CSV, summary_row)

    all_issues = validity_issues + uniqueness_issues + timeliness_issues
    if all_issues:
        issues_df = pd.DataFrame([it.__dict__ for it in all_issues])
        _append_csv(DQ_ISSUES_CSV, issues_df)
    else:
        # Ensure the file exists for dashboards that expect it.
        if not DQ_ISSUES_CSV.exists():
            pd.DataFrame(
                columns=[
                    "run_id",
                    "run_ts_utc",
                    "dimension",
                    "rule",
                    "column",
                    "failed_count",
                    "failed_pct",
                    "sample_values",
                ]
            ).to_csv(DQ_ISSUES_CSV, index=False)

    # Write a small per-run snapshot for artifact/history workflows.
    run_dir = HISTORY_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "summary.json").write_text(
        json.dumps(summary_row.iloc[0].to_dict(), indent=2),
        encoding="utf-8",
    )

    print(f"Clean CSV: {CLEAN_CSV} | exists: {CLEAN_CSV.exists()}")
    print(f"Run History CSV: {DQ_SUMMARY_CSV} | exists: {DQ_SUMMARY_CSV.exists()}")
    print(f"Issues CSV: {DQ_ISSUES_CSV} | exists: {DQ_ISSUES_CSV.exists()}")


def main() -> None:
    run()


if __name__ == "__main__":
    main()
