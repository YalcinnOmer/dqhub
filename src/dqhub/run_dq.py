from __future__ import annotations

import json
import math
import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = ROOT / "data"
SYNTH_DIR = DATA_DIR / "synthetic"
OUTPUT_DIR = DATA_DIR / "output"

REPORTS_DIR = ROOT / "reports"
HISTORY_DIR = REPORTS_DIR / "history"

INPUT_CSV = SYNTH_DIR / "input.csv"
RAW_CSV = OUTPUT_DIR / "raw.csv"
CLEAN_CSV = OUTPUT_DIR / "clean.csv"

DQ_SUMMARY_CSV = REPORTS_DIR / "dq_summary.csv"
DQ_ISSUES_CSV = REPORTS_DIR / "dq_issues.csv"

RULES_YAML = ROOT / "rules" / "rules.yaml"

_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _run_id(ts: datetime) -> str:
    return ts.strftime("%Y%m%d_%H%M%S")


def _is_nan(x: Any) -> bool:
    try:
        return x is None or (isinstance(x, float) and math.isnan(x))
    except Exception:
        return False


def _safe_str(x: Any) -> str | None:
    if _is_nan(x):
        return None
    try:
        s = str(x)
        s = s.strip()
        return s if s else None
    except Exception:
        return None


def _safe_float(x: Any) -> float | None:
    if _is_nan(x):
        return None
    try:
        return float(x)
    except Exception:
        return None


def _ensure_dirs() -> None:
    SYNTH_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    HISTORY_DIR.mkdir(parents=True, exist_ok=True)


def _load_rules_version() -> str:
    if not RULES_YAML.exists():
        return "0.0.0"
    try:
        obj = yaml.safe_load(RULES_YAML.read_text(encoding="utf-8")) or {}
        v = obj.get("version")
        if isinstance(v, str) and v.strip():
            return v.strip()
        return "1.0.0"
    except Exception:
        return "1.0.0"


@dataclass(frozen=True)
class DQResult:
    rows_raw: int
    rows_after_global_fixes: int
    rows_after_hard_clean: int
    completeness_pct: float | None
    validity_pct: float | None
    uniqueness_pct: float | None
    timeliness_pct: float | None
    overall_pct: float | None


def _read_input() -> pd.DataFrame:
    if INPUT_CSV.exists():
        return pd.read_csv(INPUT_CSV)
    if RAW_CSV.exists():
        # fallback only; CI fix is to always generate INPUT_CSV
        return pd.read_csv(RAW_CSV)
    raise FileNotFoundError(f"Missing input dataset: {INPUT_CSV}")


def _normalize(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    for c in ["id", "email", "country", "currency"]:
        if c in out.columns:
            out[c] = out[c].apply(_safe_str)

    if "email" in out.columns:
        out["email"] = out["email"].apply(lambda x: x.lower().strip() if isinstance(x, str) else x)

    if "country" in out.columns:
        out["country"] = out["country"].apply(lambda x: x.upper().strip() if isinstance(x, str) else x)

    if "currency" in out.columns:
        out["currency"] = out["currency"].apply(lambda x: x.upper().strip() if isinstance(x, str) else x)

    if "amount" in out.columns:
        out["amount"] = pd.to_numeric(out["amount"], errors="coerce")
        out["amount"] = out["amount"].round(2)

    for c in ["order_ts_utc", "ingested_ts_utc"]:
        if c in out.columns:
            out[c] = pd.to_datetime(out[c], utc=True, errors="coerce")

    return out


def _hard_clean(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    required = ["id", "email", "country", "currency", "amount", "order_ts_utc", "ingested_ts_utc"]
    present = [c for c in required if c in out.columns]

    # Drop rows missing required fields
    for c in present:
        out = out[out[c].notna()]

    # Basic constraints
    if "amount" in out.columns:
        out = out[out["amount"] >= 0]

    return out.reset_index(drop=True)


def _pct(numer: float, denom: float) -> float | None:
    if denom <= 0:
        return None
    return round(100.0 * (numer / denom), 2)


def _completeness_pct(df: pd.DataFrame) -> float | None:
    required = ["id", "email", "country", "currency", "amount", "order_ts_utc", "ingested_ts_utc"]
    cols = [c for c in required if c in df.columns]
    if not cols or df.empty:
        return None

    total_cells = float(len(df) * len(cols))
    missing = float(df[cols].isna().sum().sum())
    present_cells = total_cells - missing
    return _pct(present_cells, total_cells)


def _validity_checks(df: pd.DataFrame) -> tuple[float | None, list[dict[str, Any]]]:
    if df.empty:
        return None, []

    total = len(df)
    issues: list[dict[str, Any]] = []

    masks: list[pd.Series] = []

    if "email" in df.columns:
        m = df["email"].astype(str).fillna("").str.match(_EMAIL_RE)
        masks.append(m)
        bad = (~m).sum()
        issues.append(
            {
                "dimension": "Validity",
                "rule": "email:format",
                "column": "email",
                "failed_count": int(bad),
                "failed_pct": float(_pct(float(bad), float(total)) or 0.0),
                "sample_values": json.dumps(df.loc[~m, "email"].astype(str).head(10).tolist()),
            }
        )

    if "country" in df.columns:
        m = df["country"].astype(str).fillna("").str.match(r"^[A-Z]{2}$")
        masks.append(m)
        bad = (~m).sum()
        issues.append(
            {
                "dimension": "Validity",
                "rule": "country:iso2",
                "column": "country",
                "failed_count": int(bad),
                "failed_pct": float(_pct(float(bad), float(total)) or 0.0),
                "sample_values": json.dumps(df.loc[~m, "country"].astype(str).head(10).tolist()),
            }
        )

    if "currency" in df.columns:
        m = df["currency"].astype(str).fillna("").str.match(r"^[A-Z]{3}$")
        masks.append(m)
        bad = (~m).sum()
        issues.append(
            {
                "dimension": "Validity",
                "rule": "currency:iso3",
                "column": "currency",
                "failed_count": int(bad),
                "failed_pct": float(_pct(float(bad), float(total)) or 0.0),
                "sample_values": json.dumps(df.loc[~m, "currency"].astype(str).head(10).tolist()),
            }
        )

    if "amount" in df.columns:
        m = df["amount"].notna() & (df["amount"] >= 0)
        masks.append(m)
        bad = (~m).sum()
        issues.append(
            {
                "dimension": "Validity",
                "rule": "amount:non_negative",
                "column": "amount",
                "failed_count": int(bad),
                "failed_pct": float(_pct(float(bad), float(total)) or 0.0),
                "sample_values": json.dumps(df.loc[~m, "amount"].head(10).tolist()),
            }
        )

    if "order_ts_utc" in df.columns and "ingested_ts_utc" in df.columns:
        m = df["order_ts_utc"].notna() & df["ingested_ts_utc"].notna()
        masks.append(m)
        bad = (~m).sum()
        issues.append(
            {
                "dimension": "Validity",
                "rule": "timestamps:parsable",
                "column": "order_ts_utc,ingested_ts_utc",
                "failed_count": int(bad),
                "failed_pct": float(_pct(float(bad), float(total)) or 0.0),
                "sample_values": json.dumps([]),
            }
        )

    if not masks:
        return None, []

    # Validity score = average pass-rate across checks
    pass_rates = [float(m.sum()) / float(total) for m in masks]
    score = round(100.0 * (sum(pass_rates) / float(len(pass_rates))), 2)
    return score, issues


def _uniqueness_pct(df: pd.DataFrame) -> tuple[float | None, dict[str, Any] | None]:
    if df.empty or "id" not in df.columns:
        return None, None

    total = len(df)
    dup_mask = df.duplicated(subset=["id"], keep="first")
    dup_count = int(dup_mask.sum())

    score = round(100.0 * (1.0 - (dup_count / float(total))), 2)

    examples = df.loc[df.duplicated(subset=["id"], keep=False), "id"].astype(str).head(10).tolist()
    meta = {"key": "id", "duplicates": dup_count, "total": total, "examples": examples}
    return score, meta


def _timeliness_pct(df: pd.DataFrame) -> tuple[float | None, dict[str, Any] | None]:
    if df.empty or "order_ts_utc" not in df.columns or "ingested_ts_utc" not in df.columns:
        return None, None

    o = pd.to_datetime(df["order_ts_utc"], utc=True, errors="coerce")
    i = pd.to_datetime(df["ingested_ts_utc"], utc=True, errors="coerce")
    m = o.notna() & i.notna()
    if int(m.sum()) == 0:
        return None, None

    delay_h = (i[m] - o[m]).dt.total_seconds() / 3600.0
    ok = (delay_h >= 0) & (delay_h <= 24.0)

    score = round(100.0 * (float(ok.sum()) / float(len(delay_h))), 2)

    stats = {
        "p50_h": round(float(delay_h.quantile(0.50)), 4),
        "p95_h": round(float(delay_h.quantile(0.95)), 4),
        "max_h": round(float(delay_h.max()), 4),
        "late_24h_count": int((delay_h > 24.0).sum()),
        "late_24h_pct": round(100.0 * float((delay_h > 24.0).sum()) / float(len(delay_h)), 4),
    }
    return score, stats


def _overall(scores: list[float | None]) -> float | None:
    xs = [s for s in scores if s is not None]
    if not xs:
        return None
    return round(sum(xs) / float(len(xs)), 2)


def _append_csv(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame([row])
    if path.exists():
        df.to_csv(path, mode="a", header=False, index=False)
    else:
        df.to_csv(path, index=False)


def main() -> None:
    _ensure_dirs()

    ts = _utc_now()
    rid = _run_id(ts)

    rules_version = _load_rules_version()

    df_raw = _read_input()
    rows_raw = int(len(df_raw))

    df_norm = _normalize(df_raw)
    rows_after_global = int(len(df_norm))

    # Hard clean produces the clean dataset
    df_clean = _hard_clean(df_norm)
    rows_after_hard = int(len(df_clean))

    df_clean.to_csv(CLEAN_CSV, index=False)
    print(f"Clean CSV: {CLEAN_CSV} | exists: {CLEAN_CSV.exists()}")

    comp = _completeness_pct(df_clean)
    valid, validity_issues = _validity_checks(df_clean)
    uniq, uniq_meta = _uniqueness_pct(df_clean)
    time_score, time_meta = _timeliness_pct(df_clean)

    overall = _overall([comp, valid, uniq, time_score])

    # Issues aggregation
    issues_rows: list[dict[str, Any]] = []

    # Uniqueness issue row
    if uniq_meta is not None and uniq_meta.get("duplicates", 0) > 0:
        dup_count = int(uniq_meta["duplicates"])
        total = int(uniq_meta["total"])
        issues_rows.append(
            {
                "dimension": "Uniqueness",
                "rule": "unique:id",
                "column": "id",
                "failed_count": dup_count,
                "failed_pct": float(_pct(float(dup_count), float(total)) or 0.0),
                "sample_values": json.dumps(uniq_meta.get("examples", [])),
            }
        )

    # Timeliness issue row (late > 24h)
    if time_meta is not None:
        late = int(time_meta.get("late_24h_count", 0))
        denom = float(rows_after_hard) if rows_after_hard > 0 else 1.0
        if late > 0:
            issues_rows.append(
                {
                    "dimension": "Timeliness",
                    "rule": "latency:lte_24h",
                    "column": "order_ts_utc,ingested_ts_utc",
                    "failed_count": late,
                    "failed_pct": float(_pct(float(late), denom) or 0.0),
                    "sample_values": json.dumps([]),
                }
            )

    issues_rows.extend(validity_issues)

    # Write issues CSV (overwrite each run; append is handled by including run_id)
    if issues_rows:
        issues_df = pd.DataFrame(issues_rows)
        issues_df.insert(0, "run_ts_utc", ts.isoformat())
        issues_df.insert(0, "run_id", rid)

        if DQ_ISSUES_CSV.exists():
            # Keep prior history by appending
            issues_df.to_csv(DQ_ISSUES_CSV, mode="a", header=False, index=False)
        else:
            issues_df.to_csv(DQ_ISSUES_CSV, index=False)

    print(f"Issues CSV: {DQ_ISSUES_CSV} | exists: {DQ_ISSUES_CSV.exists()}")

    # Summary row
    summary_row = {
        "run_id": rid,
        "run_ts_utc": ts.isoformat(),
        "rows_raw": rows_raw,
        "rows_after_global_fixes": rows_after_global,
        "rows_after_hard_clean": rows_after_hard,
        "completeness_pct": comp,
        "validity_pct": valid,
        "uniqueness_pct": uniq,
        "timeliness_pct": time_score,
        "overall_pct": overall,
        "rules_version": rules_version,
    }
    _append_csv(DQ_SUMMARY_CSV, summary_row)
    print(f"Run History CSV: {DQ_SUMMARY_CSV} | exists: {DQ_SUMMARY_CSV.exists()}")

    # Optional run history snapshot directory (useful for CI artifacts)
    snap_dir = HISTORY_DIR / rid
    snap_dir.mkdir(parents=True, exist_ok=True)

    try:
        (snap_dir / "clean.csv").write_text(CLEAN_CSV.read_text(encoding="utf-8"), encoding="utf-8")
    except Exception:
        pass

    # Also store a tiny run metadata json
    meta = {
        "run_id": rid,
        "run_ts_utc": ts.isoformat(),
        "paths": {
            "input_csv": str(INPUT_CSV.relative_to(ROOT)).replace("\\", "/"),
            "clean_csv": str(CLEAN_CSV.relative_to(ROOT)).replace("\\", "/"),
            "dq_summary_csv": str(DQ_SUMMARY_CSV.relative_to(ROOT)).replace("\\", "/"),
            "dq_issues_csv": str(DQ_ISSUES_CSV.relative_to(ROOT)).replace("\\", "/"),
        },
    }
    try:
        (snap_dir / "run_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    except Exception:
        pass


if __name__ == "__main__":
    main()
