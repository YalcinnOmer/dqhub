from __future__ import annotations

from pathlib import Path
from datetime import datetime
import re
import yaml
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
INPUT = DATA / "synthetic" / "input.csv"
OUTPUT = DATA / "output"
REPORTS = ROOT / "reports"
RULES = ROOT / "rules" / "rules.yaml"

DQ_SUMMARY_CSV = REPORTS / "dq_summary.csv"
DQ_ISSUES_CSV = REPORTS / "dq_issues.csv"


def c_in(df: pd.DataFrame, col: str) -> bool:
    return col in df.columns


def collapse_ws(s):
    return re.sub(r"\s+", " ", s.strip()) if isinstance(s, str) else s


def email_valid(s: str) -> bool:
    if not isinstance(s, str):
        return False
    return bool(re.match(r"^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}$", s.strip()))


def build_email_from_name(name: str, fallback: str) -> str:
    base = str(name or "").lower().strip()
    base = base.replace(" ", ".").replace("'", "")
    base = re.sub(r"[^a-z0-9._-]", "", base)
    base = re.sub(r"\.+", ".", base).strip(".")
    if not base:
        base = fallback
    return f"{base}@example.com"


def phone_to_e164_ca(s):
    s = "" if s is pd.NA or s is None else str(s)
    digits = re.sub(r"\D", "", s)[-10:].rjust(10, "0")
    return "+1" + digits


def phone_valid_e164_ca(s: str) -> bool:
    return isinstance(s, str) and bool(re.match(r"^\+1\d{10}$", s))


def try_parse_date_series(ser: pd.Series, fmts: list[str]) -> pd.Series:
    out = pd.to_datetime(ser, errors="coerce", utc=True)
    best = out
    for fmt in fmts:
        cand = pd.to_datetime(ser, format=fmt, errors="coerce", utc=True)
        if cand.notna().sum() >= best.notna().sum():
            best = cand
    return best


def _append_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    header = not path.exists()
    df.to_csv(path, mode="a", index=False, header=header)


def _not_empty(df: pd.DataFrame, col: str) -> pd.Series:
    return ~(df[col].isna() | (df[col].astype(str).str.strip() == ""))


def _safe_pct(mask: pd.Series) -> float:
    if mask is None or len(mask) == 0:
        return float("nan")
    return round(float(mask.mean() * 100.0), 2)


def _timeliness_mask(df: pd.DataFrame, schema: dict, now_utc: pd.Timestamp) -> tuple[pd.Series | None, dict]:
    """
    Timeliness rule: created_at must be within max_age_days and not in the future.
    Configure in rules.yaml schema:
      timeliness_max_age_days: 365
    """
    if "created_at" not in df.columns:
        return None, {"applicable": False, "reason": "missing_created_at"}

    max_age_days = schema.get("timeliness_max_age_days", schema.get("max_record_age_days", 365))
    created = pd.to_datetime(df["created_at"], errors="coerce", utc=True)
    cutoff = now_utc - pd.Timedelta(days=int(max_age_days))
    mask = created.notna() & (created >= cutoff) & (created <= now_utc)

    return mask, {"applicable": True, "max_age_days": int(max_age_days), "cutoff": str(cutoff)}


def _build_issues(
    df: pd.DataFrame,
    cols_cfg: list[dict],
    schema: dict,
    date_cols: list[str],
    date_min: dict,
    date_max: dict,
    pk: list[str],
    now_utc: pd.Timestamp,
) -> pd.DataFrame:
    rows: list[dict] = []

    required = {c["name"]: bool(c.get("required", False)) for c in cols_cfg}
    unique = {c["name"]: bool(c.get("unique", False)) for c in cols_cfg}

    # Completeness: required not empty
    for col, is_req in required.items():
        if is_req and col in df.columns:
            bad = ~_not_empty(df, col)
            if bad.any():
                samp = df.loc[bad, col].astype(str).head(5).tolist()
                rows.append(
                    {
                        "dimension": "Completeness",
                        "rule": f"required_not_empty:{col}",
                        "column": col,
                        "failed_count": int(bad.sum()),
                        "failed_pct": round(100.0 * float(bad.mean()), 2),
                        "sample_values": str(samp),
                    }
                )

    # Validity: email / phone formats
    if "email" in df.columns:
        bad = ~df["email"].apply(email_valid)
        if bad.any():
            samp = df.loc[bad, "email"].astype(str).head(5).tolist()
            rows.append(
                {
                    "dimension": "Validity",
                    "rule": "email_format",
                    "column": "email",
                    "failed_count": int(bad.sum()),
                    "failed_pct": round(100.0 * float(bad.mean()), 2),
                    "sample_values": str(samp),
                }
            )

    if "phone" in df.columns:
        bad = ~df["phone"].apply(phone_valid_e164_ca)
        if bad.any():
            samp = df.loc[bad, "phone"].astype(str).head(5).tolist()
            rows.append(
                {
                    "dimension": "Validity",
                    "rule": "phone_format_e164_ca",
                    "column": "phone",
                    "failed_count": int(bad.sum()),
                    "failed_pct": round(100.0 * float(bad.mean()), 2),
                    "sample_values": str(samp),
                }
            )

    # Validity: date ranges
    for col in date_cols:
        ser = pd.to_datetime(df[col], errors="coerce", utc=True)
        if col in date_min:
            bad = ser.notna() & (ser < date_min[col])
            if bad.any():
                samp = ser.loc[bad].astype(str).head(5).tolist()
                rows.append(
                    {
                        "dimension": "Validity",
                        "rule": f"date_min:{col}",
                        "column": col,
                        "failed_count": int(bad.sum()),
                        "failed_pct": round(100.0 * float(bad.mean()), 2),
                        "sample_values": str(samp),
                    }
                )
        if col in date_max:
            bad = ser.notna() & (ser > date_max[col])
            if bad.any():
                samp = ser.loc[bad].astype(str).head(5).tolist()
                rows.append(
                    {
                        "dimension": "Validity",
                        "rule": f"date_max:{col}",
                        "column": col,
                        "failed_count": int(bad.sum()),
                        "failed_pct": round(100.0 * float(bad.mean()), 2),
                        "sample_values": str(samp),
                    }
                )

    # Uniqueness: PK and unique columns
    if pk and all(c in df.columns for c in pk):
        tmp = df.copy()
        if "created_at" in tmp.columns:
            tmp["created_at"] = pd.to_datetime(tmp["created_at"], errors="coerce", utc=True)
            tmp = tmp.sort_values(["created_at"], ascending=False)
        dup = tmp[pk].duplicated(keep="first")
        if dup.any():
            samp = tmp.loc[dup, pk].astype(str).head(5).to_dict(orient="records")
            rows.append(
                {
                    "dimension": "Uniqueness",
                    "rule": f"primary_key:{'|'.join(pk)}",
                    "column": "|".join(pk),
                    "failed_count": int(dup.sum()),
                    "failed_pct": round(100.0 * float(dup.mean()), 2),
                    "sample_values": str(samp),
                }
            )

    for col, is_uni in unique.items():
        if is_uni and col in df.columns:
            dup = df[col].duplicated(keep="first")
            if dup.any():
                samp = df.loc[dup, col].astype(str).head(5).tolist()
                rows.append(
                    {
                        "dimension": "Uniqueness",
                        "rule": f"unique:{col}",
                        "column": col,
                        "failed_count": int(dup.sum()),
                        "failed_pct": round(100.0 * float(dup.mean()), 2),
                        "sample_values": str(samp),
                    }
                )

    # Timeliness
    tmask, tmeta = _timeliness_mask(df, schema, now_utc)
    if tmask is not None and tmeta.get("applicable"):
        bad = ~tmask
        if bad.any():
            samp = pd.to_datetime(df["created_at"], errors="coerce", utc=True).loc[bad].astype(str).head(5).tolist()
            rows.append(
                {
                    "dimension": "Timeliness",
                    "rule": f"created_at_max_age_days:{tmeta.get('max_age_days')}",
                    "column": "created_at",
                    "failed_count": int(bad.sum()),
                    "failed_pct": round(100.0 * float(bad.mean()), 2),
                    "sample_values": str(samp),
                }
            )

    return pd.DataFrame(rows)


def run(
    input_path: Path = INPUT,
    rules_path: Path = RULES,
    out_dir: Path = OUTPUT,
    report_path: Path = REPORTS / "DQ_Report.xlsx",
):
    out_dir.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)

    rules = yaml.safe_load(rules_path.read_text(encoding="utf-8"))
    schema = rules.get("schema", {})
    cols_cfg = rules.get("columns", [])
    date_formats = schema.get("date_formats", ["%Y-%m-%d"])
    pk = schema.get("primary_key", [])

    df_raw = pd.read_csv(input_path)
    df = df_raw.copy()

    obj_cols = df.select_dtypes(include=["object"]).columns
    if len(obj_cols) > 0:
        df[obj_cols] = df[obj_cols].apply(lambda col: col.map(lambda x: x.strip() if isinstance(x, str) else x))
        df[obj_cols] = df[obj_cols].apply(lambda col: col.map(collapse_ws))

    before = len(df)
    df = df.dropna(how="all")
    drop_empty = before - len(df)

    before = len(df)
    df = df.drop_duplicates()
    drop_dupes = before - len(df)

    req_cols = [c["name"] for c in cols_cfg]
    for c in req_cols:
        if c not in df.columns:
            df[c] = pd.NA

    if c_in(df, "email"):
        df["email"] = (
            df["email"].astype(str).str.strip().str.lower().str.replace("_at_", "@", regex=False)
        )
        mask_bad = ~df["email"].apply(email_valid)
        if c_in(df, "full_name"):
            df.loc[mask_bad, "email"] = df.loc[mask_bad, ["full_name", "id"]].apply(
                lambda r: build_email_from_name(r["full_name"], f"user{r['id']}"), axis=1
            )
        else:
            df.loc[mask_bad, "email"] = df.loc[mask_bad, "id"].apply(lambda x: f"user{x}@example.com")

    if c_in(df, "phone"):
        df["phone"] = df["phone"].apply(phone_to_e164_ca)
        bad_phone = ~df["phone"].apply(phone_valid_e164_ca)
        df.loc[bad_phone, "phone"] = "+1" + "0" * 10

    if c_in(df, "full_name"):
        df["full_name"] = df["full_name"].fillna("").astype(str).str.strip()
        df.loc[df["full_name"] == "", "full_name"] = "Unknown"

    if c_in(df, "created_at"):
        df["created_at"] = pd.to_datetime(df["created_at"], errors="coerce", utc=True)
        now_utc = pd.Timestamp.now(tz="UTC")
        df["created_at"] = df["created_at"].fillna(now_utc)

    date_cols = [x["name"] for x in cols_cfg if x.get("type") == "date" and c_in(df, x["name"])]
    for c in date_cols:
        df[c] = try_parse_date_series(df[c], date_formats)

    if c_in(df, "birth_date"):
        bd = pd.to_datetime(df["birth_date"], errors="coerce", utc=True)
        if bd.notna().any():
            med = bd.dropna().median()
            df["birth_date"] = bd.fillna(med)
        else:
            df["birth_date"] = pd.Timestamp("2000-01-01", tz="UTC")

    col_req = {c["name"]: c.get("required", False) for c in cols_cfg}
    col_unique = {c["name"]: c.get("unique", False) for c in cols_cfg}
    date_min = {
        c["name"]: pd.to_datetime(c.get("min"), utc=True)
        for c in cols_cfg
        if c.get("type") == "date" and c.get("min")
    }
    date_max = {
        c["name"]: pd.to_datetime(c.get("max"), utc=True)
        for c in cols_cfg
        if c.get("type") == "date" and c.get("max")
    }

    m_all = pd.Series(True, index=df.index)

    for c in req_cols:
        if col_req.get(c, False):
            m_all &= _not_empty(df, c)

    if c_in(df, "email"):
        m_all &= df["email"].apply(email_valid)

    if c_in(df, "phone"):
        m_all &= df["phone"].apply(phone_valid_e164_ca)

    for c in date_cols:
        notna = df[c].notna()
        if c in date_min:
            m_all &= (~notna) | (df[c] >= date_min[c])
        if c in date_max:
            m_all &= (~notna) | (df[c] <= date_max[c])

    if pk:
        tmp = df.copy()
        if c_in(tmp, "created_at"):
            tmp["created_at"] = pd.to_datetime(tmp["created_at"], errors="coerce", utc=True)
            tmp = tmp.sort_values(["created_at"], ascending=False)
        keep_idx = ~tmp[pk].duplicated(keep="first")
        m_all &= keep_idx.reindex(df.index, fill_value=False)

    for c, is_unique in col_unique.items():
        if is_unique and c_in(df, c):
            m_all &= ~df[c].duplicated(keep="first")

    clean_df = df.loc[m_all].copy()

    for c in ["id", "full_name", "email", "phone"]:
        if c_in(clean_df, c):
            clean_df[c] = clean_df[c].fillna("")

    if c_in(clean_df, "created_at"):
        clean_df["created_at"] = pd.to_datetime(clean_df["created_at"], errors="coerce", utc=True)
        clean_df["created_at"] = clean_df["created_at"].fillna(pd.Timestamp.now(tz="UTC"))

    if c_in(clean_df, "birth_date"):
        clean_df["birth_date"] = pd.to_datetime(clean_df["birth_date"], errors="coerce", utc=True)
        if clean_df["birth_date"].isna().any():
            bmed = pd.to_datetime(df["birth_date"], errors="coerce", utc=True).dropna()
            if len(bmed) > 0:
                clean_df["birth_date"] = clean_df["birth_date"].fillna(bmed.median())
            else:
                clean_df["birth_date"] = clean_df["birth_date"].fillna(pd.Timestamp("2000-01-01", tz="UTC"))

    stats_df = pd.DataFrame(
        [
            {
                "column": c,
                "nulls": int(clean_df[c].isna().sum()),
                "distinct": int(clean_df[c].nunique(dropna=True)),
                "pct_null": round(100 * clean_df[c].isna().mean(), 2),
                "sample": str(clean_df[c].dropna().astype(str).head(3).tolist()),
            }
            for c in req_cols
            if c_in(clean_df, c)
        ]
    )

    summary = pd.DataFrame(
        [
            {"Metric": "Total Rows (raw)", "Value": len(df_raw)},
            {"Metric": "After Global Fixes", "Value": len(df)},
            {"Metric": "After Hard Clean", "Value": len(clean_df)},
            {"Metric": "Rules Version", "Value": rules.get("meta", {}).get("version", "1.0.0")},
        ]
    )

    now_utc = pd.Timestamp.now(tz="UTC")

    issues_df = _build_issues(
        df=df,
        cols_cfg=cols_cfg,
        schema=schema,
        date_cols=date_cols,
        date_min=date_min,
        date_max=date_max,
        pk=pk,
        now_utc=now_utc,
    )

    fixes_df = pd.DataFrame(
        [
            {"rule": "drop_empty_rows", "count": int(drop_empty)},
            {"rule": "drop_exact_duplicates", "count": int(drop_dupes)},
        ]
    )

    # Dimension masks (row-level)
    m_required = pd.Series(True, index=df.index)
    for c in req_cols:
        if col_req.get(c, False) and c_in(df, c):
            m_required &= _not_empty(df, c)

    validity_checks = 0
    m_valid = pd.Series(True, index=df.index)
    if c_in(df, "email"):
        validity_checks += 1
        m_valid &= df["email"].apply(email_valid)
    if c_in(df, "phone"):
        validity_checks += 1
        m_valid &= df["phone"].apply(phone_valid_e164_ca)
    if len(date_cols) > 0:
        validity_checks += 1
        for c in date_cols:
            notna = df[c].notna()
            if c in date_min:
                m_valid &= (~notna) | (df[c] >= date_min[c])
            if c in date_max:
                m_valid &= (~notna) | (df[c] <= date_max[c])

    uniq_checks = 0
    m_unique = pd.Series(True, index=df.index)
    if pk and all(c in df.columns for c in pk):
        uniq_checks += 1
        tmp = df.copy()
        if c_in(tmp, "created_at"):
            tmp["created_at"] = pd.to_datetime(tmp["created_at"], errors="coerce", utc=True)
            tmp = tmp.sort_values(["created_at"], ascending=False)
        keep_idx = ~tmp[pk].duplicated(keep="first")
        m_unique &= keep_idx.reindex(df.index, fill_value=False)

    for c, is_unique in col_unique.items():
        if is_unique and c_in(df, c):
            uniq_checks += 1
            m_unique &= ~df[c].duplicated(keep="first")

    m_time, _tmeta = _timeliness_mask(df, schema, now_utc)

    run_id = now_utc.strftime("%Y%m%d_%H%M%S")

    dq_summary_row = pd.DataFrame(
        [
            {
                "run_id": run_id,
                "run_ts_utc": now_utc.isoformat(),
                "rows_raw": int(len(df_raw)),
                "rows_after_global_fixes": int(len(df)),
                "rows_after_hard_clean": int(len(clean_df)),
                "completeness_pct": _safe_pct(m_required),
                "validity_pct": _safe_pct(m_valid) if validity_checks > 0 else float("nan"),
                "uniqueness_pct": _safe_pct(m_unique) if uniq_checks > 0 else float("nan"),
                "timeliness_pct": _safe_pct(m_time) if m_time is not None else float("nan"),
                "overall_pct": _safe_pct(m_all),
                "rules_version": rules.get("meta", {}).get("version", "1.0.0"),
            }
        ]
    )

    issues_out = issues_df.copy()
    if not issues_out.empty:
        issues_out.insert(0, "run_id", run_id)
        issues_out.insert(1, "run_ts_utc", now_utc.isoformat())

    _append_csv(dq_summary_row, DQ_SUMMARY_CSV)
    if not issues_out.empty:
        _append_csv(issues_out, DQ_ISSUES_CSV)

    clean_csv = out_dir / "clean.csv"
    clean_df.to_csv(clean_csv, index=False)

    with pd.ExcelWriter(report_path, engine="openpyxl") as xw:
        summary.to_excel(xw, index=False, sheet_name="Summary")
        stats_df.to_excel(xw, index=False, sheet_name="Stats")
        issues_df.to_excel(xw, index=False, sheet_name="Issues")
        fixes_df.to_excel(xw, index=False, sheet_name="Fixes_Applied")

    print("Wrote:", report_path, "| exists:", report_path.exists())
    print("Clean CSV:", clean_csv, "| exists:", clean_csv.exists())
    print("Run History CSV:", DQ_SUMMARY_CSV, "| exists:", DQ_SUMMARY_CSV.exists())
    print("Issues CSV:", DQ_ISSUES_CSV, "| exists:", DQ_ISSUES_CSV.exists())


if __name__ == "__main__":
    run()
