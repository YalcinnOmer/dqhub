from __future__ import annotations
from pathlib import Path
import sys, re, yaml
import pandas as pd

ROOT    = Path(__file__).resolve().parents[2]
DATA    = ROOT / "data"
CLEAN   = DATA / "output" / "clean.csv"
RULES   = ROOT / "rules" / "rules.yaml"

EMAIL_RE = re.compile(r"^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}$")
PHONE_RE = re.compile(r"^\+1\d{10}$")

def fail(msg: str) -> None:
    print(f"[FAIL] {msg}")
    sys.exit(2)

def ok(msg: str) -> None:
    print(f"[OK] {msg}")

def main() -> None:
    if not CLEAN.exists():
        fail(f"Missing clean csv: {CLEAN}")
    if not RULES.exists():
        fail(f"Missing rules: {RULES}")

    rules = yaml.safe_load(RULES.read_text(encoding="utf-8")) or {}
    cols_cfg = rules.get("columns", [])

    df = pd.read_csv(CLEAN, dtype=str, keep_default_na=False)
    if df.empty:
        fail("clean.csv is empty")

    df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)

    required = {c["name"]: bool(c.get("required", False)) for c in cols_cfg}
    unique   = {c["name"]: bool(c.get("unique", False))   for c in cols_cfg}

    for col, is_req in required.items():
        if is_req and col in df.columns:
            bad = df[col].isna() | (df[col].astype(str).str.strip() == "")
            if bad.any():
                fail(f"Required column '{col}' has {int(bad.sum())} empty values")
    ok("All required columns are non-empty")

    for col, is_uni in unique.items():
        if is_uni and col in df.columns:
            dups = df[col].duplicated(keep=False)
            if dups.any():
                fail(f"Unique column '{col}' has {int(dups.sum())} duplicated rows")
    ok("All unique constraints satisfied")

    if "email" in df.columns:
        ser = df["email"].astype(str).str.strip()
        bad = ~ser.str.fullmatch(EMAIL_RE)
        if bad.any():
            fail(f"Email format invalid in {int(bad.sum())} rows")
        ok("Email formats valid")

    if "phone" in df.columns:
        ser = df["phone"].astype(str).str.strip()
        bad = ~ser.str.fullmatch(PHONE_RE)
        if bad.any():
            fail(f"Phone format invalid in {int(bad.sum())} rows")
        ok("Phone formats valid")

    for c in cols_cfg:
        if c.get("type") == "date" and c["name"] in df.columns:
            col = c["name"]
            ser = pd.to_datetime(df[col], errors="coerce", utc=True)
            if c.get("min"):
                minv = pd.to_datetime(c["min"], utc=True)
                below = ser.notna() & (ser < minv)
                if below.any():
                    fail(f"Date '{col}' has {int(below.sum())} values below min {minv.date()}")
            if c.get("max"):
                maxv = pd.to_datetime(c["max"], utc=True)
                above = ser.notna() & (ser > maxv)
                if above.any():
                    fail(f"Date '{col}' has {int(above.sum())} values above max {maxv.date()}")
    ok("Date ranges valid")

    print("\nRESULT: PASS — clean.csv satisfies all rules.")
    sys.exit(0)

if __name__ == "__main__":
    main()
