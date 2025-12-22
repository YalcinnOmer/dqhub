from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

REQUIRED = [
    ROOT / "data" / "synthetic" / "input.csv",
    ROOT / "data" / "output" / "raw.csv",
    ROOT / "data" / "output" / "clean.csv",
    ROOT / "reports" / "dq_summary.csv",
    ROOT / "reports" / "dq_issues.csv",
    ROOT / "reports" / "DQ_Report.html",
    ROOT / "reports" / "DQ_Report.xlsx",
]


def main() -> int:
    missing = [p for p in REQUIRED if not p.exists()]
    if missing:
        print("[FAIL] Missing artifacts:")
        for p in missing:
            print(f"  - {p}")
        return 2

    print("[OK] All expected artifacts exist.")
    for p in REQUIRED:
        print(f"  - {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
