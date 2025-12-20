from __future__ import annotations

from pathlib import Path
import sys

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


def main() -> None:
    missing = [p for p in REQUIRED if not p.exists()]
    if missing:
        print("[FAIL] Missing artifacts:")
        for p in missing:
            print(f"  - {p}")
        raise SystemExit(2)

    print("[OK] All expected artifacts exist.")
    for p in REQUIRED:
        print(f"  - {p}")
    raise SystemExit(0)


if __name__ == "__main__":
    main()
