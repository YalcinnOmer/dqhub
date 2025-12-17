from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
REPORTS = ROOT / "reports"

SYNTHETIC_DIR = DATA / "synthetic"
OUTPUT_DIR = DATA / "output"

INPUT = SYNTHETIC_DIR / "input.csv"
CLEAN = OUTPUT_DIR / "clean.csv"
