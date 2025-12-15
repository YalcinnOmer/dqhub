from pathlib import Path
import subprocess, sys

ROOT = Path(__file__).resolve().parents[1]

def run_step(title: str, args: list[str]) -> None:
    print(f"\n=== {title} ===")
    proc = subprocess.run([sys.executable, *args])
    if proc.returncode != 0:
        sys.exit(proc.returncode)

def main():
    run_step("1) Generate synthetic", [str(ROOT/"scripts"/"make_synth.py")])
    run_step("2) Clean & report",      [str(ROOT/"scripts"/"run_dq.py")])
    run_step("3) Verify clean.csv",    [str(ROOT/"scripts"/"verify.py")])
    run_step("4) Build HTML report",   [str(ROOT/"scripts"/"report.py")])
    print("\nAll steps completed successfully. ✅")

if __name__ == "__main__":
    main()
