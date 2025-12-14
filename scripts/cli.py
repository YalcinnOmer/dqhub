from __future__ import annotations

import subprocess
import sys
from pathlib import Path
import typer

app = typer.Typer(help="DQ Hub CLI")

ROOT = Path(__file__).resolve().parents[1]

def _py(script: str, *args: str) -> int:
    cmd = [sys.executable, str(ROOT / "scripts" / script), *args]
    return subprocess.call(cmd)

@app.command()
def synth() -> None:
    code = _py("make_synth.py")
    raise SystemExit(code)

@app.command()
def clean() -> None:
    code = _py("run_dq.py")
    raise SystemExit(code)

@app.command()
def verify() -> None:
    code = _py("verify.py")
    raise SystemExit(code)

@app.command()
def report() -> None:
    code = _py("report.py")
    raise SystemExit(code)

@app.command()
def pipeline() -> None:
    # same order you already use
    if _py("make_synth.py") != 0: raise SystemExit(2)
    if _py("run_dq.py") != 0:     raise SystemExit(2)
    if _py("verify.py") != 0:     raise SystemExit(2)
    if _py("report.py") != 0:     raise SystemExit(2)
    print("\nAll steps completed successfully. ✅")

if __name__ == "__main__":
    app()
