from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import typer

app = typer.Typer(add_completion=False, help="DQHub CLI")

PKG_DIR = Path(__file__).resolve().parent  # .../src/dqhub


def _run(module_file: str, *args: str) -> int:
    """
    Runs a module file (located inside the dqhub package) using the current venv interpreter.
    This keeps the CLI working both in editable installs and in packaged installs.
    """
    cmd = [sys.executable, str(PKG_DIR / module_file), *args]
    return subprocess.call(cmd)


@app.command()
def synth() -> None:
    raise SystemExit(_run("make_synth.py"))


@app.command()
def clean() -> None:
    raise SystemExit(_run("run_dq.py"))


@app.command()
def verify() -> None:
    raise SystemExit(_run("verify.py"))


@app.command()
def report() -> None:
    raise SystemExit(_run("report.py"))


@app.command()
def pipeline() -> None:
    if _run("make_synth.py") != 0:
        raise SystemExit(2)
    if _run("run_dq.py") != 0:
        raise SystemExit(2)
    if _run("verify.py") != 0:
        raise SystemExit(2)
    if _run("report.py") != 0:
        raise SystemExit(2)

    typer.echo("All steps completed successfully.")


def main() -> int:
    """
    Entry point for `dqhub` console script and `python -m dqhub`.
    """
    app()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
