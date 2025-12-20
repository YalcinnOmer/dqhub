from __future__ import annotations

import typer

from .make_synth import main as make_synth_main
from .run_dq import main as run_dq_main
from .report import main as report_main
from .verify import main as verify_main
from .pipeline import main as pipeline_main

app = typer.Typer(help="DQ Hub CLI")


@app.command("synth")
def synth() -> None:
    make_synth_main()


@app.command("clean")
def clean() -> None:
    run_dq_main()


@app.command("report")
def report() -> None:
    report_main()


@app.command("verify")
def verify() -> None:
    verify_main()


@app.command("pipeline")
def pipeline() -> None:
    pipeline_main()


def main() -> None:
    app()
