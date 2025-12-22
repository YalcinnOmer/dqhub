from __future__ import annotations

import typer

from .make_synth import main as make_synth_main
from .run_dq import main as run_dq_main
from .report import main as report_main
from .verify import main as verify_main
from .pipeline import main as pipeline_main

app = typer.Typer(
    help="DQ Hub CLI - synthetic data generation, data-quality checks, and dashboard reports."
)

@app.command("synth")
def synth(
    rows: int = typer.Option(522, help="Number of synthetic rows to generate."),
    seed: int = typer.Option(42, help="Random seed for deterministic output."),
    scenario: str = typer.Option(
        "auto",
        help="Scenario: baseline | improved | auto. Auto uses baseline for first run, improved afterwards.",
    ),
) -> None:
    raise SystemExit(make_synth_main(rows=rows, seed=seed, scenario=scenario))

@app.command("clean")
def clean() -> None:
    raise SystemExit(run_dq_main())

@app.command("report")
def report() -> None:
    raise SystemExit(report_main())

@app.command("verify")
def verify() -> None:
    raise SystemExit(verify_main())

@app.command("pipeline")
def pipeline() -> None:
    raise SystemExit(pipeline_main())

def main() -> int:
    app()
    return 0
