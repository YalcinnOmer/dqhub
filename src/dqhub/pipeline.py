from __future__ import annotations

from .make_synth import main as make_synth_main
from .run_dq import main as run_dq_main
from .report import main as report_main


def main() -> None:
    make_synth_main()
    run_dq_main()
    report_main()


if __name__ == "__main__":
    main()
