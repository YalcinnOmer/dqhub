from __future__ import annotations

from .make_synth import main as make_synth_main
from .run_dq import main as run_dq_main
from .report import main as report_main
from .verify import main as verify_main


def main() -> int:
    code = make_synth_main(scenario="auto")
    if code != 0:
        return code

    code = run_dq_main()
    if code != 0:
        return code

    code = report_main()
    if code != 0:
        return code

    return verify_main()


if __name__ == "__main__":
    raise SystemExit(main())
