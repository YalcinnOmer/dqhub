from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Literal, Optional
import random

import pandas as pd
from faker import Faker

ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = ROOT / "data"
SYNTH_DIR = DATA_DIR / "synthetic"
OUTPUT_DIR = DATA_DIR / "output"
REPORTS_DIR = ROOT / "reports"

INPUT_CSV = SYNTH_DIR / "input.csv"
SUMMARY_CSV = REPORTS_DIR / "dq_summary.csv"


Scenario = Literal["baseline", "improved"]


@dataclass(frozen=True)
class SynthConfig:
    rows: int = 522
    seed: int = 42
    scenario: Scenario = "baseline"


def _ensure_dirs() -> None:
    SYNTH_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def _auto_scenario() -> Scenario:
    """
    If summary exists (>=1 run), generate the improved dataset; otherwise baseline.
    This gives a nice 'trend' story for portfolio screenshots.
    """
    if not SUMMARY_CSV.exists():
        return "baseline"
    try:
        df = pd.read_csv(SUMMARY_CSV)
        return "improved" if len(df) >= 1 else "baseline"
    except Exception:
        return "baseline"


def _rand_country(rng: random.Random) -> str:
    return rng.choice(["US", "CA", "DE", "TR", "GB", "FR", "NL", "SE"])


def _make_row(
    *,
    order_id: int,
    faker: Faker,
    rng: random.Random,
    now_utc: datetime,
    allow_missing_ingest: bool,
) -> dict:
    order_ts = now_utc - timedelta(days=rng.randint(1, 30), hours=rng.randint(0, 23), minutes=rng.randint(0, 59))
    ingest_ts = order_ts + timedelta(minutes=rng.randint(5, 180))

    row = {
        "order_id": order_id,
        "email": faker.email(),
        "country": _rand_country(rng),
        "amount": round(rng.uniform(10.0, 900.0), 2),
        "order_ts_utc": order_ts.replace(tzinfo=timezone.utc).isoformat(),
        "ingest_ts_utc": ingest_ts.replace(tzinfo=timezone.utc).isoformat(),
    }

    if allow_missing_ingest:
        row["ingest_ts_utc"] = ""

    return row


def build_synth(config: Optional[SynthConfig] = None) -> Path:
    _ensure_dirs()

    if config is None:
        config = SynthConfig(scenario=_auto_scenario())

    rng = random.Random(config.seed)
    faker = Faker()
    Faker.seed(config.seed)

    now_utc = datetime.now(timezone.utc).replace(microsecond=0)

    rows: list[dict] = []

    if config.scenario == "baseline":
        good_unique = 519
        for oid in range(1, good_unique + 1):
            rows.append(_make_row(order_id=oid, faker=faker, rng=rng, now_utc=now_utc, allow_missing_ingest=False))

        dup_ids = [10, 20, 30]
        for oid in dup_ids:
            rows.append(_make_row(order_id=oid, faker=faker, rng=rng, now_utc=now_utc, allow_missing_ingest=True))

        rows = rows[: config.rows]

    else:

        unique = 520
        for oid in range(1, unique + 1):
            rows.append(_make_row(order_id=oid, faker=faker, rng=rng, now_utc=now_utc, allow_missing_ingest=False))

        # Add 2 valid duplicates
        dup_ids = [111, 222]
        for oid in dup_ids:
            rows.append(_make_row(order_id=oid, faker=faker, rng=rng, now_utc=now_utc, allow_missing_ingest=False))

        rows = rows[: config.rows]

    df = pd.DataFrame(rows)
    df.to_csv(INPUT_CSV, index=False, encoding="utf-8")
    return INPUT_CSV


def main() -> None:
    path = build_synth()
    print(f"Wrote: {path} | exists: {path.exists()}")


if __name__ == "__main__":
    main()
