from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Literal
import random

import pandas as pd
from faker import Faker


ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
SYNTH_DIR = DATA_DIR / "synthetic"
OUTPUT_DIR = DATA_DIR / "output"
REPORTS_DIR = ROOT / "reports"

INPUT_CSV = SYNTH_DIR / "input.csv"
RAW_CSV = OUTPUT_DIR / "raw.csv"
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
    # First run: baseline (inject bad records); next runs: improved (fewer issues)
    return "improved" if SUMMARY_CSV.exists() else "baseline"


def _pick_countries(rng: random.Random, n: int) -> list[str]:
    pool = ["CA", "US", "GB", "DE", "FR", "NL", "SE", "TR"]
    return [rng.choice(pool) for _ in range(n)]


def _build_df(cfg: SynthConfig) -> pd.DataFrame:
    rng = random.Random(cfg.seed)
    faker = Faker()
    Faker.seed(cfg.seed)
    now_utc = datetime.now(timezone.utc).replace(microsecond=0)

    order_ids = list(range(1, cfg.rows + 1))
    emails = [faker.email() for _ in range(cfg.rows)]
    countries = _pick_countries(rng, cfg.rows)
    amounts = [round(rng.uniform(5.0, 950.0), 2) for _ in range(cfg.rows)]

    order_ts = [
        now_utc - timedelta(days=rng.randint(0, 35), minutes=rng.randint(0, 1440))
        for _ in range(cfg.rows)
    ]
    ingest_ts = [t + timedelta(minutes=rng.randint(0, 240)) for t in order_ts]

    df = pd.DataFrame(
        {
            "order_id": order_ids,
            "email": emails,
            "country": countries,
            "amount": amounts,
            "order_ts_utc": order_ts,
            "ingest_ts_utc": ingest_ts,
        }
    )

    if cfg.scenario == "baseline":
        # Controlled "badness" for demo value
        bad_idx = rng.sample(range(cfg.rows), k=min(6, cfg.rows))

        for i in bad_idx[:3]:
            df.loc[i, "ingest_ts_utc"] = pd.NaT

        df.loc[bad_idx[3], "email"] = "invalid-email"
        df.loc[bad_idx[4], "amount"] = -round(rng.uniform(10.0, 900.0), 2)
        df.loc[bad_idx[5], "country"] = ""

        if cfg.rows >= 30:
            df.loc[cfg.rows - 3, "order_id"] = 10
            df.loc[cfg.rows - 2, "order_id"] = 20
            df.loc[cfg.rows - 1, "order_id"] = 30

        late_idx = rng.sample(range(cfg.rows), k=min(3, cfg.rows))
        for i in late_idx:
            if pd.notna(df.loc[i, "ingest_ts_utc"]):
                df.loc[i, "ingest_ts_utc"] = df.loc[i, "order_ts_utc"] + timedelta(minutes=900)

    df["order_ts_utc"] = pd.to_datetime(df["order_ts_utc"], utc=True)
    df["ingest_ts_utc"] = pd.to_datetime(df["ingest_ts_utc"], utc=True)
    return df


def main(rows: int = 522, seed: int = 42, scenario: str | None = None) -> int:
    _ensure_dirs()

    if scenario is None or scenario.strip().lower() == "auto":
        scen: Scenario = _auto_scenario()
    else:
        s = scenario.strip().lower()
        scen = "improved" if s == "improved" else "baseline"

    cfg = SynthConfig(rows=int(rows), seed=int(seed), scenario=scen)
    df = _build_df(cfg)

    df.to_csv(INPUT_CSV, index=False, encoding="utf-8")
    print(f"Wrote: {INPUT_CSV} | exists: {INPUT_CSV.exists()}")

    RAW_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(RAW_CSV, index=False, encoding="utf-8")
    print(f"Wrote: {RAW_CSV} | exists: {RAW_CSV.exists()}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
