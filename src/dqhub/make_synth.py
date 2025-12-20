from __future__ import annotations

import argparse
import random
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Literal

import pandas as pd
from faker import Faker

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
SYNTH_DIR = DATA_DIR / "synthetic"
OUTPUT_DIR = DATA_DIR / "output"
REPORTS_DIR = ROOT / "reports"

INPUT_CSV = SYNTH_DIR / "input.csv"
RAW_CSV = OUTPUT_DIR / "raw.csv"

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


def _base_frame(cfg: SynthConfig) -> pd.DataFrame:
    rng = random.Random(cfg.seed)
    fake = Faker()
    Faker.seed(cfg.seed)

    now = datetime.now(timezone.utc).replace(microsecond=0)
    countries = ["US", "CA", "TR", "DE", "NL", "FR", "GB", "SE"]

    n_good = max(cfg.rows - 3, 1)
    order_ids = list(range(1, n_good + 1))

    rows: list[dict] = []
    for oid in order_ids:
        order_ts = now - timedelta(days=rng.randint(1, 30), minutes=rng.randint(0, 1440))
        delay = timedelta(minutes=rng.randint(1, 240))
        ingest_ts = order_ts + delay
        rows.append(
            {
                "order_id": oid,
                "email": fake.email(),
                "country": rng.choice(countries),
                "amount": round(rng.uniform(10, 1000), 2),
                "order_ts_utc": order_ts.isoformat(),
                "ingest_ts_utc": ingest_ts.isoformat(),
            }
        )

    return pd.DataFrame(rows)


def _inject_baseline_issues(df: pd.DataFrame, cfg: SynthConfig) -> pd.DataFrame:
    rng = random.Random(cfg.seed + 1)
    out = df.copy()

    if len(out) > 0:
        bad_idxs = rng.sample(list(out.index), k=min(5, len(out)))
        for i, idx in enumerate(bad_idxs):
            if i % 2 == 0:
                out.at[idx, "email"] = "invalid-email"
            else:
                out.at[idx, "amount"] = -abs(float(out.at[idx, "amount"]))

    now = datetime.now(timezone.utc).replace(microsecond=0)
    extra = pd.DataFrame(
        [
            {
                "order_id": 10,
                "email": "",
                "country": "TR",
                "amount": round(rng.uniform(10, 1000), 2),
                "order_ts_utc": (now - timedelta(days=10)).isoformat(),
                "ingest_ts_utc": "",
            },
            {
                "order_id": 20,
                "email": "",
                "country": "US",
                "amount": round(rng.uniform(10, 1000), 2),
                "order_ts_utc": (now - timedelta(days=8)).isoformat(),
                "ingest_ts_utc": "",
            },
            {
                "order_id": 30,
                "email": "",
                "country": "GB",
                "amount": round(rng.uniform(10, 1000), 2),
                "order_ts_utc": (now - timedelta(days=6)).isoformat(),
                "ingest_ts_utc": "",
            },
        ]
    )

    out2 = pd.concat([out, extra], ignore_index=True)
    return out2.head(cfg.rows)


def _inject_improved_issues(df: pd.DataFrame, cfg: SynthConfig) -> pd.DataFrame:
    rng = random.Random(cfg.seed + 2)
    out = df.copy()

    if len(out) > 0:
        bad_idxs = rng.sample(list(out.index), k=min(2, len(out)))
        for idx in bad_idxs:
            out.at[idx, "email"] = "invalid-email"

    now = datetime.now(timezone.utc).replace(microsecond=0)
    extra = pd.DataFrame(
        [
            {
                "order_id": 10,
                "email": "",
                "country": "TR",
                "amount": round(rng.uniform(10, 1000), 2),
                "order_ts_utc": (now - timedelta(days=10)).isoformat(),
                "ingest_ts_utc": "",
            }
        ]
    )

    out2 = pd.concat([out, extra], ignore_index=True)
    if len(out2) < cfg.rows:
        fill = out.sample(n=cfg.rows - len(out2), replace=True, random_state=cfg.seed)
        out2 = pd.concat([out2, fill], ignore_index=True)

    return out2.head(cfg.rows)


def build_synth(cfg: SynthConfig) -> pd.DataFrame:
    _ensure_dirs()
    base = _base_frame(cfg)
    if cfg.scenario == "improved":
        return _inject_improved_issues(base, cfg)
    return _inject_baseline_issues(base, cfg)


def write_outputs(df: pd.DataFrame) -> None:
    _ensure_dirs()
    df.to_csv(INPUT_CSV, index=False)
    print(f"Wrote: {INPUT_CSV} | exists: {INPUT_CSV.exists()}")
    df.to_csv(RAW_CSV, index=False)
    print(f"Wrote: {RAW_CSV} | exists: {RAW_CSV.exists()}")


def _parse_args(argv: list[str]) -> SynthConfig:
    parser = argparse.ArgumentParser(prog="dqhub-make-synth")
    parser.add_argument("--rows", type=int, default=522)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--scenario", choices=["baseline", "improved"], default="baseline")
    ns = parser.parse_args(argv)
    return SynthConfig(rows=ns.rows, seed=ns.seed, scenario=ns.scenario)


def main(argv: list[str] | None = None) -> int:
    args = [] if argv is None else argv
    cfg = _parse_args(args)
    df = build_synth(cfg)
    write_outputs(df)
    return 0


if __name__ == "__main__":
    import sys

    raise SystemExit(main(sys.argv[1:]))
