from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pandas as pd
from faker import Faker

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
SYNTH_DIR = DATA_DIR / "synthetic"
OUTPUT_DIR = DATA_DIR / "output"
REPORTS_DIR = ROOT / "reports"

INPUT_CSV = SYNTH_DIR / "input.csv"
RAW_CSV = OUTPUT_DIR / "raw.csv"
SYNTH_META_JSON = REPORTS_DIR / "last_synth_meta.json"


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _run_id(ts: datetime) -> str:
    return ts.strftime("%Y%m%d_%H%M%S")


@dataclass(frozen=True)
class SynthConfig:
    n_rows: int = 520
    seed: int = 42

    # Anomaly rates (keep small; this is synthetic but meant to look realistic)
    dup_id_rate: float = 0.004     # 0.4%
    bad_email_rate: float = 0.006  # 0.6%
    bad_country_rate: float = 0.004
    neg_amount_rate: float = 0.003
    late_ingest_rate: float = 0.008  # >24h latency


def _ensure_dirs() -> None:
    SYNTH_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def _env_int(name: str, default: int) -> int:
    v = os.getenv(name)
    if not v:
        return default
    try:
        return int(v)
    except Exception:
        return default


def _build_orders(cfg: SynthConfig) -> pd.DataFrame:
    faker = Faker()
    Faker.seed(cfg.seed)

    countries = ["US", "CA", "DE", "TR"]
    currency_by_country = {"US": "USD", "CA": "CAD", "DE": "EUR", "TR": "TRY"}

    base_ts = _utc_now()

    rows: list[dict[str, Any]] = []

    # Pre-generate some IDs so we can intentionally duplicate a few.
    ids = [str(i + 1) for i in range(cfg.n_rows)]
    n_dups = max(0, int(round(cfg.n_rows * cfg.dup_id_rate)))
    dup_source_ids = ids[:n_dups] if n_dups > 0 else []

    for i in range(cfg.n_rows):
        order_dt = base_ts - timedelta(days=faker.random_int(min=0, max=14), hours=faker.random_int(min=0, max=23))

        latency_minutes = faker.random_int(min=1, max=12 * 60)
        ingest_dt = order_dt + timedelta(minutes=int(latency_minutes))

        country = faker.random_element(elements=countries)
        currency = currency_by_country.get(country, "USD")

        amount = round(float(faker.pydecimal(left_digits=3, right_digits=2, positive=True, min_value=5, max_value=800)), 2)

        email = faker.email()

        row_id = ids[i]
        rows.append(
            {
                "id": row_id,
                "email": email,
                "country": country,
                "currency": currency,
                "amount": amount,
                "order_ts_utc": order_dt.isoformat(),
                "ingested_ts_utc": ingest_dt.isoformat(),
            }
        )

    # Inject anomalies (deterministic via seed)
    rng = faker.random

    def pick_indices(rate: float) -> list[int]:
        k = max(0, int(round(cfg.n_rows * rate)))
        if k <= 0:
            return []
        return rng.sample(range(cfg.n_rows), k=min(k, cfg.n_rows))

    # Duplicate IDs
    if dup_source_ids:
        target_idxs = pick_indices(cfg.dup_id_rate)
        for j, idx in enumerate(target_idxs):
            rows[idx]["id"] = dup_source_ids[j % len(dup_source_ids)]

    # Bad email
    for idx in pick_indices(cfg.bad_email_rate):
        rows[idx]["email"] = "not-an-email"

    # Bad country
    for idx in pick_indices(cfg.bad_country_rate):
        rows[idx]["country"] = "USA"  # should be 2-letter

    # Negative amount
    for idx in pick_indices(cfg.neg_amount_rate):
        rows[idx]["amount"] = -abs(float(rows[idx]["amount"]))

    # Late ingest (>24h)
    for idx in pick_indices(cfg.late_ingest_rate):
        try:
            o = datetime.fromisoformat(rows[idx]["order_ts_utc"])
            rows[idx]["ingested_ts_utc"] = (o + timedelta(hours=36)).isoformat()
        except Exception:
            pass

    return pd.DataFrame(rows)


def main() -> None:
    _ensure_dirs()

    cfg = SynthConfig(
        n_rows=_env_int("DQHUB_N", 520),
        seed=_env_int("DQHUB_SEED", 42),
    )

    ts = _utc_now()
    rid = _run_id(ts)

    df = _build_orders(cfg)

    # IMPORTANT: CI expects this path to exist
    df.to_csv(INPUT_CSV, index=False)
    print(f"Wrote: {INPUT_CSV} | exists: {INPUT_CSV.exists()}")

    # Raw is a copy (acts like “landing zone”)
    df.to_csv(RAW_CSV, index=False)
    print(f"Wrote: {RAW_CSV} | exists: {RAW_CSV.exists()}")

    meta = {
        "run_id": rid,
        "created_ts_utc": ts.isoformat(),
        "seed": cfg.seed,
        "rows": int(len(df)),
        "paths": {
            "input_csv": str(INPUT_CSV.relative_to(ROOT)).replace("\\", "/"),
            "raw_csv": str(RAW_CSV.relative_to(ROOT)).replace("\\", "/"),
        },
        "anomaly_rates": {
            "dup_id_rate": cfg.dup_id_rate,
            "bad_email_rate": cfg.bad_email_rate,
            "bad_country_rate": cfg.bad_country_rate,
            "neg_amount_rate": cfg.neg_amount_rate,
            "late_ingest_rate": cfg.late_ingest_rate,
        },
    }

    SYNTH_META_JSON.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Wrote: {SYNTH_META_JSON} | exists: {SYNTH_META_JSON.exists()}")


if __name__ == "__main__":
    main()
