from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pandas as pd


try:
    from faker import Faker
except Exception as e:  # pragma: no cover
    raise RuntimeError(
        "Faker is required for synthetic data generation. "
        "Install it with: pip install faker"
    ) from e


ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = ROOT / "data"
OUTPUT_DIR = DATA_DIR / "output"
SYNTH_DIR = DATA_DIR / "synthetic"
REPORTS_DIR = ROOT / "reports"

OUT_RAW_CSV = OUTPUT_DIR / "raw.csv"
OUT_SYNTH_INPUT_CSV = SYNTH_DIR / "input.csv"
OUT_META_JSON = REPORTS_DIR / "last_synth_meta.json"


@dataclass(frozen=True)
class SynthMeta:
    generated_at_utc: str
    rows: int
    seed: int
    dup_rate: float
    dup_count: int
    columns: list[str]
    outputs: dict[str, str]


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _ensure_dirs() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    SYNTH_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def _iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).isoformat()


def build_synthetic_orders(
    rows: int = 522,
    seed: int = 42,
    dup_rate: float = 0.004,
) -> tuple[pd.DataFrame, SynthMeta]:
    """
    Generate an orders-like dataset with columns aligned to downstream checks:
      - id (for uniqueness rules)
      - customer_email (validity: email)
      - country (validity: country code)
      - currency (validity: currency code)
      - amount (validity: non-negative numeric)
      - order_ts + ingested_ts (timeliness pair)

    dup_rate injects a small fraction of duplicated ids (for Uniqueness demo).
    """
    if rows <= 0:
        raise ValueError("rows must be > 0")

    if seed < 0:
        raise ValueError("seed must be >= 0")

    if dup_rate < 0.0 or dup_rate > 1.0:
        raise ValueError("dup_rate must be between 0 and 1")

    faker = Faker()
    faker.seed_instance(seed)

    # Keep values realistic but stable
    countries = ["CA", "US", "DE", "TR", "GB", "FR", "NL", "IT", "ES", "SE"]
    currencies = {"CA": "CAD", "US": "USD", "DE": "EUR", "TR": "TRY", "GB": "GBP", "FR": "EUR", "NL": "EUR", "IT": "EUR", "ES": "EUR", "SE": "SEK"}

    start = _utc_now() - timedelta(days=30)

    ids = list(range(1, rows + 1))

    # Inject duplicates by overwriting some ids (keeps row count constant)
    dup_count = int(round(rows * dup_rate))
    dup_count = max(0, min(rows - 1, dup_count))

    if dup_count > 0:
        # Choose indices from 1..rows-1 so we can copy a previous id deterministically
        # Faker's random_int is deterministic given the seed.
        chosen = set()
        while len(chosen) < dup_count:
            chosen.add(faker.random_int(min=1, max=rows - 1))
        for idx in sorted(chosen):
            ids[idx] = ids[idx - 1]

    records: list[dict[str, Any]] = []
    for i in range(rows):
        order_dt = start + timedelta(minutes=int(faker.random_int(min=0, max=30 * 24 * 60 - 1)))
        # Keep ingestion within 0..12h so timeliness can easily pass typical rules
        ingest_dt = order_dt + timedelta(minutes=int(faker.random_int(min=0, max=12 * 60)))

        country = faker.random_element(elements=countries)
        currency = currencies.get(country, "USD")

        # Amount: positive, two decimals
        amount = round(float(faker.pydecimal(left_digits=3, right_digits=2, positive=True, min_value=5, max_value=999)), 2)

        records.append(
            {
                "id": ids[i],
                "customer_email": faker.email(),
                "country": country,
                "currency": currency,
                "amount": amount,
                "order_ts": _iso(order_dt),
                "ingested_ts": _iso(ingest_dt),
            }
        )

    df = pd.DataFrame.from_records(records)

    cols = ["id", "customer_email", "country", "currency", "amount", "order_ts", "ingested_ts"]
    df = df[cols]

    meta = SynthMeta(
        generated_at_utc=_iso(_utc_now()),
        rows=rows,
        seed=seed,
        dup_rate=float(dup_rate),
        dup_count=int(dup_count),
        columns=cols,
        outputs={
            "raw_csv": str(OUT_RAW_CSV.relative_to(ROOT)).replace("\\", "/"),
            "synth_input_csv": str(OUT_SYNTH_INPUT_CSV.relative_to(ROOT)).replace("\\", "/"),
            "meta_json": str(OUT_META_JSON.relative_to(ROOT)).replace("\\", "/"),
        },
    )

    return df, meta


def write_outputs(df: pd.DataFrame, meta: SynthMeta) -> None:
    _ensure_dirs()

    df.to_csv(OUT_SYNTH_INPUT_CSV, index=False)
    print(f"Wrote: {OUT_SYNTH_INPUT_CSV} | exists: {OUT_SYNTH_INPUT_CSV.exists()}")

    df.to_csv(OUT_RAW_CSV, index=False)
    print(f"Wrote: {OUT_RAW_CSV} | exists: {OUT_RAW_CSV.exists()}")

    OUT_META_JSON.write_text(json.dumps(asdict(meta), indent=2), encoding="utf-8")
    print(f"Wrote: {OUT_META_JSON} | exists: {OUT_META_JSON.exists()}")


def main(rows: int = 522, seed: int = 42, dup_rate: float = 0.004) -> None:
    df, meta = build_synthetic_orders(rows=rows, seed=seed, dup_rate=dup_rate)
    write_outputs(df, meta)


if __name__ == "__main__":
    # Lightweight CLI via environment-free defaults.
    # If you want parameters: run with a small wrapper or call main(rows=..., seed=..., dup_rate=...)
    main()
