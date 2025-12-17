from __future__ import annotations

import argparse
import json
import random
import string
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pandas as pd
from faker import Faker


ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
OUTPUT_DIR = DATA_DIR / "output"
SYNTH_DIR = DATA_DIR / "synthetic"
REPORTS_DIR = ROOT / "reports"

DEFAULT_OUT = OUTPUT_DIR / "raw.csv"
DEFAULT_META = REPORTS_DIR / "last_synth_meta.json"


@dataclass(frozen=True)
class SynthConfig:
    rows: int
    seed: int
    profile: str
    missing_rate: float
    invalid_email_rate: float
    invalid_postal_rate: float
    duplicate_rate: float
    timeliness_outlier_rate: float
    range_violation_rate: float
    start_days_ago: int
    end_days_ago: int


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _run_id(now_utc: datetime) -> str:
    return now_utc.strftime("%Y%m%d_%H%M%S")


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def _choose_weighted(rng: random.Random, items: list[tuple[Any, float]]) -> Any:
    total = sum(w for _, w in items)
    r = rng.random() * total
    acc = 0.0
    for v, w in items:
        acc += w
        if r <= acc:
            return v
    return items[-1][0]


def _rand_str(rng: random.Random, n: int) -> str:
    alphabet = string.ascii_uppercase + string.digits
    return "".join(rng.choice(alphabet) for _ in range(n))


def _ca_postal(rng: random.Random) -> str:
    letters = "ABCEGHJKLMNPRSTVXY"
    return f"{rng.choice(letters)}{rng.randint(0,9)}{rng.choice(letters)} {rng.randint(0,9)}{rng.choice(letters)}{rng.randint(0,9)}"


def _us_zip(rng: random.Random) -> str:
    return f"{rng.randint(10000, 99999)}"


def _pick_country_region(rng: random.Random) -> tuple[str, str, str]:
    country = _choose_weighted(rng, [("CA", 0.7), ("US", 0.3)])
    if country == "CA":
        province = _choose_weighted(
            rng,
            [
                ("ON", 0.45),
                ("QC", 0.22),
                ("BC", 0.13),
                ("AB", 0.10),
                ("MB", 0.05),
                ("NS", 0.05),
            ],
        )
        city = _choose_weighted(
            rng,
            [
                ("Toronto", 0.35),
                ("Montreal", 0.18),
                ("Vancouver", 0.12),
                ("Calgary", 0.10),
                ("Ottawa", 0.08),
                ("Winnipeg", 0.07),
                ("Halifax", 0.05),
                ("Edmonton", 0.05),
            ],
        )
        return country, province, city
    else:
        state = _choose_weighted(
            rng,
            [
                ("NY", 0.20),
                ("CA", 0.18),
                ("TX", 0.16),
                ("FL", 0.12),
                ("IL", 0.10),
                ("WA", 0.08),
                ("MA", 0.08),
                ("PA", 0.08),
            ],
        )
        city = _choose_weighted(
            rng,
            [
                ("New York", 0.22),
                ("Los Angeles", 0.16),
                ("Houston", 0.12),
                ("Miami", 0.10),
                ("Chicago", 0.10),
                ("Seattle", 0.10),
                ("Boston", 0.10),
                ("Philadelphia", 0.10),
            ],
        )
        return country, state, city


def _profile_adjustments(cfg: SynthConfig) -> SynthConfig:
    p = cfg.profile.lower().strip()

    missing = cfg.missing_rate
    inv_email = cfg.invalid_email_rate
    inv_postal = cfg.invalid_postal_rate
    dup = cfg.duplicate_rate
    time_out = cfg.timeliness_outlier_rate
    range_v = cfg.range_violation_rate

    if p == "baseline":
        pass
    elif p == "missing_spike":
        missing = _clamp(missing * 4.0, 0.0, 0.25)
    elif p == "dup_spike":
        dup = _clamp(dup * 6.0, 0.0, 0.20)
    elif p == "timeliness_spike":
        time_out = _clamp(time_out * 6.0, 0.0, 0.20)
    elif p == "validity_spike":
        inv_email = _clamp(inv_email * 5.0, 0.0, 0.20)
        inv_postal = _clamp(inv_postal * 5.0, 0.0, 0.20)
        range_v = _clamp(range_v * 4.0, 0.0, 0.20)

    return SynthConfig(
        rows=cfg.rows,
        seed=cfg.seed,
        profile=cfg.profile,
        missing_rate=missing,
        invalid_email_rate=inv_email,
        invalid_postal_rate=inv_postal,
        duplicate_rate=dup,
        timeliness_outlier_rate=time_out,
        range_violation_rate=range_v,
        start_days_ago=cfg.start_days_ago,
        end_days_ago=cfg.end_days_ago,
    )


def _inject_missing(rng: random.Random, df: pd.DataFrame, cols: list[str], rate: float) -> None:
    if rate <= 0:
        return
    n = len(df)
    for c in cols:
        mask = [rng.random() < rate for _ in range(n)]
        df.loc[mask, c] = None


def _inject_invalid_emails(rng: random.Random, df: pd.DataFrame, rate: float) -> None:
    if rate <= 0 or "email" not in df.columns:
        return
    n = len(df)
    mask = [rng.random() < rate for _ in range(n)]
    bad_values = ["invalid_email", "no-at-symbol.com", "user@@domain.com", "user@", "@domain.com"]
    df.loc[mask, "email"] = [rng.choice(bad_values) for _ in range(sum(mask))]


def _inject_invalid_postal(rng: random.Random, df: pd.DataFrame, rate: float) -> None:
    if rate <= 0 or "postal_code" not in df.columns:
        return
    n = len(df)
    mask = [rng.random() < rate for _ in range(n)]
    df.loc[mask, "postal_code"] = [_rand_str(rng, 6) for _ in range(sum(mask))]


def _inject_range_violations(rng: random.Random, df: pd.DataFrame, rate: float) -> None:
    if rate <= 0:
        return
    n = len(df)

    def _mask() -> list[bool]:
        return [rng.random() < rate for _ in range(n)]

    if "quantity" in df.columns:
        m = _mask()
        df.loc[m, "quantity"] = [rng.choice([0, -1, -3]) for _ in range(sum(m))]

    if "discount_pct" in df.columns:
        m = _mask()
        df.loc[m, "discount_pct"] = [rng.choice([0.85, 0.95, 1.10]) for _ in range(sum(m))]

    if "unit_price" in df.columns:
        m = _mask()
        df.loc[m, "unit_price"] = [rng.choice([-5.0, -1.0]) for _ in range(sum(m))]


def _inject_timeliness_outliers(rng: random.Random, df: pd.DataFrame, rate: float) -> None:
    if rate <= 0:
        return
    if "order_ts_utc" not in df.columns or "ingested_ts_utc" not in df.columns:
        return

    # Defensive: ensure datetime dtype before timedelta operations
    df["order_ts_utc"] = pd.to_datetime(df["order_ts_utc"], utc=True, errors="coerce")
    df["ingested_ts_utc"] = pd.to_datetime(df["ingested_ts_utc"], utc=True, errors="coerce")

    n = len(df)
    mask = [rng.random() < rate for _ in range(n)]
    idx = df.index[mask]

    for i in idx:
        order_ts = df.at[i, "order_ts_utc"]
        if pd.isna(order_ts):
            continue
        if rng.random() < 0.7:
            df.at[i, "ingested_ts_utc"] = order_ts + timedelta(days=rng.randint(5, 20))
        else:
            df.at[i, "ingested_ts_utc"] = order_ts - timedelta(hours=rng.randint(1, 48))


def _inject_duplicates(rng: random.Random, df: pd.DataFrame, rate: float) -> pd.DataFrame:
    if rate <= 0 or "order_id" not in df.columns:
        return df
    n = len(df)
    dup_count = int(n * rate)
    if dup_count <= 0:
        return df

    sample_idx = rng.sample(range(n), k=min(dup_count, n))
    dup_rows = df.iloc[sample_idx].copy()

    dup_rows["duplicate_group_id"] = [f"dup_{i:04d}" for i in range(len(dup_rows))]
    df["duplicate_group_id"] = None

    out = pd.concat([df, dup_rows], ignore_index=True)
    out = out.sample(frac=1.0, random_state=rng.randint(0, 10**9)).reset_index(drop=True)
    return out


def generate_orders(cfg: SynthConfig, out_csv: Path, meta_json: Path) -> None:
    now = _utc_now()
    run_id = _run_id(now)

    cfg = _profile_adjustments(cfg)

    rng = random.Random(cfg.seed)
    fake = Faker()
    Faker.seed(cfg.seed)

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    meta_json.parent.mkdir(parents=True, exist_ok=True)
    SYNTH_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    products = [
        ("NOZ-D28-H15-M11", "Laser Nozzle", 4.19),
        ("WIN-D18-T5-1090", "Protective Window", 8.99),
        ("CER-HLD-32", "Ceramic Holder", 12.49),
        ("LEN-D20-F50", "Lens", 24.99),
        ("RNG-CER-28", "Ceramic Ring", 6.49),
        ("CAP-NOZ-SET10", "Nozzle Cap Pack", 19.99),
    ]

    channels = [("web", 0.55), ("ebay", 0.30), ("amazon", 0.15)]
    devices = [("desktop", 0.52), ("mobile", 0.40), ("tablet", 0.08)]
    payments = [("card", 0.62), ("paypal", 0.28), ("bank_transfer", 0.10)]
    ship_methods = [("standard", 0.68), ("express", 0.22), ("pickup", 0.10)]
    statuses = [("paid", 0.45), ("shipped", 0.35), ("delivered", 0.15), ("refunded", 0.05)]

    start = now - timedelta(days=cfg.start_days_ago)
    end = now - timedelta(days=cfg.end_days_ago)
    if end <= start:
        end = start + timedelta(days=1)

    rows: list[dict[str, Any]] = []

    for i in range(cfg.rows):
        order_id = f"ORD-{run_id}-{i:06d}"
        customer_id = f"CUS-{rng.randint(100000, 999999)}"

        country, region, city = _pick_country_region(rng)
        postal_code = _ca_postal(rng) if country == "CA" else _us_zip(rng)

        channel = _choose_weighted(rng, channels)
        device_type = _choose_weighted(rng, devices)
        payment_type = _choose_weighted(rng, payments)
        shipping_method = _choose_weighted(rng, ship_methods)
        order_status = _choose_weighted(rng, statuses)

        order_ts = start + timedelta(seconds=rng.randint(0, int((end - start).total_seconds())))
        lag_hours = int(abs(rng.gauss(6, 4)))
        lag_hours = max(0, min(lag_hours, 72))
        ingested_ts = order_ts + timedelta(hours=lag_hours)

        sku, product_name, base_price = rng.choice(products)
        unit_price = round(max(0.5, rng.gauss(base_price, base_price * 0.10)), 2)
        quantity = rng.randint(1, 12)

        discount_pct = round(_clamp(abs(rng.gauss(0.05, 0.05)), 0.0, 0.35), 4)
        tax_pct = 0.13 if country == "CA" else 0.0
        shipping_fee = 0.0 if (channel == "web" and unit_price * quantity > 40) else round(_clamp(abs(rng.gauss(7.5, 3.0)), 0.0, 25.0), 2)

        subtotal = unit_price * quantity
        discount_amt = subtotal * discount_pct
        taxable = max(0.0, subtotal - discount_amt)
        tax_amt = taxable * tax_pct
        order_total = round(taxable + tax_amt + shipping_fee, 2)

        currency = "CAD" if country == "CA" else "USD"

        refund_flag = 1 if (order_status == "refunded") else (1 if rng.random() < 0.02 else 0)
        return_reason = None
        if refund_flag:
            return_reason = _choose_weighted(
                rng,
                [
                    ("damaged", 0.30),
                    ("wrong_item", 0.20),
                    ("late_delivery", 0.20),
                    ("changed_mind", 0.15),
                    ("quality_issue", 0.15),
                ],
            )

        est_delivery_days = int(_clamp(abs(rng.gauss(4 if shipping_method == "express" else 8, 2.0)), 1, 21))
        actual_delivery_days = est_delivery_days + int(rng.gauss(0, 2))
        actual_delivery_days = max(1, min(actual_delivery_days, 45))

        rows.append(
            {
                "order_id": order_id,
                "customer_id": customer_id,
                "sku": sku,
                "product_name": product_name,
                "channel": channel,
                "device_type": device_type,
                "payment_type": payment_type,
                "shipping_method": shipping_method,
                "order_status": order_status,
                "refund_flag": refund_flag,
                "return_reason": return_reason,
                "country": country,
                "region": region,
                "city": city,
                "postal_code": postal_code,
                "email": fake.email(),
                # IMPORTANT: keep datetime in-memory; convert to string at the end
                "order_ts_utc": order_ts,
                "ingested_ts_utc": ingested_ts,
                "unit_price": unit_price,
                "quantity": quantity,
                "discount_pct": discount_pct,
                "tax_pct": tax_pct,
                "shipping_fee": shipping_fee,
                "currency": currency,
                "order_total": order_total,
                "estimated_delivery_days": est_delivery_days,
                "actual_delivery_days": actual_delivery_days,
            }
        )

    df = pd.DataFrame(rows)

    _inject_missing(
        rng,
        df,
        cols=["city", "postal_code", "email", "payment_type", "shipping_method", "device_type", "return_reason"],
        rate=cfg.missing_rate,
    )
    _inject_invalid_emails(rng, df, cfg.invalid_email_rate)
    _inject_invalid_postal(rng, df, cfg.invalid_postal_rate)
    _inject_range_violations(rng, df, cfg.range_violation_rate)
    _inject_timeliness_outliers(rng, df, cfg.timeliness_outlier_rate)
    df = _inject_duplicates(rng, df, cfg.duplicate_rate)

    # Convert timestamps to ISO-like strings for CSV
    for ts_col in ("order_ts_utc", "ingested_ts_utc"):
        df[ts_col] = pd.to_datetime(df[ts_col], utc=True, errors="coerce").dt.strftime("%Y-%m-%dT%H:%M:%S%z")

    df.to_csv(out_csv, index=False)

    synth_copy = SYNTH_DIR / out_csv.name
    df.to_csv(synth_copy, index=False)

    meta = {
        "run_id": run_id,
        "generated_at_utc": now.isoformat(),
        "output_csv": str(out_csv),
        "synthetic_copy_csv": str(synth_copy),
        "rows": int(len(df)),
        "config": asdict(cfg),
        "schema": list(df.columns),
    }
    meta_json.write_text(json.dumps(meta, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate synthetic orders dataset (dashboard-ready) with controlled DQ issues.")
    p.add_argument("--rows", type=int, default=1500)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--profile",
        type=str,
        default="baseline",
        choices=["baseline", "missing_spike", "dup_spike", "timeliness_spike", "validity_spike"],
    )

    p.add_argument("--missing-rate", type=float, default=0.01)
    p.add_argument("--invalid-email-rate", type=float, default=0.008)
    p.add_argument("--invalid-postal-rate", type=float, default=0.006)
    p.add_argument("--duplicate-rate", type=float, default=0.004)
    p.add_argument("--timeliness-outlier-rate", type=float, default=0.004)
    p.add_argument("--range-violation-rate", type=float, default=0.004)

    p.add_argument("--start-days-ago", type=int, default=90)
    p.add_argument("--end-days-ago", type=int, default=0)

    p.add_argument("--out", type=str, default=str(DEFAULT_OUT))
    p.add_argument("--meta", type=str, default=str(DEFAULT_META))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = SynthConfig(
        rows=args.rows,
        seed=args.seed,
        profile=args.profile,
        missing_rate=float(args.missing_rate),
        invalid_email_rate=float(args.invalid_email_rate),
        invalid_postal_rate=float(args.invalid_postal_rate),
        duplicate_rate=float(args.duplicate_rate),
        timeliness_outlier_rate=float(args.timeliness_outlier_rate),
        range_violation_rate=float(args.range_violation_rate),
        start_days_ago=int(args.start_days_ago),
        end_days_ago=int(args.end_days_ago),
    )

    out_csv = Path(args.out)
    meta_json = Path(args.meta)

    generate_orders(cfg, out_csv=out_csv, meta_json=meta_json)

    print(f"Wrote: {out_csv} | exists: {out_csv.exists()}")
    print(f"Wrote: {meta_json} | exists: {meta_json.exists()}")


if __name__ == "__main__":
    main()
