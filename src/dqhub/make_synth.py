from pathlib import Path
from datetime import datetime, timezone, date
import random, csv, re

try:
    from faker import Faker
except ImportError:
    raise SystemExit("Missing dependency: faker. Install with: pip install faker")

ROOT = Path(__file__).resolve().parents[2]
SYNTH_DIR = ROOT / "data" / "synthetic"
OUT_CSV = SYNTH_DIR / "input.csv"
ROW_COUNT = 520
SEED = 42

def iso_utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")

def to_e164_ca(phone: str) -> str:
    digits = re.sub(r"\D", "", str(phone))[-10:].rjust(10, "0")
    return "+1" + digits

def build_email_from_name(name: str) -> str:
    base = name.lower().replace(" ", ".").replace("'", "")
    base = re.sub(r"[^a-z0-9._-]", "", base)
    base = re.sub(r"\.+", ".", base).strip(".")
    if not base:
        base = "user"
    return f"{base}@example.com"

def dirty_email(email: str) -> str:
    r = random.random()
    if r < 0.03:
        return email.replace("@", "_at_")
    if r < 0.06:
        return email.replace(".", "", 1)
    if r < 0.09:
        u, d = email.split("@")
        u = u[:-1] if len(u) > 1 else u
        return f"{u}@{d}"
    return email

def pick_birth_date(fake: "Faker") -> str:
    d = fake.date_between(start_date=date(1940, 1, 1), end_date=date(2010, 12, 31))
    return d.isoformat()

def main():
    random.seed(SEED)
    fake = Faker("en_CA")
    Faker.seed(SEED)
    SYNTH_DIR.mkdir(parents=True, exist_ok=True)
    fieldnames = ["id", "full_name", "email", "phone", "birth_date", "created_at"]
    with OUT_CSV.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for i in range(100, 100 + ROW_COUNT):
            full_name = fake.name()
            email = dirty_email(build_email_from_name(full_name))
            phone_raw = fake.phone_number()
            phone = to_e164_ca(phone_raw)
            birth_date = pick_birth_date(fake)
            created_at = iso_utc_now()
            if random.random() < 0.03:
                phone = "12345"
            if random.random() < 0.02:
                created_at = "1990-01-01"
            w.writerow({
                "id": str(i),
                "full_name": full_name,
                "email": email,
                "phone": phone,
                "birth_date": birth_date,
                "created_at": created_at
            })
        dup_rows = [
            {"id": "150", "full_name": fake.name(), "email": "dup_user@example.com", "phone": to_e164_ca(fake.phone_number()), "birth_date": pick_birth_date(fake), "created_at": iso_utc_now()},
            {"id": "150", "full_name": fake.name(), "email": "dup.user@example.com", "phone": to_e164_ca(fake.phone_number()), "birth_date": pick_birth_date(fake), "created_at": iso_utc_now()}
        ]
        for r in dup_rows:
            w.writerow(r)
    print(f"Created: {OUT_CSV} | rows: ~{ROW_COUNT + len(dup_rows)} | exists: {OUT_CSV.exists()}")

if __name__ == "__main__":
    main()
