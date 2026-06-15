"""
Generate a synthetic raw MDM_Population file for end-to-end pipeline testing.

The output matches the production raw schema exactly (19 object columns) and is
intentionally *messy* in the same ways catalogued in docs/Data-Cleaning-Guide.md
so the cleaning stage has real work to do:
    - mixed case, extra whitespace, CamelCase joins, accented/unicode letters
    - name titles (MR/DR) and trailing generational suffixes (JR/SR/III)
    - varied SSN / phone / DOB / ZIP / state formatting
    - text-based nulls (UNKNOWN, N/A, "") and address placeholders (HOMELESS)
    - junk values the cleaner must nullify (junk SSNs/phones/emails/ZIPs)
    - invalid-marker records (DO NOT USE, BABY BOY, TEST) → valid_record=False

It also PLANTS true duplicate pairs that are detectable end-to-end. Each
duplicate shares the *canonical* values a given deterministic rule needs, but
rendered with different messy formatting — so the two records still clean to the
same value and survive blocking + rule matching. Scenarios cover every rule:
EXACT_SSN, EMAIL_EXACT, NAME_DOB_PHONE, NAME_DOB_EMAIL, NAME_DOB_ADDRESS,
NAME_DOB_SEX.

USAGE:
    python scripts/generate_synthetic_raw.py
    python scripts/generate_synthetic_raw.py --n 5000 --dup-rate 0.25 --seed 7

OUTPUT:
    data/raw/MDM_Population.csv  (overwritten on each run)
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT = _PROJECT_ROOT / "data" / "raw" / "MDM_Population.csv"

COLUMNS = [
    "PATID", "FirstNM", "LastNM", "MiddleNM", "SuffixNM", "BirthDT", "SSN",
    "AddressLine1", "AddressLine2", "CityNM", "ZipCD", "StateCD", "CountryNM",
    "PrimaryPhoneNBR", "Phone01NBR", "Phone02NBR", "Phone03NBR", "Email",
    "SexAtBirthDSC",
]

FIRST_NAMES = [
    "JAMES", "MARY", "ROBERT", "PATRICIA", "JOHN", "JENNIFER", "MICHAEL",
    "LINDA", "CARLOS", "MARIA", "LUIS", "SOFIA", "DAVID", "KAREN", "DIANE",
    "PRIYA", "LINH", "JOSE", "THOMAS", "ANNA", "PEDRO", "SARA", "ELIZABETH",
    "ANDRE", "RENEE",
]
LAST_NAMES = [
    "SMITH", "JOHNSON", "GARCIA", "MARTINEZ", "GUTIERREZ", "SCHMIDT",
    "SANCHEZ", "RODRIGUEZ", "THOMPSON", "WILSON", "NGUYEN", "PATEL",
    "GONZALEZ", "ROBINSON", "ANDERSON", "HERNANDEZ", "JONES", "HARRIS",
    "JACKSON", "OBRIEN",
]
MIDDLE_NAMES = ["LEE", "ANN", "MARIE", "JAMES", "RAE", "JO", "KUMAR"]
# (city, 2-letter state, 5-digit ZIP) — ZIP first digit is consistent w/ state.
PLACES = [
    ("CHICAGO", "IL", "60601"), ("EVANSTON", "IL", "60201"),
    ("AURORA", "IL", "60505"), ("HOUSTON", "TX", "77002"),
    ("DALLAS", "TX", "75201"), ("MIAMI", "FL", "33101"),
    ("ATLANTA", "GA", "30303"), ("NEW YORK", "NY", "10001"),
    ("BOSTON", "MA", "02108"), ("DENVER", "CO", "80202"),
]
# (canonical-after-cleaning street, raw-spelled-out variant for messiness)
STREETS = [
    ("MAIN ST", "Main Street"), ("OAK AVE", "Oak Avenue"),
    ("MAPLE DR", "Maple Drive"), ("LINCOLN RD", "Lincoln Road"),
    ("PARK LN", "Park Lane"), ("ELM CT", "Elm Court"),
    ("CEDAR PL", "Cedar Place"), ("HILL BLVD", "Hill Boulevard"),
]

TEXT_NULLS = ["UNKNOWN", "NULL", "N/A", "NA", "NONE", ""]
ADDRESS_PLACEHOLDERS = ["HOMELESS", "NO ADDRESS", "TRANSIENT", "BAD ADDRESS", "?"]
INVALID_MARKERS = [
    "DO NOT USE", "DONOTUSE", "BABY BOY", "BABY GIRL", "DUPLICATE",
    "TEST", "DOUBLE ACCOUNT", "<MRG>",
]
JUNK_SSNS = ["111111111", "123456789", "000000000", "111223333", "219099999"]
JUNK_PHONES = ["5555555555", "0000000000", "1234567890", "9115551234",
               "1115551234"]
JUNK_EMAILS = ["noemail@noemail.com", "test@test.com", "patient@patient.com",
               "none@none.com", "unknown@unknown.com", "declined@hb.org"]
JUNK_ZIPS = ["00000", "11111", "99999", "12345", "54321"]

_ACCENTS = {"A": "Á", "E": "É", "I": "Í", "O": "Ó", "U": "Ú", "N": "Ñ"}


def _maybe(value, present_prob: float, rng: random.Random):
    """Return `value` with prob `present_prob`, else None (missing field)."""
    return value if rng.random() < present_prob else None


# ── Messy renderers — all preserve the value's CANONICAL cleaned form ─────────
def _messy_case(s: str, rng: random.Random) -> str:
    style = rng.random()
    out = s.lower() if style < 0.4 else (s.title() if style < 0.8 else s.upper())
    if rng.random() < 0.15:  # stray internal/edge whitespace
        out = f"  {out} "
    return out


def _add_accents(s: str, rng: random.Random) -> str:
    return "".join(
        _ACCENTS.get(ch.upper(), ch) if rng.random() < 0.25 else ch for ch in s
    )


def _messy_first(s: str, rng: random.Random) -> str:
    out = _messy_case(s, rng)
    if rng.random() < 0.1:
        out = _add_accents(out, rng)
    if rng.random() < 0.08:  # title prefix — cleaner strips it
        out = f"{rng.choice(['Mr', 'Mrs', 'Ms', 'Dr'])} {out}"
    return out


def _messy_last(s: str, rng: random.Random) -> str:
    out = _messy_case(s, rng)
    if rng.random() < 0.1:
        out = _add_accents(out, rng)
    if rng.random() < 0.08:  # trailing generational suffix — cleaner strips it
        out = f"{out} {rng.choice(['JR', 'SR', 'III'])}"
    return out


def _messy_ssn(digits: str, rng: random.Random) -> str:
    fmt = rng.choice(["{a}-{g}-{s}", "{a}{g}{s}", "{a} {g} {s}"])
    return fmt.format(a=digits[:3], g=digits[3:5], s=digits[5:])


def _messy_phone(digits: str, rng: random.Random) -> str:
    a, b, c = digits[:3], digits[3:6], digits[6:]
    return rng.choice([
        f"({a}) {b}-{c}", f"{a}-{b}-{c}", f"{a}.{b}.{c}",
        f"+1{a}{b}{c}", digits, f"1-{a}-{b}-{c}",
    ])


def _messy_dob(iso: str, rng: random.Random) -> str:
    y, m, d = iso.split("-")
    months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep",
              "Oct", "Nov", "Dec"]
    return rng.choice([
        iso, f"{m}/{d}/{y}", f"{int(m)}/{int(d)}/{y}",
        f"{months[int(m) - 1]} {int(d)}, {y}",
    ])


def _messy_state(code: str, rng: random.Random) -> str:
    names = {"IL": "Illinois", "TX": "Texas", "FL": "Florida", "GA": "Georgia",
             "NY": "New York", "MA": "Massachusetts", "CO": "Colorado"}
    r = rng.random()
    if r < 0.7:
        return code
    if r < 0.85 and code in names:
        return names[code].upper() if rng.random() < 0.5 else names[code]
    return code.lower()


# ── Canonical patients ───────────────────────────────────────────────────────
def _new_ssn_digits(rng: random.Random) -> str:
    area = rng.randint(1, 665)
    group = rng.randint(1, 99)
    serial = rng.randint(1, 9999)
    return f"{area:03d}{group:02d}{serial:04d}"


def _new_phone_digits(rng: random.Random) -> str:
    blocked_npa = {"211", "311", "411", "511", "611", "711", "811", "911", "555"}
    npa = rng.randint(200, 999)
    while str(npa)[0] in "01" or str(npa) in blocked_npa:
        npa = rng.randint(200, 999)
    nxx = rng.randint(200, 999)
    while str(nxx) == "555":
        nxx = rng.randint(200, 999)
    return f"{npa}{nxx}{rng.randint(0, 9999):04d}"


def _canonical_patient(rng: random.Random) -> dict:
    """A patient's canonical (already-clean) values, before messy rendering."""
    first = rng.choice(FIRST_NAMES)
    last = rng.choice(LAST_NAMES)
    city, state, zip5 = rng.choice(PLACES)
    street_canon, street_raw = rng.choice(STREETS)
    return {
        "first": first,
        "last": last,
        "middle": _maybe(rng.choice(MIDDLE_NAMES), 0.19, rng),
        "suffix": _maybe(rng.choice(["JR", "SR", "III"]), 0.02, rng),
        "dob": f"{rng.randint(1930, 2015):04d}-{rng.randint(1, 12):02d}-"
               f"{rng.randint(1, 28):02d}",
        "ssn": _maybe(_new_ssn_digits(rng), 0.357, rng),
        "street_canon": street_canon,
        "street_raw": f"{rng.randint(1, 9999)} {street_raw}",
        "city": city,
        "state": state,
        "zip": zip5,
        "phones": [_new_phone_digits(rng)
                   for _ in range(rng.choice([0, 1, 1, 2, 2, 3]))],
        "email": _maybe(
            f"{first.lower()}.{last.lower()}{rng.randint(1, 999)}@gmail.com",
            0.376, rng,
        ),
        "sex": _maybe(rng.choice(["MALE", "FEMALE", "OTHER"]), 0.981, rng),
    }


def _render_raw(patid: str, c: dict, rng: random.Random,
                force: frozenset[str] = frozenset()) -> dict:
    """Render a canonical patient into a messy raw record.

    `force` names canonical keys whose fields must be present and faithfully
    rendered (used for a duplicate's rule-relevant fields so the match survives).
    """
    def keep(field: str, prob: float) -> bool:
        return field in force or rng.random() < prob

    phones = c["phones"]
    phone_slots = [None, None, None, None]
    for i, ph in enumerate(phones[:4]):
        phone_slots[i] = _messy_phone(ph, rng)
    # If a phone match is forced, guarantee the first slot carries a phone.
    if "phones" in force and phones:
        phone_slots[0] = _messy_phone(phones[0], rng)

    rec = {
        "PATID": patid,
        "FirstNM": _messy_first(c["first"], rng) if keep("first", 0.997) else None,
        "LastNM": _messy_last(c["last"], rng) if keep("last", 0.998) else None,
        "MiddleNM": _messy_case(c["middle"], rng) if c["middle"] else None,
        "SuffixNM": c["suffix"],
        "BirthDT": _messy_dob(c["dob"], rng) if keep("dob", 0.998) else None,
        "SSN": _messy_ssn(c["ssn"], rng) if (c["ssn"] and keep("ssn", 0.99))
        else None,
        "AddressLine1": (c["street_raw"] if "street_canon" in force
                         else _messy_case(c["street_raw"], rng))
        if keep("street_canon", 0.968) else None,
        "AddressLine2": _maybe(f"Apt {rng.randint(1, 40)}", 0.295, rng),
        "CityNM": _messy_case(c["city"], rng) if keep("city", 0.975) else None,
        "ZipCD": c["zip"] if keep("zip", 0.973) else None,
        "StateCD": _messy_state(c["state"], rng) if keep("state", 0.972) else None,
        "CountryNM": _maybe("USA", 0.99, rng),
        "PrimaryPhoneNBR": phone_slots[0],
        "Phone01NBR": phone_slots[1],
        "Phone02NBR": phone_slots[2],
        "Phone03NBR": phone_slots[3],
        "Email": _messy_case(c["email"], rng)
        if (c["email"] and keep("email", 1.0)) else None,
        "SexAtBirthDSC": _messy_case(c["sex"], rng)
        if (c["sex"] and keep("sex", 1.0)) else None,
    }

    # Inject text-based nulls into a few optional, non-forced fields.
    for field in ("MiddleNM", "AddressLine2", "CityNM"):
        if field not in force and rng.random() < 0.05:
            rec[field] = rng.choice(TEXT_NULLS)
    if "street_canon" not in force and rng.random() < 0.03:
        rec["AddressLine1"] = rng.choice(ADDRESS_PLACEHOLDERS)
    return rec


def _junk_record(patid: str, rng: random.Random) -> dict:
    """A messy record carrying values the cleaner must nullify / invalidate."""
    rec = _render_raw(patid, _canonical_patient(rng), rng)
    kind = rng.random()
    if kind < 0.35:  # invalid-marker name → valid_record=False
        marker = rng.choice(INVALID_MARKERS)
        if rng.random() < 0.5:
            rec["FirstNM"] = marker
        else:
            rec["LastNM"] = marker
    else:  # junk values the cleaner nullifies field-by-field
        rec["SSN"] = rng.choice(JUNK_SSNS)
        rec["PrimaryPhoneNBR"] = rng.choice(JUNK_PHONES)
        rec["Email"] = rng.choice(JUNK_EMAILS)
        rec["ZipCD"] = rng.choice(JUNK_ZIPS)
    return rec


_SCENARIO_FORCE = {
    "ssn": frozenset({"ssn"}),
    "email": frozenset({"email"}),
    "name_dob_sex": frozenset({"first", "last", "dob", "sex"}),
    "name_dob_phone": frozenset({"first", "last", "dob", "phones"}),
    "name_dob_email": frozenset({"first", "last", "dob", "email"}),
    "name_dob_address": frozenset({"first", "last", "dob",
                                   "street_canon", "city", "state", "zip"}),
}


def _scenario_ok(base: dict, scenario: str) -> bool:
    """True if `base` carries the canonical values this scenario needs."""
    if scenario == "ssn":
        return base["ssn"] is not None
    if scenario == "email":
        return base["email"] is not None
    if scenario == "name_dob_phone":
        return bool(base["phones"])
    if scenario == "name_dob_sex":
        return base["sex"] is not None
    if scenario == "name_dob_email":
        return base["email"] is not None
    return True  # name_dob_address always has address components


def _plant_duplicate(base: dict, dup_patid: str, scenario: str,
                     rng: random.Random) -> dict:
    """Build a duplicate sharing the canonical fields `scenario` needs."""
    dup = _canonical_patient(rng)
    force = _SCENARIO_FORCE[scenario]
    if scenario == "ssn":
        dup["ssn"] = base["ssn"]
    elif scenario == "email":
        dup["email"] = base["email"]
    else:
        dup["first"], dup["last"], dup["dob"] = (
            base["first"], base["last"], base["dob"])
        if scenario == "name_dob_sex":
            dup["sex"] = base["sex"]
        elif scenario == "name_dob_phone":
            dup["phones"] = [base["phones"][0]] + dup["phones"]
        elif scenario == "name_dob_email":
            dup["email"] = base["email"]
        elif scenario == "name_dob_address":
            for k in ("street_canon", "street_raw", "city", "state", "zip"):
                dup[k] = base[k]
    return _render_raw(dup_patid, dup, rng, force=force)


def generate(n: int, dup_rate: float, junk_rate: float, seed: int) -> pd.DataFrame:
    """Generate base patients + planted duplicates + standalone junk records."""
    rng = random.Random(seed)
    np.random.seed(seed)

    base_canon = [_canonical_patient(rng) for _ in range(n)]
    records = [_render_raw(f"PAT{i:06d}", c, rng) for i, c in enumerate(base_canon)]

    scenarios = list(_SCENARIO_FORCE.keys())
    n_dups_target = int(n * dup_rate)
    dup_idx = 0
    attempts = 0
    while dup_idx < n_dups_target and attempts < n_dups_target * 20:
        attempts += 1
        base = rng.choice(base_canon)
        scenario = rng.choice(scenarios)
        if not _scenario_ok(base, scenario):
            continue
        records.append(_plant_duplicate(base, f"DUP{dup_idx:06d}", scenario, rng))
        dup_idx += 1

    for j in range(int(n * junk_rate)):
        records.append(_junk_record(f"JNK{j:06d}", rng))

    df = pd.DataFrame(records, columns=COLUMNS)
    return df.sample(frac=1.0, random_state=seed).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=3000,
                        help="Number of base patients (default 3000)")
    parser.add_argument("--dup-rate", type=float, default=0.2,
                        help="Planted duplicates as a fraction of n (default 0.2)")
    parser.add_argument("--junk-rate", type=float, default=0.08,
                        help="Standalone junk records as a fraction of n "
                             "(default 0.08)")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT,
                        help=f"Output CSV path (default {DEFAULT_OUT})")
    args = parser.parse_args()

    df = generate(args.n, args.dup_rate, args.junk_rate, args.seed)
    n_dups = int(df["PATID"].str.startswith("DUP").sum())
    n_junk = int(df["PATID"].str.startswith("JNK").sum())
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)

    print(f"Wrote {len(df):,} records to {args.out}")
    print(f"  base patients:      {len(df) - n_dups - n_junk:,}")
    print(f"  planted duplicates: {n_dups:,}")
    print(f"  junk records:       {n_junk:,}")
    print(f"  columns ({len(df.columns)}): {list(df.columns)}")


if __name__ == "__main__":
    main()
