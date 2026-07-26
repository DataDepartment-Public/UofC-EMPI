"""Tiny synthetic fixtures — small enough for a fast, real end-to-end smoke
test of both training scripts (not mocked), with none of the properties of
real PHI (random names/SSNs, no relation to any actual person)."""

from __future__ import annotations

import random

import pandas as pd
import pytest


def _synthetic_cleaned_records(n_people: int, seed: int = 7) -> pd.DataFrame:
    rng = random.Random(seed)
    first_names = ["alex", "jordan", "sam", "taylor", "morgan", "casey", "riley", "drew"]
    last_names = ["smith", "johnson", "lee", "garcia", "brown", "davis", "clark", "lewis"]
    rows = []
    for i in range(n_people):
        first = rng.choice(first_names)
        last = rng.choice(last_names)
        dob = f"19{rng.randint(50, 99)}-0{rng.randint(1, 9)}-{rng.randint(10, 28)}"
        ssn = f"{rng.randint(100, 999)}{rng.randint(10, 99)}{rng.randint(1000, 9999)}"
        rows.append(
            {
                "PATID": f"P{i:05d}",
                "FirstNM_clean": first,
                "LastNM_clean": last,
                "MiddleNM_clean": None,
                "BirthDT_clean": dob,
                "SSN_clean": ssn,
                "Email_clean": f"{first}.{last}{i}@example.com",
                "AddressLine1_clean": f"{100 + i} main st",
                "SexAtBirthDSC_clean": rng.choice(["M", "F"]),
                "Phones_set": str({f"555000{rng.randint(1000, 9999)}"}),
            }
        )
        # every third person gets a near-duplicate record (a plausible pair)
        if i % 3 == 0:
            dup = dict(rows[-1])
            dup["PATID"] = f"P{i:05d}D"
            dup["Email_clean"] = f"{first}.{last}{i}alt@example.com"
            rows.append(dup)
    return pd.DataFrame(rows)


def _labels_for(cleaned: pd.DataFrame) -> pd.DataFrame:
    """Every base/duplicate pair -> positive label; a handful of random
    cross pairs -> negative label, with a few flagged ambiguous."""
    rng = random.Random(11)
    positives = []
    for patid in cleaned["PATID"]:
        if patid.endswith("D"):
            positives.append((patid[:-1], patid))

    all_ids = list(cleaned["PATID"])
    negatives = []
    while len(negatives) < len(positives) * 2:
        a, b = rng.sample(all_ids, 2)
        pair = tuple(sorted((a, b)))
        if pair not in positives and pair not in negatives:
            negatives.append(pair)

    rows = []
    for a, b in positives:
        rows.append({"PATID_A": a, "PATID_B": b, "final_gold_label": True, "ambiguous_pair": False})
    for i, (a, b) in enumerate(negatives):
        ambiguous = i % 4 == 0
        rows.append(
            {
                "PATID_A": a,
                "PATID_B": b,
                "final_gold_label": False,
                "ambiguous_pair": ambiguous,
            }
        )
    return pd.DataFrame(rows)


@pytest.fixture
def synthetic_cleaned_parquet(tmp_path):
    # Large enough that every label class comfortably survives a stratified
    # 60/20/20 split twice over (sklearn's train_test_split errors on classes
    # too small to place >=1 member in every split).
    cleaned = _synthetic_cleaned_records(n_people=200)
    path = tmp_path / "cleaned.parquet"
    cleaned.to_parquet(path)
    return path, cleaned


@pytest.fixture
def synthetic_gold_labels_csv(tmp_path, synthetic_cleaned_parquet):
    _, cleaned = synthetic_cleaned_parquet
    labels = _labels_for(cleaned)
    path = tmp_path / "gold_labels.csv"
    labels.to_csv(path, index=False)
    return path


@pytest.fixture
def synthetic_silver_labels_csv(tmp_path, synthetic_cleaned_parquet):
    """Same shape, renamed to the FS trainer's expected `silver_label` column
    (0/1, not the gold-labels' two-boolean-column schema)."""
    _, cleaned = synthetic_cleaned_parquet
    labels = _labels_for(cleaned)
    labels["silver_label"] = (labels["final_gold_label"]).astype(int)
    out = labels[["PATID_A", "PATID_B", "silver_label"]]
    path = tmp_path / "silver_labels.csv"
    out.to_csv(path, index=False)
    return path
