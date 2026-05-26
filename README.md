# EMPI — Entity Resolution / Master Patient Index

Capstone project for **AllianceChicago**. The goal is to group records from the `MDM_Population` dataset that belong to the same patient across source systems. The pipeline has two stages: per-field **data cleaning** that standardizes raw fields and derives matching artifacts, and multi-pass **blocking** that emits candidate pairs for a downstream entity-resolution matcher.

---

## Data Cleaning

Standardizes every field according to a fixed set of rules and produces a versioned cleaned Parquet alongside derived columns (`last_4_SSN`, `ZipCD_clean_base`/`_ext`, `full_name_tokens`, `full_name_compact`, `Phones_set`, `Address_normalized`, and a record-level `valid_record` boolean).

- **Raw input location:** `data/raw/` (CSV or XLSX files).
- **Transformation rules:** [`docs/Data-Cleaning-Guide.md`](docs/Data-Cleaning-Guide.md) — single source of truth, field by field, in the order rules must be applied. Code in `src/data/transformations.py` mirrors this document.
- **Output location:** `data/processed/<stem>_cleaned_v<N>_<YYYY_MM_DD>.parquet` — the version `<N>` is auto-incremented past the highest existing version for that input stem (the date suffix is ignored when computing the next version).

### Run

```bash
# Clean every supported file in data/raw/
python -m src.data.clean

# Clean a single file (optionally to a custom output directory)
python -m src.data.clean <path/to/raw_file.csv> [<output_dir>]
```

When re-reading a processed Parquet elsewhere in code, use `src.data.clean.load_cleaned(path)`. Parquet preserves dtypes natively, so leading zeros on `PATID`, `SSN`, `ZipCD`, and `last_4_SSN` survive without any explicit dtype handling.

---

## Blocking

Multi-pass blocking scheme that emits candidate `(PATID_A, PATID_B)` pairs likely to refer to the same patient, alongside metadata about which blocks generated each pair. Eight blocks: SSN exact, Double-Metaphone(Last)+DOB, Last+BirthYear+First[:3], phone-set intersection, email exact, DM(Last)+ZIP+BirthYear, Soundex(First)+Soundex(Last)+BirthYear, and Last+First+SSN-last4. Output is splink-compatible.

- **Input:** a cleaned dataset produced by the cleaning stage above (`data/processed/<stem>_cleaned_v<N>_<YYYY_MM_DD>.parquet`).
- **Implementation:** `src/features/blocking.py` defines the blocking logic; `src/features/run_blocking.py` is the pipeline script that loads the cleaned Parquet, runs all 8 blocks, prints an audit report, and saves the candidate pairs.
- **Output location:** `data/blocking/candidate_pairs_v<N>_<YYYY_MM_DD>.parquet`. The version `<N>` is auto-incremented past the highest existing `candidate_pairs_v<N>_*.parquet` in `data/blocking/` (starts at `v1` when empty). Schema: `PATID_A | PATID_B | source_blocks | n_blocks`.

### Run

```bash
# Defaults: --input  highest-versioned MDM_Population_cleaned_v<N>_*.parquet
#                    in data/processed/ (auto-resolved at runtime)
#           --output data/blocking/
# Output filename version (v<N>) auto-increments past the highest already
# in the output dir.
python src/features/run_blocking.py

# Override any of the defaults
python src/features/run_blocking.py \
    --input  data/processed/MDM_Population_cleaned_v1_2026_05_24.parquet \
    --output data/blocking
```
