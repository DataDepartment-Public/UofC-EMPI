# data/processed/ — Stage 1 output: cleaned dataset `[IMPLEMENTED]`

Full contract: `docs/Data-Contract.md` → "Stage 1 — Cleaned dataset."

- **Producer:** `src/preprocessing/transformations.py::transform_dataframe`
  (via `src/preprocessing/clean.py`)
- **Contract:** `contracts.CleanedRecords` (`strict=False` — passthrough
  columns allowed)
- **Consumers:** blocking (Stage 2), deterministic rules (Stage 3, for
  attribute agreement), `fs_matcher/train.py`
- **File:** `MDM_Population_cleaned_<run_id>.parquet`
- **Grain:** one row per source record (`PATID`)

For every cleaned source field the producer keeps the original in
`<field>_raw` (never modified) alongside the standardized `<field>_clean`.

### Identity & validity

| Column | Dtype | Nullable | Notes |
|---|---|---|---|
| `PATID` | string | no | Record identity; passthrough, never transformed. |
| `valid_record` | bool | no | `False` if any field-level rule flagged the record, or both names null. Blocking drops `valid_record == False`. |

### Cleaned attribute columns (consumed downstream)

| Column | Dtype | Nullable | Consumed by |
|---|---|---|---|
| `FirstNM_clean` | string | yes | blocking (B4), rules |
| `LastNM_clean` | string | yes | blocking (B3,B4,B7,B8,B9), rules |
| `BirthDT_clean` | datetime64[ns] / NaT | yes | blocking (B3,B4,B7,B8), rules |
| `SSN_clean` | string (9 digits) | yes | blocking (B1), rules |
| `last_4_SSN` | string (4 digits) | yes | blocking (B9) |
| `Email_clean` | string | yes | blocking (B6), rules |
| `ZipCD_clean_base` | string (5 digits) | yes | blocking (B7) |
| `AddressLine1_clean` | string | yes | rules (NAME_DOB_ADDRESS) |
| `SexAtBirthDSC_clean` | string ∈ {MALE,FEMALE,OTHER} | yes | rules (NAME_DOB_SEX) |
| `Phones_set` | list&lt;string&gt; | yes | blocking (B5), rules (phone agreement) |

`Phones_set` (and `full_name_tokens`) serialize as a native Parquet
`list<string>` on disk but round-trip as a NumPy `ndarray` — consumers must
tolerate `ndarray \| list \| set \| str` and normalize to a `frozenset`. See
`Data-Contract.md`'s "Multi-valued column serialization" section.

**Other cleaned/derived columns** (produced, not yet consumed downstream):
`MiddleNM_clean`, `SuffixNM_clean`, `AddressLine2_clean`, `CityNM_clean`,
`StateCD_clean`, `ZipCD_clean_ext`, `PrimaryPhoneNBR_clean`,
`Phone01/02/03NBR_clean`, `full_name_tokens`, `full_name_compact`,
`Address_normalized` (`NaN` when `libpostal` is unavailable), plus every
`<field>_raw`.
