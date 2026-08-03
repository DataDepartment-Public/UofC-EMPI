# Data Cleaning & Transformation Log

This document tracks all transformations applied to the `MDM_Population` dataset to prepare it for the Entity Resolution / Master Patient Index (MPI) pipeline.

## Output schema conventions

- For every cleaned field, preserve the original value in a `<field_name>_raw` column and write the cleaned value to `<field_name>_clean`. The `_raw` column is never modified after ingest.
- Add a single boolean column called `valid_record` at the record level, initialized to `True`. Any field-level rule that flags the record as invalid sets this column to `False` for the whole row. There is only one `valid_record` per record (no per-field validity flags). Even if a field is mark as valid_record = False, the data transformations will still apply.

## Order of operations

- Within each field, apply transformations **in the order listed**. Steps are intentionally unnumbered so new rules can be inserted without renumbering.

## Unicode normalization rule

Every field whose steps say "Apply Unicode normalization" uses the same procedure:

1. Apply `unidecode.unidecode(s)` to transliterate Unicode characters to their closest ASCII equivalents. This handles:
   - Diacriticals: `Á → A`, `Ñ → N`, `Ü → U`, `Ç → C`, `É → E`, etc.
   - Ligatures and stroked letters: `Æ → AE`, `Œ → OE`, `ß → ss`, `Ø → O`, `Ł → L`, `Đ → D`, `Þ → Th`.
   - Non-Latin scripts: Cyrillic, Greek, Arabic, Hangul, CJK, etc. are transliterated to their romanized forms (`Иван → Ivan`).
2. Any characters that `unidecode` cannot transliterate (very rare — typically obscure symbols) are dropped at the subsequent per-field character-class filter step.

Dependency: `pip install Unidecode`. Output is guaranteed ASCII, after which the per-field `[A-Z \-']` (names/city) and `[A-Z0-9 ]` (addresses) filters apply cleanly.

# Field only transformations

---

## PATID (Unique Patient ID)

- No preprocessing needed

---

## FirstNM (First Name)

- Apply Unicode normalization (see [Unicode normalization rule](#unicode-normalization-rule))
- Split CamelCase joins by inserting a space at every lowercase→uppercase boundary (e.g., `WilliamEarl → William Earl`). Must run before uppercase conversion
- Convert all values to uppercase, collapse multiple spaces and trim whitespace
- Mark the records that contain these strings as invalid(valid_record = False):
  -`BABYBOY`, `BABY BOY`, `BABYGIRL`, `BABY GIRL`,  `DUPLICATE`, `DONOTUSE`, `DO NOT USE`, `DONT USE`, `DO NOT USE DUPLICATE`, `DO NOT USE DOUBLE ACCOUNT`, `DONOT USED`, `DO NOT USE DOBBLE ACCOUNT`, `DO NOT`, `DUPLICATE DO NOT USE`, `DUPLICATE DONT USE`, `DON'T USE`, `MEDICARE`, `DOUBLE ACCOUNT`, `DUPLICATE ACCOUNT`, `ACCOUNT`, `<MRG>`, `INACTIVE`, `UPRR`
- Any character not in [A-Z \-'] is removed(Remove characters that aren't alphabetic, space, hyphen or apostrophes)
- Standardize text-based nulls to `NaN`:
  - `UNKNOWN`, `NULL`, `NAN`, `NONE`, `N/A`, `NA`, empty strings
- Mark the records equal to these strings as invalid(valid_record = False):
  - `TEST`, `BABY`
- Mark the records whose **raw** value matches the regex `^\d+$` or `^[Ii][Dd]\d+$` (all-numeric, or `ID` followed by digits — e.g., `12345`, `ID4567`) as invalid (valid_record = False)
- Remove titles from beginning of string:
  - `MR `, `MRS `, `MS `, `DR `, `REV `
- Move generational suffixes to the suffix field (SuffixNM_clean):
  - ` JR`, ` SR`, ` II`, ` III`, ` IV`, ` V`
---

## LastNM (Last Name)

- Apply Unicode normalization (see [Unicode normalization rule](#unicode-normalization-rule))
- Split CamelCase joins by inserting a space at every lowercase→uppercase boundary (e.g., `RocaGarcia → Roca Garcia`). Must run before uppercase conversion
- Convert to uppercase, collapse multiple spaces and trim whitespace
- Mark the records that contain these strings as invalid(valid_record = False):
  - `BABYBOY`, `BABY BOY`, `BABYGIRL`, `BABY GIRL`,  `DUPLICATE`, `DONOTUSE`, `DO NOT USE`, `DONT USE`, `DO NOT USE DUPLICATE`, `DO NOT USE DOUBLE ACCOUNT`, `DONOT USED`, `DO NOT USE DOBBLE ACCOUNT`, `DO NOT`, `DUPLICATE DO NOT USE`, `DUPLICATE DONT USE`, `DON'T USE`, `MEDICARE`, `DOUBLE ACCOUNT`, `DUPLICATE ACCOUNT`, `ACCOUNT`, `<MRG>`, `INACTIVE`, `UPRR`
- Any character not in [A-Z \-'] is removed(Remove characters that aren't alphabetic, space, hyphen or apostrophes)
- Standardize text-based nulls to `NaN`
  - `UNKNOWN`, `NULL`, `NAN`, `NONE`, `N/A`, `NA`, empty strings
- Mark the records equal to these strings as invalid(valid_record = False):
  - `TEST`, `BABY`
- Mark the records whose **raw** value matches the regex `^\d+$` or `^[Ii][Dd]\d+$` (all-numeric, or `ID` followed by digits — e.g., `12345`, `ID4567`) as invalid (valid_record = False)
- Move generational suffixes appended to surname to the suffix field (SuffixNM_clean):
  - ` JR`, ` SR`, ` II`, ` III`, ` IV`, ` V`

---

## MiddleNM (Middle Name)

- Apply Unicode normalization (see [Unicode normalization rule](#unicode-normalization-rule))
- Split CamelCase joins by inserting a space at every lowercase→uppercase boundary. Must run before uppercase conversion
- Convert to uppercase, collapse multiple spaces and trim whitespace
- Mark the records that contain these strings as invalid(valid_record = False):
  - `BABYBOY`, `BABY BOY`, `BABYGIRL`, `BABY GIRL`,  `DUPLICATE`, `DONOTUSE`, `DO NOT USE`, `DONT USE`, `DO NOT USE DUPLICATE`, `DO NOT USE DOUBLE ACCOUNT`, `DONOT USED`, `DO NOT USE DOBBLE ACCOUNT`, `DO NOT`, `DUPLICATE DO NOT USE`, `DUPLICATE DONT USE`, `DON'T USE`, `MEDICARE`, `DOUBLE ACCOUNT`, `DUPLICATE ACCOUNT`, `ACCOUNT`, `<MRG>`, `INACTIVE`, `UPRR`
- Mark the records equal to these strings as invalid(valid_record = False):
  - `TEST`, `BABY`
- Mark the records whose **raw** value matches the regex `^\d+$` or `^[Ii][Dd]\d+$` (all-numeric, or `ID` followed by digits — e.g., `12345`, `ID4567`) as invalid (valid_record = False)
- Standardize text-based nulls to `NaN`:
  - `NMI`, `UNKNOWN`, `NULL`, `N/A`, `-`
- Any character not in [A-Z \-'] is removed(Remove characters that aren't alphabetic, space, hyphen or apostrophes)
- Move generational suffixes to the suffix field (SuffixNM_clean):
  - ` JR`, ` SR`, ` II`, ` III`, ` IV`, ` V`

---

## SuffixNM (Name Suffix)

- Convert to uppercase, collapse multiple spaces and trim whitespace
- Standardize text-based nulls to `NaN`
  - `UNKNOWN`, `NULL`, `NAN`, `NONE`, `N/A`, `NA`, empty strings
- Remove punctuation
- Normalize ordinal suffixes:
  - `2ND → II`
  - `3RD → III`
  - `4TH → IV`
  - `5TH → V`
- Retain only valid generational suffixes:
  - `JR`, `SR`, `I`, `II`, `III`, `IV`, `V`
- Convert all invalid suffixes to `NaN`

---

# BirthDT (Date of Birth)

- Parse values into standardized datetime format
- Convert invalid or unparseable values to `NaT`
- Nullify dates equal or before `1900-01-01`
- Nullify future dates

---

# SSN (Social Security Number)

- Remove all non-numeric characters
- Left-pad 7- or 8-digit SSNs with leading zeros
- Convert non-9-digit values to `NaN`
- Nullify low-entropy placeholder patterns (these pass area/group/serial and `stdnum` validation, yet are clerical placeholders, not identities — see `_is_placeholder_ssn`):
  - **≤ 2 distinct digits** across the 9 positions (e.g., `111111111`, `222222222`, `121212121`)
  - **one digit fills ≥ 7 of the 9 positions** (e.g., `333333330`)
  - a **full ascending or descending consecutive run (mod 10)** (e.g., `012345678`, `123456789`, `234567890`, `987654321`)
- Validate structural SSA rules. Treat the 9 digits as `AAA-GG-SSSS` (area–group–serial) and nullify any value where:
  - Area number (digits 1–3) is `000`, `666`, in the range `900–999`
  - Group number (digits 4–5) is `00`
  - Serial number (digits 6–9) is `0000`
- Nullify known exact junk values:
  - `010101010`, `090909090`, `000000001`, `999999998`, `111223333`, `219099999`, `457555462`, `333333330`, `003333333`, `333333300`, `033333333`, `333333339`, `099999999`, `333333000`
- Extract in a new field `last_4_SSN` the last 4 digits of the cleaned SSN (for values that survive all the junk/validity rules above)
- **Partial-SSN recovery for `last_4_SSN`.** Some source systems store only the last 4 digits of the SSN, either bare (`1234`) or left-padded (`01234`, `000001234`). Such a value can never become a valid `SSN_clean` — it has no area/group digits — but its last 4 are genuine and are the field's main coverage win. When a value fails full-SSN validation above, still populate `last_4_SSN` if **every digit before the final four is a zero, or there are none**:
  - Recovered: `1234`, `01234`, `0001234`, `000001234` → `1234`
  - Not recovered: `900112222`, `111223333`, `123456789` — the leading digits carry information, so the value is a malformed or junk *full* SSN, not a stored partial, and its last 4 are not trustworthy
  - Values shorter than 4 digits yield `NaN` (no last-4 to recover)
  - `0000` is never emitted — a real SSN's serial (digits 6–9) is never `0000`
  - `SSN_clean` stays `NaN` in every recovery case. `last_4_SSN` being populated does **not** imply `SSN_clean` is populated

---

# AddressLine1

- Apply Unicode normalization (see [Unicode normalization rule](#unicode-normalization-rule))
- Convert to uppercase and trim whitespace
- Standardize text-based nulls to `NaN` and nullify invalid strings:
  - `UNKNOWN`, `NULL`, `NAN`, `NONE`, `N/A`, `NA`, `X`, empty strings
  - `HOMELESS`, `TRANSIENT`, `BAD ADDRESS`, `NO ADDRESS`, `?`, `NO MAIL`, `GENERAL DELIVERY`, `NKA`, `.`, `NOT TAKEN`, `ZOCDOC`, `GET`, `HOPE CENTER`, `INCORRECT ADDRESS`, `<MRG>`
- Mark the records that contain these strings as invalid(valid_record = False):
  - `BABYBOY`, `BABY BOY`, `BABYGIRL`, `BABY GIRL`,  `DUPLICATE`, `DONOTUSE`, `DO NOT USE`, `DONT USE`, `DO NOT USE DUPLICATE`, `DO NOT USE DOUBLE ACCOUNT`, `DONOT USED`, `DO NOT USE DOBBLE ACCOUNT`, `DO NOT`, `DUPLICATE DO NOT USE`, `DUPLICATE DONT USE`, `DON'T USE`, `MEDICARE`, `DOUBLE ACCOUNT`, `DUPLICATE ACCOUNT`, `ACCOUNT`, `<MRG>`, `INACTIVE`, `UPRR`
- Mark the records equal to these strings as invalid(valid_record = False):
  - `TEST`, `BABY`
- Remove any non-alphanumeric or space character
- Standardize street suffixes using USPS Publication 28 Appendix C abbreviations. Match against whole-word tokens (split on whitespace) so embedded substrings in real street names are not rewritten:
  - `STREET`, `STR`, `STRT` → `ST`
  - `AVENUE`, `AV`, `AVN`, `AVENU` → `AVE`
  - `BOULEVARD`, `BOUL`, `BOULV` → `BLVD`
  - `ROAD` → `RD`
  - `DRIVE`, `DRIV`, `DRV` → `DR`
  - `LANE` → `LN`
  - `COURT`, `CRT` → `CT`
  - `PLACE` → `PL`
  - `CIRCLE`, `CIRC`, `CRCL`, `CRCLE` → `CIR`
  - `TERRACE`, `TERR` → `TER`
  - `PARKWAY`, `PARKWY`, `PKWAY`, `PKY` → `PKWY`
  - `HIGHWAY`, `HIGHWY`, `HIWAY`, `HIWY`, `HWAY` → `HWY`
  - `TRAIL`, `TRL` → `TRL`
  - `ALLEY`, `ALLY`, `ALLEE` → `ALY`
  - `SQUARE`, `SQR`, `SQRE`, `SQU` → `SQ`
  - `PLAZA`, `PLZA` → `PLZ`
  - `EXPRESSWAY`, `EXPR`, `EXPW` → `EXPY`
  - `FREEWAY`, `FRWAY`, `FRWY` → `FWY`
  - `CROSSING`, `CRSSNG` → `XING`
  - `BRIDGE`, `BRDGE` → `BRG`
  - `ROUTE` → `RTE`
  - `TURNPIKE`, `TRNPK` → `TPKE`
  - `POINT`, `POINTS` → `PT`
  - `RIDGE` → `RDG`
  - `VIEW` → `VW`
  - `VILLAGE`, `VILLG`, `VILLIAGE` → `VLG`
  - `ESTATES` → `EST`
  - `HEIGHTS` → `HTS`
  - `LOOP`, `PIKE`, `RUN`, `WALK`, `WAY` → unchanged (already canonical)
- Standardize directional prefixes/suffixes. Match against whole-word tokens:
  - `NORTH` → `N`
  - `SOUTH` → `S`
  - `EAST` → `E`
  - `WEST` → `W`
  - `NORTHEAST` → `NE`
  - `NORTHWEST` → `NW`
  - `SOUTHEAST` → `SE`
  - `SOUTHWEST` → `SW`
- Normalize street numbers:
  - Remove leading zeros from the leading numeric token only (e.g., `00123 MAIN ST → 123 MAIN ST`). Do not touch later digit runs (`123 W 45TH ST` stays as-is).

---

# AddressLine2
- Apply Unicode normalization (see [Unicode normalization rule](#unicode-normalization-rule))
- Convert to uppercase and trim whitespace
- Standardize text-based nulls to `NaN` and nullify invalid strings:
  - `UNKNOWN`, `NULL`, `NAN`, `NONE`, `N/A`, `NA`, empty strings
  - `HOMELESS`, `TRANSIENT`, `BAD ADDRESS`, `NO ADDRESS`, `?`, `NO MAIL`, `GENERAL DELIVERY`, `NKA`, `.`, `NOT TAKEN`, `ZOCDOC`, `GET`, `HOPE CENTER`, `INCORRECT ADDRESS`
- Mark the records that contain these strings as invalid(valid_record = False):
  - `BABYBOY`, `BABY BOY`, `BABYGIRL`, `BABY GIRL`,  `DUPLICATE`, `DONOTUSE`, `DO NOT USE`, `DONT USE`, `DO NOT USE DUPLICATE`, `DO NOT USE DOUBLE ACCOUNT`, `DONOT USED`, `DO NOT USE DOBBLE ACCOUNT`, `DO NOT`, `DUPLICATE DO NOT USE`, `DUPLICATE DONT USE`, `DON'T USE`, `MEDICARE`, `DOUBLE ACCOUNT`, `DUPLICATE ACCOUNT`, `ACCOUNT`, `<MRG>`, `INACTIVE`, `UPRR`
- Mark the records equal to these strings as invalid(valid_record = False):
  - `TEST`, `BABY`
- Remove any non-alphanumeric or space character
- Standardize unit designators using USPS Publication 28 Appendix C2 abbreviations. Match against whole-word tokens:
  - `APARTMENT`, `APRT` → `APT`
  - `SUITE` → `STE`
  - `ROOM` → `RM`
  - `FLOOR`, `FLR` → `FL`
  - `BUILDING`, `BLD`, `BLDNG` → `BLDG`
  - `DEPARTMENT` → `DEPT`
  - `PENTHOUSE` → `PH`
  - `BASEMENT` → `BSMT`
  - `LOWER` → `LOWR`
  - `UPPER` → `UPPR`
  - `TRAILER` → `TRLR`
  - `HANGAR` → `HNGR`
  - `OFFICE` → `OFC`
  - `FRONT` → `FRNT`
  - `LOBBY` → `LBBY`
  - `SPACE` → `SPC`
  - `REAR`, `SIDE`, `UNIT`, `KEY`, `LOT`, `PIER`, `SLIP`, `STOP` → unchanged (already canonical)

---

# CityNM (City Name)

- Apply Unicode normalization (see [Unicode normalization rule](#unicode-normalization-rule))
- Convert to uppercase and trim whitespace
- Standardize text-based nulls to `NaN`
  - `UNKNOWN`, `NULL`, `NAN`, `NONE`, `N/A`, `NA`, empty strings
- Any character not in [A-Z \-'] is removed(Remove characters that aren't alphabetic, space, hyphen or apostrophes)

---

# ZipCD (ZIP Code)

- Remove all non-numeric characters
- Left-pad ZIPs missing leading zeros if Zip has 4 digits or 8 digits
- Split ZIP into:
  - `ZipCD_base`: First (5 digits if present)
  - `ZipCD_ext`: Extension(last 4 digits when the ZIP is 9 numbers)
- Truncate primary ZIP to first 5 digits
- Convert non-5-digit ZIPs to `NaN`
- Nullify placeholder ZIPs:
  - All five digits identical: `00000`, `11111`, `22222`, `33333`, `44444`, `55555`, `66666`, `77777`, `88888`, `99999`
  - Sequential digits: `12345`, `54321`
- Cross-check ZIP against state

---

# StateCD (State Code)

- Convert to uppercase and trim whitespace
- Standardize text-based nulls to `NaN`
  - `UNKNOWN`, `NULL`, `NAN`, `NONE`, `N/A`, `NA`, empty strings
- Map full state names to standard 2-letter abbreviations (e.g., `CALIFORNIA → CA`, `NEW YORK → NY`)
- Convert numeric CCN state codes to the corresponding 2-letter USPS abbreviation. The CCN state code is the 2-digit prefix CMS uses in the CMS Certification Number (CCN). When the source value is numeric (e.g., `05`, `33`), map. Pad single-digit values (`5 → 05`) before lookup
- Convert invalid values (anything not resolvable to a valid 2-letter USPS code after the mappings above) to `NaN`

---

# CountryNM (Country Name)

- No preprocessing needed. Not to be used initially

---

# PrimaryPhoneNBR, Phone01NBR, Phone02NBR, Phone03NBR

- Remove all non-numeric characters
- Delete leading 1s if any and number is 11 digits long
- Convert non-10-digit values to `NaN`
- Nullify invalid patterns. Treat the 10-digit number as `NPA-NXX-XXXX` (area code – central office code – line number) and nullify any value that matches any of these rules:
  - **Repeating digits** — all 10 digits identical (`DDDDDDDDDD` for any digit `D` in 0–9, e.g., `0000000000`, `5555555555`)
  - **Sequential digits** — the 10 digits form an ascending or descending consecutive sequence:
    - `0123456789`, `1234567890`
    - `9876543210`, `0987654321`
  - **Invalid area code (NPA, digits 1–3)** per NANP rules:
    - First digit is `0` or `1` (NPA first digit must be 2–9): nullify any NPA in `000–199`
    - N11 codes reserved for special services: `211`, `311`, `411`, `511`, `611`, `711`, `811`, `911`
    - Unassigned: `555`
  - **Invalid central office code (NXX, digits 4–6)** per NANP rules:
    - First digit is `0` or `1` (NXX first digit must be 2–9)
    - Fictional/reserved range: digits 4–7 equal `5550` or `5551` (the `555-0100` through `555-0199` line block reserved for fiction)

---

# Email

- Convert to lowercase and delete whitespaces
- Format validation — domain structure:
  - Required pattern:  `^[^\s@]+@[^\s@]+\.[^\s@]+$` (`[one or more chars]@[one or more chars].[one or more chars]`)
  - Nullify if pattern not matched
- Standardize text-based nulls to `NaN` and nullify invalid strings:
  - `unknown`, `null`, `nan`, `none`, `n/a`, `na`, empty strings
- Nullify values missing `@`
- Nullify exact known junk email values:
- `noemail@noemail.com`, `noemail@textmessage.com`, `noemail@textmessaging.com`, `aca@eriefamilyhealth.org`, `aca@eriefamilyheath.org`, `declined@hamakua-health.org`, `aca@eriefamilyhealth.com`, `noemail@textmessages.com`, `decline@hamakua-health.org`, `noemai@noemail.com`, `patientdeclined@howardbrown.org`, `ptdeclined@hb.org`, `noemail@email.com`, `no@email.com`, `none@none.com`, `unknown@unknown.com`, `unknown@email.com`, `test@test.com`, `test@example.com`, `donotreply@donotreply.com`, `noreply@noreply.com`, `email@email.com`, `patient@patient.com`, `none@noemail.com`, `na@na.com`, `null@null.com`, `none@gmail.com`
- Nullify values that contain:
  - `decline`, `noemail`
- Pattern-based junk detection — local part (before `@`):
  - Begins with: `noemail`, `no-email`, `no_email`, `noreply`, `no-reply`, `donotreply`, `do-not-reply`, `unknown`, `test`, `patient`, `none`, `null`, `na`
  - Local part is 1–2 characters total (invalid personal emails)
  - Contains substring `123456` (sequential digit pattern)
- Pattern-based junk detection — domain (after `@`):
  - `@example.com`, `@example.org`, `@example.net` — RFC reserved test domains
  - `@test.com`, `@test.org`, `@noemail.com`, `@noreply.com`, `@donotreply.com`, `@unknown.com`, `@123.com`

---

# SexAtBirthDSC

- Convert to uppercase and trim whitespace
- Nullify:
  - `UNKNOWN`
  - `NULL`
  - `N/A`
- Retain valid values:
  - `MALE`
  - `FEMALE`
  - `OTHER`

---

# Global Cross-Field Transformations
## First Name and Last Name
- Mark as invalid (`valid_record = False`) only when **both** `FirstNM_clean` and `LastNM_clean` are null

## Name token derivations
After per-field cleaning of `FirstNM`, `MiddleNM`, and `LastNM`, derive two record-level matching artifacts:
- `full_name_tokens` — single record-level Python `set[str]` built by splitting each of `FirstNM_clean`, `MiddleNM_clean`, `LastNM_clean` on whitespace and hyphens, dropping empty strings, and taking the union across all three fields. Used for order-invariant set-overlap / Jaccard matching. Robust to field misassignment, name-order swaps, spacing, and hyphenation differences (`ANNE-MARIE` ≡ `ANNE MARIE` ≡ `ANNEMARIE` after token derivation). Apostrophes are not split characters, so `O'BRIEN` remains a single token. Although Python sets are unordered in memory, when serialized for storage or display (Parquet, JSON, this document, etc.) sort tokens **alphabetically** for a canonical, reproducible representation.
- `full_name_compact` — concatenate `FirstNM_clean`, `MiddleNM_clean`, and `LastNM_clean` (in that order, skipping nulls), strip all non-letter characters. Both `("MARY SMITH", null, null)` and `("MARY", null, "SMITH")` collapse to `MARYSMITH`, so first/last misassignment becomes a non-issue for matching. Note that this artifact is order-dependent — for cases where the same person appears with name-order swaps across records (e.g., Vietnamese native order vs. US order), `full_name_tokens` is the matching artifact that handles this correctly.


## Phones consolidation
Per-slot cleaning of `PrimaryPhoneNBR`, `Phone01NBR`, `Phone02NBR`, `Phone03NBR` is performed independently and stored in `<field>_clean` columns. After cleaning, build the following derived field for record linkage:
- `Phones_set` — sorted, deduplicated set of string of all non-null cleaned phone numbers across the four slots.

## Address parsing and standardization (libpostal)
After `AddressLine1_clean`, `AddressLine2_clean`, `CityNM_clean`, `StateCD_clean`, and `ZipCD_clean` have been computed, derive a structured, standardized address representation using the `libpostal` library. Passing the full address (not just the street lines) gives libpostal enough context to parse and expand accurately.
- Build the input string by concatenating, in order and skipping nulls, separated by a comma and space:
  - `AddressLine1_clean`
  - `AddressLine2_clean`
  - `CityNM_clean`
  - `StateCD_clean`
  - `ZipCD_clean_base` (and `ZipCD_clean_ext` appended with a hyphen if present)
- Pass the concatenated string through libpostal's `expand_address` to generate the canonical/normalized form
- Pass the concatenated string through libpostal's `parse_address` to extract structured components
- Store the normalized full address in a new field `Address_normalized` (libpostal canonical expansion). When `expand_address` returns multiple expansions, retain the first result. If libpostal returns no parse (empty result), leave the field as `NaN`.


# Data Cleaning Output
The data cleaning output will be a new csv file that will contain the raw and new columns. This file will be versioned, always being a higher version that the existing files in the output folder. v1,v2,v3,...
