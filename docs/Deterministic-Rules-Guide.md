# Deterministic Matching Rules Guide

This document describes the deterministic matching rules used in the EMPI (Enterprise Master Patient Index) pipeline to identify potential duplicate patient records.

## Overview

The deterministic matching approach uses blocking strategies combined with exact-match rules to efficiently identify candidate record pairs that likely represent the same patient. The process follows a two-stage approach:

1. **Blocking**: Group records by shared attributes to reduce the comparison space
2. **Rule Application**: Apply deterministic rules within blocks to identify matches

## Blocking Strategies

Blocking reduces the O(n²) comparison problem by grouping records on shared keys. Each block key is designed to capture different match scenarios while preventing cartesian explosion.

### Block Definitions

| Block | Key Composition | Purpose |
|-------|----------------|---------|
| B1 | `SSN_clean` | Direct SSN matches |
| B3 | `LastNM_clean` + `BirthDT_clean` | Name + exact DOB |
| B4 | `LastNM_clean` + `BirthYear` + `FirstNM_clean[:3]` | Name + birth year + first name prefix |
| B5 | `PhonePrimary_clean` | Phone number matches |
| B6 | `Email_clean` | Email matches |
| B7 | `LastNM_clean` + `ZipCD_base` + `BirthYear` | Name + location + birth year |
| B8 | `FirstNM_clean[:1]` + `LastNM_clean[:1]` + `BirthYear` | Initial-based broad block |
| B9 | `LastNM_clean` + `FirstNM_clean` + `SSN_Last4` | Full name + last 4 SSN |

### Block Safety Controls

To prevent memory issues and false positive clusters:
- **Maximum block size**: 3,000 records per block key
- **Minimum non-null ratio**: 30% (blocks with excessive nulls are skipped)
- Records with null keys are excluded from their respective blocks

---

## Match Rules

### Rule 1: EXACT_SSN
**Confidence: 1.00**

Matches records with identical Social Security Numbers.

```
Conditions:
- SSN_clean_L is not null
- SSN_clean_L == SSN_clean_R
```

**Quality Metrics:**
- First Name Agreement: 83.42%
- Last Name Agreement: 74.68%
- DOB Agreement: 87.39%
- Sex Agreement: 77.51%
- Quality Score: 0.8075

**Notes:** SSN matches may have name/DOB discrepancies due to data entry errors, name changes, or SSN reuse issues.

---

### Rule 2: NAME_DOB_PHONE
**Confidence: 0.985**

Matches records with identical first name, last name, date of birth, and phone number.

```
Conditions:
- FirstNM_clean_L == FirstNM_clean_R
- LastNM_clean_L == LastNM_clean_R
- BirthDT_clean_L == BirthDT_clean_R
- PhonePrimary_clean_L == PhonePrimary_clean_R
```

**Quality Metrics:**
- First Name Agreement: 100%
- Last Name Agreement: 100%
- DOB Agreement: 100%
- Sex Agreement: 72.52%
- Quality Score: 0.9313

---

### Rule 3: NAME_DOB_EMAIL
**Confidence: 0.99**

Matches records with identical first name, last name, date of birth, and email address.

```
Conditions:
- FirstNM_clean_L == FirstNM_clean_R
- LastNM_clean_L == LastNM_clean_R
- BirthDT_clean_L == BirthDT_clean_R
- Email_clean_L == Email_clean_R
```

**Notes:** Currently not producing matches in the output (may be absorbed by other rules).

---

### Rule 4: NAME_DOB_ADDRESS
**Confidence: 0.97**

Matches records with identical first name, last name, date of birth, and street address.

```
Conditions:
- FirstNM_clean_L == FirstNM_clean_R
- LastNM_clean_L == LastNM_clean_R
- BirthDT_clean_L == BirthDT_clean_R
- AddressLine1_clean_L == AddressLine1_clean_R
```

**Quality Metrics:**
- First Name Agreement: 100%
- Last Name Agreement: 100%
- DOB Agreement: 100%
- Sex Agreement: 0% (likely due to missing sex data)
- Quality Score: 0.75

---

### Rule 5: NAME_DOB_SEX
**Confidence: 0.98**

Matches records with identical first name, last name, date of birth, and sex at birth.

```
Conditions:
- FirstNM_clean_L == FirstNM_clean_R
- LastNM_clean_L == LastNM_clean_R
- BirthDT_clean_L == BirthDT_clean_R
- SexAtBirthDSC_clean_L == SexAtBirthDSC_clean_R
```

**Quality Metrics:**
- First Name Agreement: 100%
- Last Name Agreement: 100%
- DOB Agreement: 100%
- Sex Agreement: 100%
- Quality Score: 1.0000

**Notes:** Highest quality rule by agreement metrics. Most frequently triggered rule.

---

### Rule 6: EMAIL_EXACT
**Confidence: 0.995**

Matches records with identical email addresses.

```
Conditions:
- Email_clean_L is not null
- Email_clean_L == Email_clean_R
```

**Quality Metrics:**
- First Name Agreement: 28.26%
- Last Name Agreement: 28.26%
- DOB Agreement: 32.78%
- Sex Agreement: 55.26%
- Quality Score: 0.3614

**Notes:** Lowest quality score. Email sharing (family accounts, shared devices) causes many false positives. Consider downgrading confidence or adding secondary conditions.

---

## Results Summary

### Match Distribution

| Match Rule | Count | % of Matches |
|------------|-------|--------------|
| NAME_DOB_SEX | 18,094 | 38.8% |
| EMAIL_EXACT | 12,599 | 27.0% |
| NAME_DOB_PHONE | 8,723 | 18.7% |
| EXACT_SSN | 5,695 | 12.2% |
| NAME_DOB_ADDRESS | 1,558 | 3.3% |
| **Total** | **46,669** | **100%** |

### Coverage Statistics

| Metric | Value |
|--------|-------|
| Total Patients | 163,364 |
| Patients in ≥1 Match | 54,900 |
| Unmatched Patients | 108,464 |
| Coverage Rate | 33.61% |

### Quality Indicators

| Metric | Value |
|--------|-------|
| Total Matches | 46,669 |
| Average Match Confidence | 0.9871 |
| Suspicious Match Rate | 23.68% |
| Maximum Cluster Size | 91 |

---

## Suspicious Match Analysis

Suspicious matches are defined as pairs where:
- DOB differs between records, OR
- Last name differs between records, OR
- Both SSNs are present but differ

**Total suspicious matches: 11,052 (23.68% of all matches)**

Common suspicious match patterns include:
- Minor spelling variations in last names (e.g., `WOOLFOLK` vs `WOOLSOLK`)
- Single-digit DOB differences (potential data entry errors)
- Hyphenation differences (e.g., `HR-DELGADO` vs `HR-OYOLA`)
- Suffix handling issues (e.g., `BUCKLEYJR` vs `BUCKLEY`)

---

## Cluster Analysis

Large match clusters may indicate:
- Bad blocking keys capturing unrelated records
- Common placeholder values not filtered
- Overly broad matching rules

**Largest clusters observed: 91 linked records**

Clusters of this size warrant manual review to identify potential false positive chains.

---

## Quality Score Interpretation

The quality score is a composite of agreement rates across first name, last name, DOB, and sex:

| Score Range | Interpretation |
|-------------|----------------|
| > 0.95 | Excellent deterministic rule |
| 0.85 - 0.95 | Strong rule |
| 0.70 - 0.85 | Moderate rule (review recommended) |
| < 0.70 | Risky/noisy rule (consider removal or additional conditions) |

### Rule Rankings by Quality

1. **NAME_DOB_SEX** (1.0000) - Excellent
2. **NAME_DOB_PHONE** (0.9313) - Strong
3. **EXACT_SSN** (0.8075) - Moderate
4. **NAME_DOB_ADDRESS** (0.7500) - Moderate
5. **EMAIL_EXACT** (0.3614) - Risky


