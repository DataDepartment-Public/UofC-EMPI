"""Data transformations for the MDM_Population dataset.

Faithful implementation of docs/Data-Cleaning-Guide.md. Produces, for each
cleaned input column, a `<col>_raw` (original, never modified) and a
`<col>_clean` (transformed) column, plus a single record-level boolean
`valid_record` and the derived cross-field artifacts the MPI/entity-resolution
pipeline consumes (`last_4_SSN`, `ZipCD_clean_base`/`_ext`,
`full_name_tokens`, `full_name_compact`, `Phones_set`, `Address_normalized`).
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from stdnum.us import ssn as _us_ssn
from unidecode import unidecode

try:
    from postal.expand import expand_address as _expand_address
    _LIBPOSTAL_AVAILABLE = True
except Exception:
    _LIBPOSTAL_AVAILABLE = False


# =============================================================================
# CONSTANTS — verbatim from docs/Data-Cleaning-Guide.md
# =============================================================================

NAME_NULL_VALUES = frozenset({'UNKNOWN', 'NULL', 'NAN', 'NONE', 'N/A', 'NA', ''})
MIDDLE_NULL_VALUES = frozenset({'NMI', 'UNKNOWN', 'NULL', 'N/A', '-', ''})
GENERIC_NULL_VALUES = frozenset({'UNKNOWN', 'NULL', 'NAN', 'NONE', 'N/A', 'NA', ''})
EMAIL_NULL_VALUES = frozenset({'unknown', 'null', 'nan', 'none', 'n/a', 'na', ''})
SEX_NULL_VALUES = frozenset({'UNKNOWN', 'NULL', 'N/A', ''})

INVALID_CONTAINS = (
    'BABYBOY', 'BABY BOY', 'BABYGIRL', 'BABY GIRL',
    'DUPLICATE DO NOT USE', 'DUPLICATE DONT USE',
    'DO NOT USE DUPLICATE', 'DO NOT USE DOUBLE ACCOUNT',
    'DO NOT USE DOBBLE ACCOUNT', 'DUPLICATE ACCOUNT',
    'DOUBLE ACCOUNT', 'DONOT USED',
    'DO NOT USE', 'DONOTUSE', 'DONT USE', "DON'T USE",
    'DUPLICATE', 'DO NOT', 'MEDICARE', 'ACCOUNT', '<MRG>',
    'INACTIVE', 'UPRR',
)

INVALID_EQUALS = frozenset({'TEST', 'BABY'})

# Numeric-only or ID-prefixed-digits patterns in name fields (e.g., `12345`,
# `ID4567`). Tested against the raw value, since downstream cleaning strips
# digits from names.
NUMERIC_NAME_PATTERN = re.compile(r"^\d+$|^[Ii][Dd]\d+$")

ADDRESS_PLACEHOLDERS = frozenset({
    'HOMELESS', 'TRANSIENT', 'BAD ADDRESS', 'NO ADDRESS', '?',
    'NO MAIL', 'GENERAL DELIVERY', 'NKA', '.', 'NOT TAKEN',
    'ZOCDOC', 'GET', 'HOPE CENTER', 'INCORRECT ADDRESS', '<MRG>', 'X',
})

ADDRESS2_PLACEHOLDERS = frozenset({
    'HOMELESS', 'TRANSIENT', 'BAD ADDRESS', 'NO ADDRESS', '?',
    'NO MAIL', 'GENERAL DELIVERY', 'NKA', '.', 'NOT TAKEN',
    'ZOCDOC', 'GET', 'HOPE CENTER', 'INCORRECT ADDRESS',
})

TITLES = ('MR', 'MRS', 'MS', 'DR', 'REV')

GENERATIONAL_SUFFIXES = frozenset({'JR', 'SR', 'I', 'II', 'III', 'IV', 'V'})
LASTNM_TRAILING_SUFFIXES = ('III', 'IV', 'II', 'JR', 'SR', 'V')
ORDINAL_TO_ROMAN = {'2ND': 'II', '3RD': 'III', '4TH': 'IV', '5TH': 'V'}

STREET_SUFFIX_MAP = {
    'STREET': 'ST', 'STR': 'ST', 'STRT': 'ST',
    'AVENUE': 'AVE', 'AV': 'AVE', 'AVN': 'AVE', 'AVENU': 'AVE',
    'BOULEVARD': 'BLVD', 'BOUL': 'BLVD', 'BOULV': 'BLVD',
    'ROAD': 'RD',
    'DRIVE': 'DR', 'DRIV': 'DR', 'DRV': 'DR',
    'LANE': 'LN',
    'COURT': 'CT', 'CRT': 'CT',
    'PLACE': 'PL',
    'CIRCLE': 'CIR', 'CIRC': 'CIR', 'CRCL': 'CIR', 'CRCLE': 'CIR',
    'TERRACE': 'TER', 'TERR': 'TER',
    'PARKWAY': 'PKWY', 'PARKWY': 'PKWY', 'PKWAY': 'PKWY', 'PKY': 'PKWY',
    'HIGHWAY': 'HWY', 'HIGHWY': 'HWY', 'HIWAY': 'HWY', 'HIWY': 'HWY', 'HWAY': 'HWY',
    'TRAIL': 'TRL', 'TRL': 'TRL',
    'ALLEY': 'ALY', 'ALLY': 'ALY', 'ALLEE': 'ALY',
    'SQUARE': 'SQ', 'SQR': 'SQ', 'SQRE': 'SQ', 'SQU': 'SQ',
    'PLAZA': 'PLZ', 'PLZA': 'PLZ',
    'EXPRESSWAY': 'EXPY', 'EXPR': 'EXPY', 'EXPW': 'EXPY',
    'FREEWAY': 'FWY', 'FRWAY': 'FWY', 'FRWY': 'FWY',
    'CROSSING': 'XING', 'CRSSNG': 'XING',
    'BRIDGE': 'BRG', 'BRDGE': 'BRG',
    'ROUTE': 'RTE',
    'TURNPIKE': 'TPKE', 'TRNPK': 'TPKE',
    'POINT': 'PT', 'POINTS': 'PT',
    'RIDGE': 'RDG',
    'VIEW': 'VW',
    'VILLAGE': 'VLG', 'VILLG': 'VLG', 'VILLIAGE': 'VLG',
    'ESTATES': 'EST',
    'HEIGHTS': 'HTS',
}

DIRECTION_MAP = {
    'NORTH': 'N', 'SOUTH': 'S', 'EAST': 'E', 'WEST': 'W',
    'NORTHEAST': 'NE', 'NORTHWEST': 'NW',
    'SOUTHEAST': 'SE', 'SOUTHWEST': 'SW',
}

UNIT_DESIGNATOR_MAP = {
    'APARTMENT': 'APT', 'APRT': 'APT',
    'SUITE': 'STE',
    'ROOM': 'RM',
    'FLOOR': 'FL', 'FLR': 'FL',
    'BUILDING': 'BLDG', 'BLD': 'BLDG', 'BLDNG': 'BLDG',
    'DEPARTMENT': 'DEPT',
    'PENTHOUSE': 'PH',
    'BASEMENT': 'BSMT',
    'LOWER': 'LOWR',
    'UPPER': 'UPPR',
    'TRAILER': 'TRLR',
    'HANGAR': 'HNGR',
    'OFFICE': 'OFC',
    'FRONT': 'FRNT',
    'LOBBY': 'LBBY',
    'SPACE': 'SPC',
}

STATE_NAME_TO_CODE = {
    'ALABAMA': 'AL', 'ALASKA': 'AK', 'ARIZONA': 'AZ', 'ARKANSAS': 'AR',
    'CALIFORNIA': 'CA', 'COLORADO': 'CO', 'CONNECTICUT': 'CT', 'DELAWARE': 'DE',
    'FLORIDA': 'FL', 'GEORGIA': 'GA', 'HAWAII': 'HI', 'IDAHO': 'ID',
    'ILLINOIS': 'IL', 'INDIANA': 'IN', 'IOWA': 'IA', 'KANSAS': 'KS',
    'KENTUCKY': 'KY', 'LOUISIANA': 'LA', 'MAINE': 'ME', 'MARYLAND': 'MD',
    'MASSACHUSETTS': 'MA', 'MICHIGAN': 'MI', 'MINNESOTA': 'MN',
    'MISSISSIPPI': 'MS', 'MISSOURI': 'MO', 'MONTANA': 'MT', 'NEBRASKA': 'NE',
    'NEVADA': 'NV', 'NEW HAMPSHIRE': 'NH', 'NEW JERSEY': 'NJ',
    'NEW MEXICO': 'NM', 'NEW YORK': 'NY', 'NORTH CAROLINA': 'NC',
    'NORTH DAKOTA': 'ND', 'OHIO': 'OH', 'OKLAHOMA': 'OK', 'OREGON': 'OR',
    'PENNSYLVANIA': 'PA', 'RHODE ISLAND': 'RI', 'SOUTH CAROLINA': 'SC',
    'SOUTH DAKOTA': 'SD', 'TENNESSEE': 'TN', 'TEXAS': 'TX', 'UTAH': 'UT',
    'VERMONT': 'VT', 'VIRGINIA': 'VA', 'WASHINGTON': 'WA',
    'WEST VIRGINIA': 'WV', 'WISCONSIN': 'WI', 'WYOMING': 'WY',
    'DISTRICT OF COLUMBIA': 'DC',
    'PUERTO RICO': 'PR', 'VIRGIN ISLANDS': 'VI', 'GUAM': 'GU',
    'AMERICAN SAMOA': 'AS', 'NORTHERN MARIANA ISLANDS': 'MP',
}

# CMS Certification Number state prefixes (Pub 100-07 SOM Appendix A).
CCN_STATE_CODES = {
    '01': 'AL', '02': 'AK', '03': 'AZ', '04': 'AR', '05': 'CA',
    '06': 'CO', '07': 'CT', '08': 'DE', '09': 'DC', '10': 'FL',
    '11': 'GA', '12': 'HI', '13': 'ID', '14': 'IL', '15': 'IN',
    '16': 'IA', '17': 'KS', '18': 'KY', '19': 'LA', '20': 'ME',
    '21': 'MD', '22': 'MA', '23': 'MI', '24': 'MN', '25': 'MS',
    '26': 'MO', '27': 'MT', '28': 'NE', '29': 'NV', '30': 'NH',
    '31': 'NJ', '32': 'NM', '33': 'NY', '34': 'NC', '35': 'ND',
    '36': 'OH', '37': 'OK', '38': 'OR', '39': 'PA', '40': 'PR',
    '41': 'RI', '42': 'SC', '43': 'SD', '44': 'TN', '45': 'TX',
    '46': 'UT', '47': 'VT', '48': 'VA', '49': 'VI', '50': 'WA',
    '51': 'WV', '52': 'WI', '53': 'WY', '54': 'GU',
}

US_STATE_CODES = frozenset(STATE_NAME_TO_CODE.values())

JUNK_EMAIL_EXACT = frozenset({
    'noemail@noemail.com', 'noemail@textmessage.com',
    'noemail@textmessaging.com', 'aca@eriefamilyhealth.org',
    'aca@eriefamilyheath.org', 'declined@hamakua-health.org',
    'aca@eriefamilyhealth.com', 'noemail@textmessages.com',
    'decline@hamakua-health.org', 'noemai@noemail.com',
    'patientdeclined@howardbrown.org', 'ptdeclined@hb.org',
    'noemail@email.com', 'no@email.com', 'none@none.com',
    'unknown@unknown.com', 'unknown@email.com', 'test@test.com',
    'test@example.com', 'donotreply@donotreply.com',
    'noreply@noreply.com', 'email@email.com', 'patient@patient.com',
    'none@noemail.com', 'na@na.com', 'null@null.com', 'none@gmail.com',
})

JUNK_EMAIL_CONTAINS = ('decline', 'noemail')

JUNK_EMAIL_LOCAL_PREFIXES = (
    'noemail', 'no-email', 'no_email', 'noreply', 'no-reply',
    'donotreply', 'do-not-reply', 'unknown', 'test', 'patient',
    'none', 'null', 'na',
)

JUNK_EMAIL_DOMAINS = frozenset({
    'example.com', 'example.org', 'example.net',
    'test.com', 'test.org', 'noemail.com', 'noreply.com',
    'donotreply.com', 'unknown.com', '123.com',
})

JUNK_SSN_EXACT = frozenset({
    '010101010', '090909090', '000000001', '999999998',
    '111223333', '219099999', '457555462',
    '333333330', '003333333', '333333300', '033333333',
    '333333339', '099999999', '333333000',
})

# A structurally valid SSN whose digits carry almost no information is a
# placeholder a clerk typed to satisfy a required field (e.g. 333333330,
# 003333333, 012345678). These pass area/group/serial structural checks AND
# stdnum.us.ssn validation, yet on the real AllianceChicago data a single such
# value (333333330) chained 22 unrelated patients into one EXACT_SSN cluster.
# We treat an SSN as a placeholder when it has <=2 distinct digits, one digit
# fills >=7 of the 9 positions, or it is a full ascending/descending digit run.
# A genuine SSN essentially never looks like this.
_SSN_MAX_DIGIT_DOMINANCE = 7


def _is_sequential_ssn(digits: str) -> bool:
    """True for a full ascending or descending consecutive-digit run (mod 10).

    Catches 012345678, 123456789, 234567890, 987654321, ... generically rather
    than relying on a hardcoded enumeration.
    """
    d = [int(c) for c in digits]
    asc = all((d[i + 1] - d[i]) % 10 == 1 for i in range(len(d) - 1))
    desc = all((d[i] - d[i + 1]) % 10 == 1 for i in range(len(d) - 1))
    return asc or desc


def _is_placeholder_ssn(digits: str) -> bool:
    """True for low-entropy SSNs that are clerical placeholders, not identities."""
    if len(set(digits)) <= 2:
        return True
    if Counter(digits).most_common(1)[0][1] >= _SSN_MAX_DIGIT_DOMINANCE:
        return True
    return _is_sequential_ssn(digits)

INVALID_NPA_N11 = frozenset({'211', '311', '411', '511', '611', '711', '811', '911'})
SEQUENTIAL_PHONES = frozenset({
    '0123456789', '1234567890', '9876543210', '0987654321',
})

PLACEHOLDER_ZIPS_ALL_SAME = frozenset({
    '00000', '11111', '22222', '33333', '44444',
    '55555', '66666', '77777', '88888', '99999',
})
PLACEHOLDER_ZIPS_SEQUENTIAL = frozenset({'12345', '54321'})

# ZIP first digit → set of US state codes whose ZIPs start with that digit
# (USPS sectional center geography). Used for the "Cross-check ZIP against
# state" rule.
ZIP_PREFIX_TO_STATES = {
    '0': {'CT', 'MA', 'ME', 'NH', 'NJ', 'PR', 'RI', 'VT', 'VI'},
    '1': {'DE', 'NY', 'PA'},
    '2': {'DC', 'MD', 'NC', 'SC', 'VA', 'WV'},
    '3': {'AL', 'FL', 'GA', 'MS', 'TN'},
    '4': {'IN', 'KY', 'MI', 'OH'},
    '5': {'IA', 'MN', 'MT', 'ND', 'SD', 'WI'},
    '6': {'IL', 'KS', 'MO', 'NE'},
    '7': {'AR', 'LA', 'OK', 'TX'},
    '8': {'AZ', 'CO', 'ID', 'NM', 'NV', 'UT', 'WY'},
    '9': {'AK', 'AS', 'CA', 'GU', 'HI', 'MP', 'OR', 'WA'},
}


# =============================================================================
# UTILITIES
# =============================================================================

def _apply_unicode(s: str) -> str:
    """Transliterate `s` to ASCII via unidecode. See guide §Unicode."""
    return unidecode(s)


def _split_camelcase(s: str) -> str:
    """Insert a space at every lowercase→uppercase boundary."""
    return re.sub(r'([a-z])([A-Z])', r'\1 \2', s)


def _collapse_ws(s: str) -> str:
    return re.sub(r'\s+', ' ', s).strip()


def _contains_any(text: str, patterns) -> bool:
    return any(p in text for p in patterns)


def _whole_word_replace(text: str, mapping: dict) -> str:
    """Token-level (split-on-whitespace) replacement using `mapping`.
    Embedded substrings inside real words are preserved."""
    if not text:
        return text
    return ' '.join(mapping.get(tok, tok) for tok in text.split(' '))


def _strip_leading_zeros_first_token(text: str) -> str:
    """If the first whitespace-delimited token is all digits, drop its
    leading zeros (preserving at least one digit). Later digit runs untouched."""
    if not text:
        return text
    parts = text.split(' ', 1)
    head = parts[0]
    if head.isdigit():
        stripped = head.lstrip('0') or '0'
        parts[0] = stripped
    return ' '.join(parts)


# =============================================================================
# PER-FIELD CLEANERS
# =============================================================================

def clean_first_name(value) -> Tuple[Optional[str], bool, Optional[str]]:
    """Clean a `FirstNM` value. Returns (cleaned, is_invalid, moved_suffix).

    A trailing generational suffix is removed from the first name and returned
    as `moved_suffix` so the orchestrator can move it into `SuffixNM_clean`."""
    if pd.isna(value):
        return np.nan, False, np.nan

    raw = str(value).strip()
    is_invalid = bool(NUMERIC_NAME_PATTERN.match(raw))

    text = _apply_unicode(raw)
    text = _split_camelcase(text)
    text = _collapse_ws(text.upper())

    is_invalid = is_invalid or _contains_any(text, INVALID_CONTAINS)

    text = re.sub(r"[^A-Z \-']", '', text)
    text = _collapse_ws(text)

    if text in NAME_NULL_VALUES:
        return np.nan, is_invalid, np.nan

    if text in INVALID_EQUALS:
        is_invalid = True

    for title in TITLES:
        if text.startswith(f'{title} '):
            text = text[len(title) + 1:]
            break

    moved_suffix = np.nan
    for suffix in LASTNM_TRAILING_SUFFIXES:
        sentinel = f' {suffix}'
        if text.endswith(sentinel):
            text = text[: -len(sentinel)]
            moved_suffix = suffix
            break

    text = _collapse_ws(text)
    if not text:
        return np.nan, is_invalid, moved_suffix

    return text, is_invalid, moved_suffix


def clean_last_name(value) -> Tuple[Optional[str], bool, Optional[str]]:
    """Clean a `LastNM` value. Returns (cleaned, is_invalid, moved_suffix).

    A trailing generational suffix is removed from the surname and returned as
    `moved_suffix` so the orchestrator can move it into `SuffixNM_clean`."""
    if pd.isna(value):
        return np.nan, False, np.nan

    raw = str(value).strip()
    is_invalid = bool(NUMERIC_NAME_PATTERN.match(raw))

    text = _apply_unicode(raw)
    text = _split_camelcase(text)
    text = _collapse_ws(text.upper())

    is_invalid = is_invalid or _contains_any(text, INVALID_CONTAINS)

    text = re.sub(r"[^A-Z \-']", '', text)
    text = _collapse_ws(text)

    if text in NAME_NULL_VALUES:
        return np.nan, is_invalid, np.nan

    if text in INVALID_EQUALS:
        is_invalid = True

    moved_suffix = np.nan
    for suffix in LASTNM_TRAILING_SUFFIXES:
        sentinel = f' {suffix}'
        if text.endswith(sentinel):
            text = text[: -len(sentinel)]
            moved_suffix = suffix
            break

    text = _collapse_ws(text)
    if not text:
        return np.nan, is_invalid, moved_suffix

    return text, is_invalid, moved_suffix


def clean_middle_name(value) -> Tuple[Optional[str], bool, Optional[str]]:
    """Clean a `MiddleNM` value. Returns (cleaned, is_invalid, moved_suffix).

    A trailing generational suffix is removed from the middle name and returned
    as `moved_suffix` so the orchestrator can move it into `SuffixNM_clean`."""
    if pd.isna(value):
        return np.nan, False, np.nan

    raw = str(value).strip()
    is_invalid = bool(NUMERIC_NAME_PATTERN.match(raw))

    text = _apply_unicode(raw)
    text = _split_camelcase(text)
    text = _collapse_ws(text.upper())

    is_invalid = is_invalid or _contains_any(text, INVALID_CONTAINS)

    if text in INVALID_EQUALS:
        is_invalid = True

    if text in MIDDLE_NULL_VALUES:
        return np.nan, is_invalid, np.nan

    text = re.sub(r"[^A-Z \-']", '', text)
    text = _collapse_ws(text)

    if not text or text in MIDDLE_NULL_VALUES:
        return np.nan, is_invalid, np.nan

    moved_suffix = np.nan
    for suffix in LASTNM_TRAILING_SUFFIXES:
        sentinel = f' {suffix}'
        if text.endswith(sentinel):
            text = text[: -len(sentinel)]
            moved_suffix = suffix
            break

    text = _collapse_ws(text)
    if not text:
        return np.nan, is_invalid, moved_suffix

    return text, is_invalid, moved_suffix


def clean_suffix(value) -> Optional[str]:
    """Clean a `SuffixNM` value."""
    if pd.isna(value):
        return np.nan

    text = _collapse_ws(str(value).upper())

    if text in GENERIC_NULL_VALUES:
        return np.nan

    text = re.sub(r'[^\w\s]', '', text)
    text = _collapse_ws(text)

    if text in ORDINAL_TO_ROMAN:
        text = ORDINAL_TO_ROMAN[text]

    if text in GENERATIONAL_SUFFIXES:
        return text

    return np.nan


def clean_birth_date(value, reference_date=None):
    """Clean a `BirthDT` value. Returns pd.Timestamp or pd.NaT."""
    if pd.isna(value):
        return pd.NaT

    if reference_date is None:
        reference_date = pd.Timestamp.today().normalize()
    else:
        reference_date = pd.Timestamp(reference_date)

    parsed = pd.to_datetime(value, errors='coerce')
    if pd.isna(parsed):
        return pd.NaT

    cutoff = pd.Timestamp('1900-01-01')
    if parsed <= cutoff:
        return pd.NaT
    if parsed > reference_date:
        return pd.NaT

    return parsed


def clean_ssn(value) -> Tuple[Optional[str], Optional[str]]:
    """Clean an `SSN` value. Returns (clean_ssn, last_4_SSN)."""
    if pd.isna(value):
        return np.nan, np.nan

    digits = re.sub(r'\D', '', str(value))

    if len(digits) in (7, 8):
        digits = digits.zfill(9)

    if len(digits) != 9:
        return np.nan, np.nan

    if len(set(digits)) == 1:
        return np.nan, np.nan

    if digits[:3] == digits[3:6] == digits[6:9]:
        return np.nan, np.nan

    area = digits[:3]
    if area == '000' or area == '666':
        return np.nan, np.nan
    if 900 <= int(area) <= 999:
        return np.nan, np.nan
    if digits[3:5] == '00':
        return np.nan, np.nan
    if digits[5:9] == '0000':
        return np.nan, np.nan

    if digits in JUNK_SSN_EXACT:
        return np.nan, np.nan

    # Low-entropy placeholders (e.g. 333333330) and full ascending/descending
    # digit runs (e.g. 012345678) pass every structural check above but are not
    # real identities — reject them before they create EXACT_SSN false-merges.
    if _is_placeholder_ssn(digits):
        return np.nan, np.nan

    # Final gate: stdnum's US SSN validator (known-bogus advertising SSNs,
    # invalid area/group/serial combinations the rules above don't enumerate).
    if not _us_ssn.is_valid(digits):
        return np.nan, np.nan

    # Extract last 4 only from a fully validated SSN.
    last_4 = digits[-4:]
    return digits, last_4


def clean_address_line1(value) -> Tuple[Optional[str], bool]:
    """Clean an `AddressLine1` value. Returns (cleaned, is_invalid)."""
    if pd.isna(value):
        return np.nan, False

    text = _apply_unicode(str(value))
    text = _collapse_ws(text.upper())

    if text in GENERIC_NULL_VALUES or text in ADDRESS_PLACEHOLDERS:
        return np.nan, False

    is_invalid = _contains_any(text, INVALID_CONTAINS)
    if text in INVALID_EQUALS:
        is_invalid = True

    text = re.sub(r'[^A-Z0-9 ]', '', text)
    text = _collapse_ws(text)

    text = _whole_word_replace(text, STREET_SUFFIX_MAP)
    text = _whole_word_replace(text, DIRECTION_MAP)
    text = _strip_leading_zeros_first_token(text)

    text = _collapse_ws(text)
    if not text:
        return np.nan, is_invalid

    return text, is_invalid


def clean_address_line2(value) -> Tuple[Optional[str], bool]:
    """Clean an `AddressLine2` value. Returns (cleaned, is_invalid)."""
    if pd.isna(value):
        return np.nan, False

    text = _apply_unicode(str(value))
    text = _collapse_ws(text.upper())

    if text in GENERIC_NULL_VALUES or text in ADDRESS2_PLACEHOLDERS:
        return np.nan, False

    is_invalid = _contains_any(text, INVALID_CONTAINS)
    if text in INVALID_EQUALS:
        is_invalid = True

    text = re.sub(r'[^A-Z0-9 ]', '', text)
    text = _collapse_ws(text)

    text = _whole_word_replace(text, UNIT_DESIGNATOR_MAP)

    text = _collapse_ws(text)
    if not text:
        return np.nan, is_invalid

    return text, is_invalid


def clean_city(value) -> Optional[str]:
    """Clean a `CityNM` value."""
    if pd.isna(value):
        return np.nan

    text = _apply_unicode(str(value))
    text = _collapse_ws(text.upper())

    if text in GENERIC_NULL_VALUES:
        return np.nan

    text = re.sub(r"[^A-Z \-']", '', text)
    text = _collapse_ws(text)

    if not text:
        return np.nan

    return text


def clean_zip(value) -> Tuple[Optional[str], Optional[str]]:
    """Clean a `ZipCD` value. Returns (base, ext)."""
    if pd.isna(value):
        return np.nan, np.nan

    digits = re.sub(r'\D', '', str(value))

    if len(digits) == 4:
        digits = digits.zfill(5)
    elif len(digits) == 8:
        digits = digits.zfill(9)

    if len(digits) >= 5:
        base = digits[:5]
    else:
        return np.nan, np.nan

    ext = digits[5:9] if len(digits) >= 9 else np.nan
    if isinstance(ext, str) and len(ext) != 4:
        ext = np.nan

    if len(base) != 5:
        return np.nan, np.nan

    if base in PLACEHOLDER_ZIPS_ALL_SAME or base in PLACEHOLDER_ZIPS_SEQUENTIAL:
        return np.nan, np.nan

    return base, ext


def clean_state(value) -> Optional[str]:
    """Clean a `StateCD` value."""
    if pd.isna(value):
        return np.nan

    text = str(value).upper().strip()

    if text in GENERIC_NULL_VALUES:
        return np.nan

    if text in STATE_NAME_TO_CODE:
        return STATE_NAME_TO_CODE[text]

    if text.isdigit():
        if len(text) == 1:
            text = text.zfill(2)
        return CCN_STATE_CODES.get(text, np.nan)

    if text in US_STATE_CODES:
        return text

    return np.nan


def clean_phone(value) -> Optional[str]:
    """Clean a phone slot (PrimaryPhoneNBR / Phone01NBR / Phone02NBR / Phone03NBR)."""
    if pd.isna(value):
        return np.nan

    digits = re.sub(r'\D', '', str(value))

    if len(digits) == 11 and digits.startswith('1'):
        digits = digits[1:]

    if len(digits) != 10:
        return np.nan

    if len(set(digits)) == 1:
        return np.nan

    if digits in SEQUENTIAL_PHONES:
        return np.nan

    npa = digits[:3]
    nxx = digits[3:6]

    if npa[0] in '01':
        return np.nan
    if npa in INVALID_NPA_N11:
        return np.nan
    if npa == '555':
        return np.nan

    if nxx[0] in '01':
        return np.nan
    if digits[3:7] in {'5550', '5551'}:
        return np.nan

    return digits


def clean_email(value) -> Optional[str]:
    """Clean an `Email` value."""
    if pd.isna(value):
        return np.nan

    text = re.sub(r'\s+', '', str(value).lower())

    if not re.match(r'^[^\s@]+@[^\s@]+\.[^\s@]+$', text):
        return np.nan

    if text in EMAIL_NULL_VALUES:
        return np.nan

    if '@' not in text:
        return np.nan

    if text in JUNK_EMAIL_EXACT:
        return np.nan

    if any(needle in text for needle in JUNK_EMAIL_CONTAINS):
        return np.nan

    local, _, domain = text.partition('@')

    if any(local.startswith(p) for p in JUNK_EMAIL_LOCAL_PREFIXES):
        return np.nan

    if len(local) <= 2:
        return np.nan

    if '123456' in local:
        return np.nan

    if domain in JUNK_EMAIL_DOMAINS:
        return np.nan

    return text


def clean_sex_at_birth(value) -> Optional[str]:
    """Clean a `SexAtBirthDSC` value. Retains MALE/FEMALE/OTHER."""
    if pd.isna(value):
        return np.nan

    text = str(value).upper().strip()

    if text in SEX_NULL_VALUES:
        return np.nan

    if text in {'MALE', 'FEMALE', 'OTHER'}:
        return text

    return np.nan


# =============================================================================
# CROSS-FIELD DERIVATIONS
# =============================================================================

def derive_full_name_tokens(first_clean, middle_clean, last_clean) -> list:
    """Split each name field on whitespace and hyphens, union the tokens.
    Apostrophes are not split. Returns a sorted list for canonical serialization."""
    tokens: set = set()
    for part in (first_clean, middle_clean, last_clean):
        if pd.isna(part):
            continue
        for tok in re.split(r'[\s\-]+', str(part)):
            if tok:
                tokens.add(tok)
    return sorted(tokens)


def derive_full_name_compact(first_clean, middle_clean, last_clean):
    """Concatenate First→Middle→Last (skipping nulls), strip non-letters."""
    parts = [str(p) for p in (first_clean, middle_clean, last_clean) if pd.notna(p)]
    if not parts:
        return np.nan
    compact = re.sub(r'[^A-Za-z]', '', ''.join(parts))
    return compact.upper() if compact else np.nan


def derive_phones_set(*phones) -> list:
    """Sorted, deduplicated list of non-null cleaned phones."""
    return sorted({p for p in phones if pd.notna(p)})


def derive_address_normalized(line1, line2, city, state, zip_base, zip_ext):
    """First result of libpostal `expand_address` on the comma-joined full
    address, or NaN if libpostal is unavailable / returns empty."""
    if not _LIBPOSTAL_AVAILABLE:
        return np.nan

    parts = [str(p) for p in (line1, line2, city, state) if pd.notna(p)]
    if pd.notna(zip_base):
        zip_str = str(zip_base)
        if pd.notna(zip_ext):
            zip_str = f'{zip_str}-{zip_ext}'
        parts.append(zip_str)

    if not parts:
        return np.nan

    try:
        expansions = _expand_address(', '.join(parts))
    except Exception:
        return np.nan

    return expansions[0] if expansions else np.nan


# =============================================================================
# ORCHESTRATOR
# =============================================================================

RENAME_TO_RAW = (
    'FirstNM', 'LastNM', 'MiddleNM', 'SuffixNM',
    'BirthDT', 'SSN',
    'AddressLine1', 'AddressLine2', 'CityNM', 'ZipCD', 'StateCD',
    'PrimaryPhoneNBR', 'Phone01NBR', 'Phone02NBR', 'Phone03NBR',
    'Email', 'SexAtBirthDSC',
)


def _rename_to_raw(df: pd.DataFrame) -> pd.DataFrame:
    """Move each cleaned input column to `<col>_raw`, dropping any pre-existing
    `<col>_raw` so re-runs are idempotent."""
    for col in RENAME_TO_RAW:
        if col in df.columns:
            raw_col = f'{col}_raw'
            if raw_col in df.columns:
                df.drop(columns=[raw_col], inplace=True)
            df.rename(columns={col: raw_col}, inplace=True)
    return df


def transform_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Apply every rule from docs/Data-Cleaning-Guide.md and return a new
    DataFrame with `<col>_raw` + `<col>_clean` columns, `valid_record`, and the
    cross-field derivations."""
    out = df.copy()
    out = _rename_to_raw(out)

    valid = pd.Series(True, index=out.index)

    def _flag(series: pd.Series) -> None:
        nonlocal valid
        valid = valid & ~series.fillna(False).astype(bool)

    # Generational suffixes moved out of FirstNM/LastNM/MiddleNM, merged into
    # SuffixNM_clean below. The first non-null suffix in field order (First,
    # Last, Middle) wins; a record carries a generational suffix in at most one
    # of these fields in practice, so the order only matters for pathological rows.
    moved_suffix = pd.Series(np.nan, index=out.index, dtype=object)

    # ---- FirstNM ----
    if 'FirstNM_raw' in out.columns:
        res = out['FirstNM_raw'].apply(clean_first_name)
        out['FirstNM_clean'] = res.apply(lambda x: x[0])
        _flag(res.apply(lambda x: x[1]))
        first_suffix = res.apply(lambda x: x[2])
        moved_suffix = moved_suffix.where(moved_suffix.notna(), first_suffix)

    # ---- LastNM ----
    if 'LastNM_raw' in out.columns:
        res = out['LastNM_raw'].apply(clean_last_name)
        out['LastNM_clean'] = res.apply(lambda x: x[0])
        _flag(res.apply(lambda x: x[1]))
        last_suffix = res.apply(lambda x: x[2])
        moved_suffix = moved_suffix.where(moved_suffix.notna(), last_suffix)

    # ---- MiddleNM ----
    if 'MiddleNM_raw' in out.columns:
        res = out['MiddleNM_raw'].apply(clean_middle_name)
        out['MiddleNM_clean'] = res.apply(lambda x: x[0])
        _flag(res.apply(lambda x: x[1]))
        mid_suffix = res.apply(lambda x: x[2])
        moved_suffix = moved_suffix.where(moved_suffix.notna(), mid_suffix)

    # ---- SuffixNM ----
    if 'SuffixNM_raw' in out.columns:
        out['SuffixNM_clean'] = out['SuffixNM_raw'].apply(clean_suffix)
    else:
        out['SuffixNM_clean'] = pd.Series(np.nan, index=out.index, dtype=object)

    # Move generational suffixes pulled off LastNM/MiddleNM into SuffixNM_clean,
    # only filling rows where SuffixNM_clean is still null.
    out['SuffixNM_clean'] = out['SuffixNM_clean'].where(
        out['SuffixNM_clean'].notna(), moved_suffix)

    # ---- BirthDT ----
    if 'BirthDT_raw' in out.columns:
        out['BirthDT_clean'] = out['BirthDT_raw'].apply(clean_birth_date)

    # ---- SSN ----
    if 'SSN_raw' in out.columns:
        res = out['SSN_raw'].apply(clean_ssn)
        out['SSN_clean'] = res.apply(lambda x: x[0])
        out['last_4_SSN'] = res.apply(lambda x: x[1])

    # ---- AddressLine1 ----
    if 'AddressLine1_raw' in out.columns:
        res = out['AddressLine1_raw'].apply(clean_address_line1)
        out['AddressLine1_clean'] = res.apply(lambda x: x[0])
        _flag(res.apply(lambda x: x[1]))

    # ---- AddressLine2 ----
    if 'AddressLine2_raw' in out.columns:
        res = out['AddressLine2_raw'].apply(clean_address_line2)
        out['AddressLine2_clean'] = res.apply(lambda x: x[0])
        _flag(res.apply(lambda x: x[1]))

    # ---- CityNM ----
    if 'CityNM_raw' in out.columns:
        out['CityNM_clean'] = out['CityNM_raw'].apply(clean_city)

    # ---- ZipCD ----
    if 'ZipCD_raw' in out.columns:
        res = out['ZipCD_raw'].apply(clean_zip)
        out['ZipCD_clean_base'] = res.apply(lambda x: x[0])
        out['ZipCD_clean_ext'] = res.apply(lambda x: x[1])

    # ---- StateCD ----
    if 'StateCD_raw' in out.columns:
        out['StateCD_clean'] = out['StateCD_raw'].apply(clean_state)

    # ---- ZIP × State cross-check ----
    if 'ZipCD_clean_base' in out.columns and 'StateCD_clean' in out.columns:
        def _zip_state_mismatch(row):
            zb = row['ZipCD_clean_base']
            st = row['StateCD_clean']
            if pd.isna(zb) or pd.isna(st):
                return False
            valid_states = ZIP_PREFIX_TO_STATES.get(str(zb)[0], set())
            return st not in valid_states
        mask = out.apply(_zip_state_mismatch, axis=1)
        out.loc[mask, 'ZipCD_clean_base'] = np.nan
        out.loc[mask, 'ZipCD_clean_ext'] = np.nan

    # ---- Phone slots ----
    for slot in ('PrimaryPhoneNBR', 'Phone01NBR', 'Phone02NBR', 'Phone03NBR'):
        raw = f'{slot}_raw'
        if raw in out.columns:
            out[f'{slot}_clean'] = out[raw].apply(clean_phone)

    # ---- Email ----
    if 'Email_raw' in out.columns:
        out['Email_clean'] = out['Email_raw'].apply(clean_email)

    # ---- SexAtBirthDSC ----
    if 'SexAtBirthDSC_raw' in out.columns:
        out['SexAtBirthDSC_clean'] = out['SexAtBirthDSC_raw'].apply(clean_sex_at_birth)

    # ---- Global cross-field: both names null → invalid ----
    if 'FirstNM_clean' in out.columns and 'LastNM_clean' in out.columns:
        both_null = out['FirstNM_clean'].isna() & out['LastNM_clean'].isna()
        valid = valid & ~both_null

    # ---- Derived: full_name_tokens / full_name_compact ----
    if 'FirstNM_clean' in out.columns and 'LastNM_clean' in out.columns:
        middle = out['MiddleNM_clean'] if 'MiddleNM_clean' in out.columns else pd.Series(np.nan, index=out.index)
        out['full_name_tokens'] = [
            derive_full_name_tokens(first, mid, last)
            for first, mid, last in zip(out['FirstNM_clean'], middle, out['LastNM_clean'])
        ]
        out['full_name_compact'] = [
            derive_full_name_compact(first, mid, last)
            for first, mid, last in zip(out['FirstNM_clean'], middle, out['LastNM_clean'])
        ]

    # ---- Derived: Phones_set ----
    phone_clean_cols = [
        f'{s}_clean' for s in
        ('PrimaryPhoneNBR', 'Phone01NBR', 'Phone02NBR', 'Phone03NBR')
        if f'{s}_clean' in out.columns
    ]
    if phone_clean_cols:
        out['Phones_set'] = [
            derive_phones_set(*row)
            for row in zip(*(out[c] for c in phone_clean_cols))
        ]

    # ---- Derived: Address_normalized (libpostal) ----
    if 'AddressLine1_clean' in out.columns:
        line2 = out['AddressLine2_clean'] if 'AddressLine2_clean' in out.columns else pd.Series(np.nan, index=out.index)
        city = out['CityNM_clean'] if 'CityNM_clean' in out.columns else pd.Series(np.nan, index=out.index)
        state = out['StateCD_clean'] if 'StateCD_clean' in out.columns else pd.Series(np.nan, index=out.index)
        zip_b = out['ZipCD_clean_base'] if 'ZipCD_clean_base' in out.columns else pd.Series(np.nan, index=out.index)
        zip_e = out['ZipCD_clean_ext'] if 'ZipCD_clean_ext' in out.columns else pd.Series(np.nan, index=out.index)
        out['Address_normalized'] = [
            derive_address_normalized(a1, a2, c, s, zb, ze)
            for a1, a2, c, s, zb, ze in zip(out['AddressLine1_clean'], line2, city, state, zip_b, zip_e)
        ]

    out['valid_record'] = valid

    return out
