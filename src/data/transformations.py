"""
Data Transformations for MDM_Population Dataset
Implements all transformations from Data-Cleaning-Guide.md for the MPI pipeline.
"""

import re
import unicodedata
from datetime import datetime, date
from typing import Optional, List, Set, Tuple

import numpy as np
import pandas as pd


# =============================================================================
# CONSTANTS
# =============================================================================

NULL_VALUES = {'UNKNOWN', 'NULL', 'NAN', 'NONE', 'N/A', 'NA', ''}
MIDDLE_NAME_NULLS = {'NMI', 'UNKNOWN', 'NULL', 'N/A', '-', ''}

INVALID_NAME_PATTERNS = {
    'BABY', 'BABYBOY', 'BABYGIRL', 'TEST', 'DUPLICATE', 'DONOTUSE',
    'DO NOT USE', 'DONT USE', 'DO NOT USE DUPLICATE', 'DO NOT USE DOUBLE ACCOUNT',
    'DONOT USED', 'DO NOT USE DOBBLE ACCOUNT', 'DO NOT', 'DUPLICATE DO NOT USE',
    'DUPLICATE DONT USE', "DON'T USE"
}

SYSTEM_NAMES = {'TEST', 'DONOTUSE', 'MEDICARE', 'BLUE', 'CROSS', 'IDXDR', 'ACCOUNT'}

TITLES = {'MR', 'MRS', 'MS', 'DR', 'REV'}
GENERATIONAL_SUFFIXES = {'JR', 'SR', 'II', 'III', 'IV', 'V', 'I'}
ORDINAL_TO_ROMAN = {'2ND': 'II', '3RD': 'III', '4TH': 'IV', '5TH': 'V'}

VALID_SEX_VALUES = {'MALE', 'FEMALE', 'OTHER', 'M', 'F'}

US_STATE_CODES = {
    'AL', 'AK', 'AZ', 'AR', 'CA', 'CO', 'CT', 'DE', 'FL', 'GA', 'HI', 'ID', 'IL',
    'IN', 'IA', 'KS', 'KY', 'LA', 'ME', 'MD', 'MA', 'MI', 'MN', 'MS', 'MO', 'MT',
    'NE', 'NV', 'NH', 'NJ', 'NM', 'NY', 'NC', 'ND', 'OH', 'OK', 'OR', 'PA', 'RI',
    'SC', 'SD', 'TN', 'TX', 'UT', 'VT', 'VA', 'WA', 'WV', 'WI', 'WY', 'DC', 'PR',
    'VI', 'GU', 'AS', 'MP'
}

STATE_NAME_TO_CODE = {
    'ALABAMA': 'AL', 'ALASKA': 'AK', 'ARIZONA': 'AZ', 'ARKANSAS': 'AR',
    'CALIFORNIA': 'CA', 'COLORADO': 'CO', 'CONNECTICUT': 'CT', 'DELAWARE': 'DE',
    'FLORIDA': 'FL', 'GEORGIA': 'GA', 'HAWAII': 'HI', 'IDAHO': 'ID',
    'ILLINOIS': 'IL', 'INDIANA': 'IN', 'IOWA': 'IA', 'KANSAS': 'KS',
    'KENTUCKY': 'KY', 'LOUISIANA': 'LA', 'MAINE': 'ME', 'MARYLAND': 'MD',
    'MASSACHUSETTS': 'MA', 'MICHIGAN': 'MI', 'MINNESOTA': 'MN', 'MISSISSIPPI': 'MS',
    'MISSOURI': 'MO', 'MONTANA': 'MT', 'NEBRASKA': 'NE', 'NEVADA': 'NV',
    'NEW HAMPSHIRE': 'NH', 'NEW JERSEY': 'NJ', 'NEW MEXICO': 'NM', 'NEW YORK': 'NY',
    'NORTH CAROLINA': 'NC', 'NORTH DAKOTA': 'ND', 'OHIO': 'OH', 'OKLAHOMA': 'OK',
    'OREGON': 'OR', 'PENNSYLVANIA': 'PA', 'RHODE ISLAND': 'RI', 'SOUTH CAROLINA': 'SC',
    'SOUTH DAKOTA': 'SD', 'TENNESSEE': 'TN', 'TEXAS': 'TX', 'UTAH': 'UT',
    'VERMONT': 'VT', 'VIRGINIA': 'VA', 'WASHINGTON': 'WA', 'WEST VIRGINIA': 'WV',
    'WISCONSIN': 'WI', 'WYOMING': 'WY', 'DISTRICT OF COLUMBIA': 'DC',
    'PUERTO RICO': 'PR', 'VIRGIN ISLANDS': 'VI', 'GUAM': 'GU'
}

STREET_SUFFIX_MAP = {
    'STREET': 'ST', 'AVENUE': 'AVE', 'ROAD': 'RD', 'BOULEVARD': 'BLVD',
    'DRIVE': 'DR', 'LANE': 'LN', 'COURT': 'CT', 'CIRCLE': 'CIR',
    'PLACE': 'PL', 'TERRACE': 'TER', 'HIGHWAY': 'HWY', 'PARKWAY': 'PKWY',
    'WAY': 'WAY', 'TRAIL': 'TRL', 'EXPRESSWAY': 'EXPY'
}

DIRECTION_MAP = {
    'NORTH': 'N', 'SOUTH': 'S', 'EAST': 'E', 'WEST': 'W',
    'NORTHEAST': 'NE', 'NORTHWEST': 'NW', 'SOUTHEAST': 'SE', 'SOUTHWEST': 'SW'
}

UNIT_DESIGNATOR_MAP = {
    'SUITE': 'STE', 'APARTMENT': 'APT', 'ROOM': 'RM', 'UNIT': 'UNIT',
    'BUILDING': 'BLDG', 'FLOOR': 'FL', 'DEPARTMENT': 'DEPT'
}

PLACEHOLDER_ADDRESSES = {
    'HOMELESS', 'TRANSIENT', 'BAD ADDRESS', 'DO NOT USE', 'NO ADDRESS', '?', 'NO MAIL',
    'GENERAL DELIVERY', 'NKA', '.', 'NOT TAKEN', 'ZOCDOC', 'GET', 'HOPE CENTER', 'INCORRECT ADDRESS'
}
PLACEHOLDER_ZIPS = {'00000', '99999', '11111', '12345'}
PLACEHOLDER_EMAILS = {
    'test@', 'noemail@', 'none@none.com', 'noemail@noemail.com',
    'none', 'declined', 'decline', 'no email', 'na', 'n', 'aca@eriefamilyhealth.org',
    'noemail@textmessage.com', 'noemail@textmessaging.com', 'none listed', 'blank',
    'none provided', 'declined portal', 'aca@eriefamilyheath.org', 'no e-mail', 'refused',
    'aca@eriefamilyhealth.com', 'no', 'denied', 'no email address', 'no email cm',
    'no email address daj', 'noemail@textmessages.com', 'decline@hamakua-health.org',
    'noemai@noemail.com', 'patientdeclined@howardbrown.org', 'ptdeclined@hb.org',
    'noemail@noemail', 'not age appropriate', 'unknown', 'pt declined', 'none@gmail.com',
    'declined@hamakua-health.org'
}

JUNK_EMAIL_EXACT = {
    'noemail@noemail.com', 'noemail@email.com', 'no@email.com', 'none@none.com',
    'unknown@unknown.com', 'unknown@email.com', 'test@test.com', 'test@example.com',
    'donotreply@donotreply.com', 'noreply@noreply.com', 'email@email.com',
    'patient@patient.com', 'none@noemail.com', 'na@na.com', 'null@null.com'
}

JUNK_EMAIL_LOCAL_PREFIXES = {
    'noemail', 'no-email', 'no_email', 'noreply', 'no-reply', 'donotreply',
    'do-not-reply', 'unknown', 'test', 'patient', 'none', 'null', 'na'
}

JUNK_EMAIL_DOMAINS = {
    'example.com', 'example.org', 'example.net', 'test.com', 'test.org',
    'noemail.com', 'noreply.com', 'donotreply.com', 'unknown.com', '123.com'
}

JUNK_SSN_EXACT = {
    '010101010', '090909090', '000000001', '999999998',
    '111223333', '219099999', '457555462'
}

JUNK_SSN_SEQUENTIAL = {'123456789', '987654321', '123123123'}

COMPOUND_LAST_NAME_PREFIXES = {
    'DE LA', 'DE LOS', 'DEL', 'VAN', 'VON', 'MC', 'MAC', "O'", 'ST',
    'LA', 'LE', 'DI', 'DA', 'DOS', 'DAS', 'DE'
}


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def normalize_unicode(text: str) -> str:
    """Remove diacritical marks using Unicode normalization."""
    if pd.isna(text) or not isinstance(text, str):
        return text
    normalized = unicodedata.normalize('NFD', text)
    return ''.join(c for c in normalized if unicodedata.category(c) != 'Mn')


def collapse_whitespace(text: str) -> str:
    """Collapse multiple spaces into single space and trim."""
    if pd.isna(text) or not isinstance(text, str):
        return text
    return re.sub(r'\s+', ' ', text).strip()


def standardize_null(value: str, null_set: Set[str] = None) -> Optional[str]:
    """Convert text-based null values to NaN."""
    if null_set is None:
        null_set = NULL_VALUES
    if pd.isna(value):
        return np.nan
    if not isinstance(value, str):
        return value
    if value.strip().upper() in null_set or value.strip() == '':
        return np.nan
    return value


def remove_numeric_chars(text: str) -> str:
    """Remove all numeric characters from text."""
    if pd.isna(text) or not isinstance(text, str):
        return text
    return re.sub(r'\d', '', text)


def clean_punctuation_preserve_hyphens_apostrophes(text: str) -> str:
    """Remove punctuation except hyphens and apostrophes."""
    if pd.isna(text) or not isinstance(text, str):
        return text
    return re.sub(r"[^\w\s\-']", '', text)


def remove_ehr_artifacts(text: str) -> str:
    """Remove EHR merge tags like <mrg>."""
    if pd.isna(text) or not isinstance(text, str):
        return text
    return re.sub(r'<[^>]+>', '', text)


# =============================================================================
# NAME TRANSFORMATIONS
# =============================================================================

def clean_first_name(value: str, valid_record_flag: bool = True) -> Tuple[str, bool]:
    """
    Clean first name field.
    Returns (cleaned_value, valid_record_flag)
    """
    if pd.isna(value):
        return np.nan, valid_record_flag
    
    text = str(value).upper().strip()
    text = collapse_whitespace(text)
    
    # Check for invalid patterns - flag record as invalid
    if text in INVALID_NAME_PATTERNS or text.replace(' ', '') in INVALID_NAME_PATTERNS:
        valid_record_flag = False
    
    # Standardize nulls
    if text in NULL_VALUES or text == '':
        return np.nan, valid_record_flag
    
    # Remove titles from beginning
    for title in TITLES:
        pattern = rf'^{title}\.?\s+'
        text = re.sub(pattern, '', text, flags=re.IGNORECASE)
    
    # Remove EHR artifacts
    text = remove_ehr_artifacts(text)
    
    # Unicode normalization
    text = normalize_unicode(text)
    
    # Remove numeric characters
    text = remove_numeric_chars(text)
    
    # Clean punctuation
    text = clean_punctuation_preserve_hyphens_apostrophes(text)
    
    # Final cleanup
    text = collapse_whitespace(text)
    
    if text == '':
        return np.nan, valid_record_flag
    
    return text, valid_record_flag


def clean_last_name(value: str, valid_record_flag: bool = True) -> Tuple[str, bool]:
    """
    Clean last name field.
    Returns (cleaned_value, valid_record_flag)
    """
    if pd.isna(value):
        return np.nan, valid_record_flag
    
    text = str(value).upper().strip()
    text = collapse_whitespace(text)
    
    # Check for invalid patterns
    if text in INVALID_NAME_PATTERNS or text.replace(' ', '') in INVALID_NAME_PATTERNS:
        valid_record_flag = False
    
    # Standardize nulls
    if text in NULL_VALUES or text == '':
        return np.nan, valid_record_flag
    
    # Remove generational suffixes from end
    for suffix in GENERATIONAL_SUFFIXES:
        pattern = rf'\s+{suffix}\.?$'
        text = re.sub(pattern, '', text, flags=re.IGNORECASE)
    
    # Remove EHR artifacts
    text = remove_ehr_artifacts(text)
    
    # Unicode normalization
    text = normalize_unicode(text)
    
    # Remove numeric characters
    text = remove_numeric_chars(text)
    
    # Clean punctuation
    text = clean_punctuation_preserve_hyphens_apostrophes(text)
    
    # Final cleanup
    text = collapse_whitespace(text)
    
    if text == '':
        return np.nan, valid_record_flag
    
    return text, valid_record_flag


def clean_middle_name(value: str, valid_record_flag: bool = True) -> Tuple[str, bool]:
    """
    Clean middle name field.
    Returns (cleaned_value, valid_record_flag)
    """
    if pd.isna(value):
        return np.nan, valid_record_flag
    
    text = str(value).upper().strip()
    text = collapse_whitespace(text)
    
    # Check for invalid patterns
    if text in INVALID_NAME_PATTERNS or text.replace(' ', '') in INVALID_NAME_PATTERNS:
        valid_record_flag = False
    
    # Standardize nulls (middle name has additional null patterns)
    if text in MIDDLE_NAME_NULLS or text == '':
        return np.nan, valid_record_flag
    
    # Unicode normalization
    text = normalize_unicode(text)
    
    # Clean punctuation
    text = clean_punctuation_preserve_hyphens_apostrophes(text)
    
    # Remove numeric characters
    text = remove_numeric_chars(text)
    
    # Final cleanup
    text = collapse_whitespace(text)
    
    if text == '':
        return np.nan, valid_record_flag
    
    return text, valid_record_flag


def clean_suffix(value: str) -> str:
    """Clean name suffix field."""
    if pd.isna(value):
        return np.nan
    
    text = str(value).upper().strip()
    text = collapse_whitespace(text)
    
    # Standardize nulls
    if text in NULL_VALUES or text == '':
        return np.nan
    
    # Remove punctuation
    text = re.sub(r'[^\w\s]', '', text)
    
    # Normalize ordinal suffixes
    for ordinal, roman in ORDINAL_TO_ROMAN.items():
        if text == ordinal:
            text = roman
            break
    
    # Only retain valid suffixes
    if text not in GENERATIONAL_SUFFIXES:
        return np.nan
    
    return text


# =============================================================================
# DATE OF BIRTH TRANSFORMATIONS
# =============================================================================

def clean_birth_date(value, reference_date: date = None) -> Tuple[Optional[date], dict]:
    """
    Clean date of birth field.
    Returns (cleaned_date, indicators_dict)
    """
    indicators = {
        'INVALID_DOB': False,
        'DEFAULT_DOB_PATTERN': False,
        'FUTURE_DOB': False
    }
    
    if reference_date is None:
        reference_date = date.today()
    
    if pd.isna(value):
        return pd.NaT, indicators
    
    parsed_date = None
    
    # Try parsing various formats
    if isinstance(value, (datetime, date)):
        parsed_date = value if isinstance(value, date) else value.date()
    else:
        text = str(value).strip()
        formats = ['%Y-%m-%d', '%m/%d/%Y', '%d/%m/%Y', '%Y%m%d', '%m-%d-%Y']
        for fmt in formats:
            try:
                parsed_date = datetime.strptime(text[:10], fmt).date()
                break
            except (ValueError, IndexError):
                continue
    
    if parsed_date is None:
        indicators['INVALID_DOB'] = True
        return pd.NaT, indicators
    
    # Check for future dates
    if parsed_date > reference_date:
        indicators['FUTURE_DOB'] = True
        return pd.NaT, indicators
    
    # Check for implausible dates (age > 115)
    age = (reference_date - parsed_date).days / 365.25
    if age > 115:
        indicators['INVALID_DOB'] = True
        return pd.NaT, indicators
    
    # Check for known default patterns
    if parsed_date == date(1900, 1, 1):
        indicators['DEFAULT_DOB_PATTERN'] = True
        return pd.NaT, indicators
    
    # Flag January 1st patterns (potential defaults)
    if parsed_date.month == 1 and parsed_date.day == 1:
        indicators['DEFAULT_DOB_PATTERN'] = True
    
    return parsed_date, indicators


# =============================================================================
# SSN TRANSFORMATIONS
# =============================================================================

def clean_ssn(value: str) -> Tuple[Optional[str], dict]:
    """
    Clean SSN field with enhanced validation per Data-Cleaning-Guide.
    Returns (cleaned_ssn, indicators_dict)
    """
    indicators = {
        'is_missing_SSN': False,
        'is_junk_SSN': False,
        'is_invalid_SSN': False,
        'PADDED_SSN': False
    }
    
    if pd.isna(value) or str(value).strip() == '':
        indicators['is_missing_SSN'] = True
        return np.nan, indicators
    
    # Strip all formatting characters (hyphens, spaces, dots, parentheses)
    ssn = re.sub(r'[\-\s\.\(\)]', '', str(value))
    # Remove any remaining non-numeric characters
    ssn = re.sub(r'\D', '', ssn)
    
    # Left-pad short SSNs (7 or 8 digits)
    if len(ssn) in (7, 8):
        ssn = ssn.zfill(9)
        indicators['PADDED_SSN'] = True
    
    # Length validation: must be exactly 9 digits
    if len(ssn) != 9:
        indicators['is_invalid_SSN'] = True
        return np.nan, indicators
    
    # Check for repeating single digit patterns (all same digit)
    if len(set(ssn)) == 1:
        indicators['is_junk_SSN'] = True
        return np.nan, indicators
    
    # Check for known sequential patterns
    if ssn in JUNK_SSN_SEQUENTIAL:
        indicators['is_junk_SSN'] = True
        return np.nan, indicators
    
    # Check for known exact junk values
    if ssn in JUNK_SSN_EXACT:
        indicators['is_junk_SSN'] = True
        return np.nan, indicators
    
    # Validate area number (first 3 digits)
    area_number = ssn[:3]
    # 000 - never assigned by SSA
    if area_number == '000':
        indicators['is_invalid_SSN'] = True
        return np.nan, indicators
    # 666 - never assigned by SSA
    if area_number == '666':
        indicators['is_invalid_SSN'] = True
        return np.nan, indicators
    # 9XX range - reserved for ITINs, not valid SSNs
    if area_number.startswith('9'):
        indicators['is_invalid_SSN'] = True
        return np.nan, indicators
    
    # Validate group number (middle 2 digits, positions 4-5)
    group_number = ssn[3:5]
    if group_number == '00':
        indicators['is_invalid_SSN'] = True
        return np.nan, indicators
    
    # Validate serial number (last 4 digits, positions 6-9)
    serial_number = ssn[5:9]
    if serial_number == '0000':
        indicators['is_invalid_SSN'] = True
        return np.nan, indicators
    
    return ssn, indicators


def extract_last_4_ssn(ssn: str) -> Optional[str]:
    """Extract last 4 digits of SSN."""
    if pd.isna(ssn) or len(str(ssn)) < 4:
        return np.nan
    return str(ssn)[-4:]


# =============================================================================
# ADDRESS TRANSFORMATIONS
# =============================================================================

def clean_address(value: str, valid_record_flag: bool = True) -> Tuple[Optional[str], bool, dict]:
    """
    Clean address field.
    Returns (cleaned_address, valid_record_flag, indicators_dict)
    """
    indicators = {
        'IS_POBOX': False,
        'HAS_TRAPPED_UNIT': False,
        'INVALID_ADDRESS': False,
        'HOMELESS_ADDRESS': False
    }
    
    if pd.isna(value):
        return np.nan, valid_record_flag, indicators
    
    text = str(value).upper().strip()
    text = collapse_whitespace(text)
    
    # Standardize nulls
    if text in NULL_VALUES or text == '':
        return np.nan, valid_record_flag, indicators
    
    # Check for placeholder addresses
    if text in PLACEHOLDER_ADDRESSES:
        indicators['INVALID_ADDRESS'] = True
        if text in {'HOMELESS', 'TRANSIENT'}:
            indicators['HOMELESS_ADDRESS'] = True
        return np.nan, valid_record_flag, indicators
    
    # Check for invalid patterns - mark record as invalid
    if text in INVALID_NAME_PATTERNS or text.replace(' ', '') in INVALID_NAME_PATTERNS:
        valid_record_flag = False
    
    # Detect PO Box
    po_box_pattern = r'P\.?\s*O\.?\s*BOX|POST\s+OFFICE\s+BOX'
    if re.search(po_box_pattern, text):
        indicators['IS_POBOX'] = True
        text = re.sub(po_box_pattern, 'PO BOX', text)
    
    # Remove punctuation (periods, commas, #)
    text = re.sub(r'[.,#]', '', text)
    
    # Standardize street suffixes
    for full, abbrev in STREET_SUFFIX_MAP.items():
        pattern = rf'\b{full}\b'
        text = re.sub(pattern, abbrev, text)
    
    # Standardize directions
    for full, abbrev in DIRECTION_MAP.items():
        pattern = rf'\b{full}\b'
        text = re.sub(pattern, abbrev, text)
    
    # Standardize unit designators
    for full, abbrev in UNIT_DESIGNATOR_MAP.items():
        pattern = rf'\b{full}\b'
        text = re.sub(pattern, abbrev, text)
        if re.search(rf'\b{abbrev}\s+\d', text):
            indicators['HAS_TRAPPED_UNIT'] = True
    
    # Remove leading zeros from street numbers
    text = re.sub(r'^0+(\d+)', r'\1', text)
    
    # Final cleanup
    text = collapse_whitespace(text)
    
    if text == '':
        return np.nan, valid_record_flag, indicators
    
    return text, valid_record_flag, indicators


# =============================================================================
# CITY TRANSFORMATIONS
# =============================================================================

CITY_CORRECTIONS = {
    'CHCAGO': 'CHICAGO',
    'CHIGAGO': 'CHICAGO',
    'CHICAO': 'CHICAGO',
}

def clean_city(value: str) -> Tuple[Optional[str], dict]:
    """
    Clean city name field.
    Returns (cleaned_city, indicators_dict)
    """
    indicators = {
        'HAS_NUMERIC_CITY': False,
        'SUSPICIOUS_CITY_LENGTH': False
    }
    
    if pd.isna(value):
        return np.nan, indicators
    
    text = str(value).upper().strip()
    text = collapse_whitespace(text)
    
    # Standardize nulls
    if text in NULL_VALUES or text == '':
        return np.nan, indicators
    
    # Unicode normalization
    text = normalize_unicode(text)
    
    # Check for numeric characters before removing
    if re.search(r'\d', text):
        indicators['HAS_NUMERIC_CITY'] = True
    
    # Remove numeric characters
    text = remove_numeric_chars(text)
    
    # Clean punctuation (preserve spaces, hyphens, apostrophes)
    text = re.sub(r"[^\w\s\-']", '', text)
    
    # Fix common misspellings
    if text in CITY_CORRECTIONS:
        text = CITY_CORRECTIONS[text]
    
    # Final cleanup
    text = collapse_whitespace(text)
    
    # Check for suspicious length
    if len(text) < 2:
        indicators['SUSPICIOUS_CITY_LENGTH'] = True
    
    if text == '':
        return np.nan, indicators
    
    return text, indicators


# =============================================================================
# ZIP CODE TRANSFORMATIONS
# =============================================================================

def clean_zip(value: str) -> Tuple[Optional[str], Optional[str], dict]:
    """
    Clean ZIP code field.
    Returns (zip_base, zip_ext, indicators_dict)
    """
    indicators = {
        'INVALID_ZIP': False,
        'JUNK_ZIP': False
    }
    
    if pd.isna(value):
        return np.nan, np.nan, indicators
    
    # Remove non-numeric characters except hyphen
    text = str(value).strip()
    
    # Split on hyphen for ZIP+4
    parts = text.split('-')
    zip_base = re.sub(r'\D', '', parts[0])
    zip_ext = re.sub(r'\D', '', parts[1]) if len(parts) > 1 else np.nan
    
    # Left-pad if needed
    if len(zip_base) < 5:
        zip_base = zip_base.zfill(5)
    
    # Truncate to 5 digits
    zip_base = zip_base[:5]
    
    # Must be exactly 5 digits
    if len(zip_base) != 5:
        indicators['INVALID_ZIP'] = True
        return np.nan, np.nan, indicators
    
    # Check for placeholder ZIPs
    if zip_base in PLACEHOLDER_ZIPS:
        indicators['JUNK_ZIP'] = True
        return np.nan, np.nan, indicators
    
    return zip_base, zip_ext if pd.notna(zip_ext) and zip_ext else np.nan, indicators


# =============================================================================
# STATE TRANSFORMATIONS
# =============================================================================

def clean_state(value: str) -> Tuple[Optional[str], dict]:
    """
    Clean state code field.
    Returns (cleaned_state, indicators_dict)
    """
    indicators = {
        'INVALID_STATE': False,
        'NUMERIC_STATE_ARTIFACT': False
    }
    
    if pd.isna(value):
        return np.nan, indicators
    
    text = str(value).upper().strip()
    
    # Standardize nulls
    if text in NULL_VALUES or text == '':
        return np.nan, indicators
    
    # Check for numeric artifacts
    if text.isdigit():
        indicators['NUMERIC_STATE_ARTIFACT'] = True
        return np.nan, indicators
    
    # Map full names to codes
    if text in STATE_NAME_TO_CODE:
        text = STATE_NAME_TO_CODE[text]
    
    # Validate against US state codes
    if text not in US_STATE_CODES:
        indicators['INVALID_STATE'] = True
        return np.nan, indicators
    
    return text, indicators


# =============================================================================
# COUNTRY TRANSFORMATIONS
# =============================================================================

def clean_country(value: str) -> Tuple[Optional[str], dict]:
    """
    Clean country name field.
    Returns (cleaned_country, indicators_dict)
    """
    indicators = {
        'IS_FOREIGN_COUNTRY': False,
        'MAPPED_NUMERIC_COUNTRY': False
    }
    
    if pd.isna(value):
        return np.nan, indicators
    
    text = str(value).upper().strip()
    
    # Standardize nulls
    if text in NULL_VALUES or text == '':
        return np.nan, indicators
    
    # Map numeric artifact
    if text == '1':
        indicators['MAPPED_NUMERIC_COUNTRY'] = True
        return 'US', indicators
    
    # Normalize US variants
    us_variants = {'USA', 'UNITED STATES', 'U.S.A.', 'US', 'UNITED STATES OF AMERICA'}
    if text in us_variants or re.sub(r'[^\w]', '', text) in {'USA', 'US'}:
        return 'US', indicators
    
    # Non-US country
    indicators['IS_FOREIGN_COUNTRY'] = True
    return text, indicators


# =============================================================================
# PHONE TRANSFORMATIONS
# =============================================================================

def clean_phone(value: str) -> Tuple[Optional[str], dict]:
    """
    Clean phone number field.
    Returns (cleaned_phone, indicators_dict)
    """
    indicators = {
        'INVALID_PHONE': False,
        'JUNK_PHONE': False,
        'NORMALIZED_COUNTRY_CODE': False
    }
    
    if pd.isna(value):
        return np.nan, indicators
    
    # Remove non-numeric characters
    phone = re.sub(r'\D', '', str(value))
    
    # Normalize 11-digit numbers starting with 1
    if len(phone) == 11 and phone.startswith('1'):
        phone = phone[1:]
        indicators['NORMALIZED_COUNTRY_CODE'] = True
    
    # Must be exactly 10 digits
    if len(phone) != 10:
        indicators['INVALID_PHONE'] = True
        return np.nan, indicators
    
    # Check for junk patterns (repeating/sequential)
    if len(set(phone)) == 1:
        indicators['JUNK_PHONE'] = True
        return np.nan, indicators
    
    if phone == '1234567890':
        indicators['JUNK_PHONE'] = True
        return np.nan, indicators
    
    # Check for invalid area codes
    area_code = phone[:3]
    invalid_area_codes = {'000', '111', '999'}
    if area_code in invalid_area_codes:
        indicators['INVALID_PHONE'] = True
        return np.nan, indicators
    
    return phone, indicators


# =============================================================================
# EMAIL TRANSFORMATIONS
# =============================================================================

def clean_email(value: str) -> Tuple[Optional[str], dict]:
    """
    Clean email field with enhanced pattern-based junk detection per Data-Cleaning-Guide.
    Returns (cleaned_email, indicators_dict)
    """
    indicators = {
        'is_missing_Email': False,
        'is_junk_Email': False,
        'is_invalid_Email': False
    }
    
    if pd.isna(value) or str(value).strip() == '':
        indicators['is_missing_Email'] = True
        return np.nan, indicators
    
    text = str(value).lower().strip()
    
    # Nullify primitive placeholders
    if text in {'nan', 'none', 'null', 'na', ''}:
        indicators['is_junk_Email'] = True
        return np.nan, indicators
    
    # Format validation - presence of @
    if '@' not in text:
        indicators['is_invalid_Email'] = True
        return np.nan, indicators
    
    # Format validation - domain structure: [chars]@[chars].[chars]
    email_pattern = r'^.+@.+\..+$'
    if not re.match(email_pattern, text):
        indicators['is_invalid_Email'] = True
        return np.nan, indicators
    
    # Check for exact known junk email values
    if text in JUNK_EMAIL_EXACT:
        indicators['is_junk_Email'] = True
        return np.nan, indicators
    
    # Split into local part and domain
    local_part, domain = text.rsplit('@', 1)
    
    # Pattern-based junk detection - local part
    for prefix in JUNK_EMAIL_LOCAL_PREFIXES:
        if local_part.startswith(prefix):
            indicators['is_junk_Email'] = True
            return np.nan, indicators
    
    # Local part is 1-2 characters (invalid personal emails)
    if len(local_part) <= 2:
        indicators['is_junk_Email'] = True
        return np.nan, indicators
    
    # Contains sequential digit pattern
    if '123456' in local_part:
        indicators['is_junk_Email'] = True
        return np.nan, indicators
    
    # Pattern-based junk detection - domain
    if domain in JUNK_EMAIL_DOMAINS:
        indicators['is_junk_Email'] = True
        return np.nan, indicators
    
    # Full email validation pattern
    full_email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    if not re.match(full_email_pattern, text):
        indicators['is_invalid_Email'] = True
        return np.nan, indicators
    
    return text, indicators


# =============================================================================
# SEX AT BIRTH TRANSFORMATIONS
# =============================================================================

def clean_sex_at_birth(value: str) -> Tuple[Optional[str], dict]:
    """
    Clean sex at birth field.
    Returns (cleaned_sex, indicators_dict)
    """
    indicators = {
        'UNKNOWN_SEX': False,
        'INVALID_SEX_VALUE': False
    }
    
    if pd.isna(value):
        return np.nan, indicators
    
    text = str(value).upper().strip()
    
    # Standardize nulls - include OTHER as nullifiable per guide
    if text in {'UNKNOWN', 'NULL', 'N/A', 'NA', 'OTHER', ''}:
        indicators['UNKNOWN_SEX'] = True
        return np.nan, indicators
    
    # Normalize M/F to full form
    sex_map = {'M': 'MALE', 'F': 'FEMALE'}
    if text in sex_map:
        text = sex_map[text]
    
    # Validate - only MALE and FEMALE are valid
    valid_values = {'MALE', 'FEMALE'}
    if text not in valid_values:
        indicators['INVALID_SEX_VALUE'] = True
        return np.nan, indicators
    
    return text, indicators


# =============================================================================
# CROSS-COLUMN NAME PROCESSING
# =============================================================================

def detect_camelcase(text: str) -> List[str]:
    """
    Detect and split CamelCase patterns.
    Returns list of words if CamelCase detected, otherwise empty list.
    """
    if pd.isna(text) or not isinstance(text, str):
        return []
    
    # Check if text contains CamelCase pattern (lowercase followed by uppercase)
    if not re.search(r'[a-z][A-Z]', text):
        return []
    
    # Split on uppercase boundaries
    words = re.findall(r'[A-Z]?[a-z]+|[A-Z]+(?=[A-Z][a-z]|\d|\W|$)', text)
    return [w.upper() for w in words] if len(words) > 1 else []


def extract_compound_prefix(tokens: List[str]) -> Tuple[List[str], str]:
    """
    Extract compound last name prefix from token list.
    Returns (remaining_tokens, compound_last_name)
    """
    if len(tokens) < 2:
        return tokens, ''
    
    # Check for compound prefixes (2-word prefixes first, then 1-word)
    for prefix_len in [2, 1]:
        if len(tokens) > prefix_len:
            potential_prefix = ' '.join(tokens[-prefix_len-1:-1]).upper()
            if potential_prefix in COMPOUND_LAST_NAME_PREFIXES:
                compound_last = ' '.join(tokens[-prefix_len-1:])
                return tokens[:-prefix_len-1], compound_last
    
    # Check single-word prefixes like MC, MAC, O'
    for i, token in enumerate(tokens[:-1]):
        if token.upper() in {'MC', 'MAC', "O'", 'DE', 'LA', 'LE', 'DI', 'DA', 'VAN', 'VON', 'ST'}:
            if i < len(tokens) - 1:
                compound_last = ' '.join(tokens[i:])
                return tokens[:i], compound_last
    
    return tokens, ''


def process_name_redistribution(
    first_nm: str, 
    middle_nm: str, 
    last_nm: str, 
    suffix_nm: str = None
) -> Tuple[str, str, str, str, dict]:
    """
    Full name redistribution and cross-column name processing per Data-Cleaning-Guide.
    Returns (first_clean, middle_clean, last_clean, suffix_clean, indicators_dict)
    """
    indicators = {
        'INFERRED_LAST_FROM_FIRST': False,
        'MOVED_MIDDLE_FROM_FIRST': False,
        'CAMELCASE_SPLIT': False,
        'EXTRACTED_SUFFIX': False,
        'needs_name_review': False,
        'REMOVED_DUPLICATE_LAST_TOKEN': False
    }
    
    first_clean = first_nm if pd.notna(first_nm) else ''
    middle_clean = middle_nm if pd.notna(middle_nm) else ''
    last_clean = last_nm if pd.notna(last_nm) else ''
    suffix_clean = suffix_nm if pd.notna(suffix_nm) else ''
    
    # CamelCase detection (before other processing)
    if first_clean and not ' ' in first_clean:
        camel_words = detect_camelcase(first_clean)
        if camel_words:
            indicators['CAMELCASE_SPLIT'] = True
            indicators['needs_name_review'] = True
            if not last_clean and len(camel_words) >= 2:
                first_clean = camel_words[0]
                if len(camel_words) > 2 and not middle_clean:
                    middle_clean = ' '.join(camel_words[1:-1])
                last_clean = camel_words[-1]
                indicators['INFERRED_LAST_FROM_FIRST'] = True
            elif last_clean and len(camel_words) >= 2 and not middle_clean:
                first_clean = camel_words[0]
                middle_clean = ' '.join(camel_words[1:])
                indicators['MOVED_MIDDLE_FROM_FIRST'] = True
    
    # Flag names > 20 chars with no spaces
    if first_clean and len(first_clean) > 20 and ' ' not in first_clean:
        indicators['needs_name_review'] = True
    
    # Tokenize first name
    first_tokens = first_clean.split() if first_clean else []
    
    # Extract suffix if present in tokens
    if first_tokens and first_tokens[-1].upper() in GENERATIONAL_SUFFIXES:
        if not suffix_clean:
            suffix_clean = first_tokens[-1].upper()
            first_tokens = first_tokens[:-1]
            indicators['EXTRACTED_SUFFIX'] = True
    
    # Handle compound last name prefixes
    remaining_tokens, compound_last = extract_compound_prefix(first_tokens)
    if compound_last and not last_clean:
        first_tokens = remaining_tokens
        last_clean = compound_last
        indicators['INFERRED_LAST_FROM_FIRST'] = True
    
    # Case 1: FirstNM has 3+ words AND LastNM is empty
    if not last_clean and len(first_tokens) >= 3:
        last_clean = first_tokens[-1]
        if not middle_clean:
            middle_clean = ' '.join(first_tokens[1:-1])
        first_clean = first_tokens[0]
        indicators['INFERRED_LAST_FROM_FIRST'] = True
        indicators['needs_name_review'] = True
    
    # Case 2: FirstNM has 2 words AND LastNM is empty
    elif not last_clean and len(first_tokens) == 2:
        first_clean = first_tokens[0]
        last_clean = first_tokens[1]
        indicators['INFERRED_LAST_FROM_FIRST'] = True
        indicators['needs_name_review'] = True
    
    # Case 3: FirstNM has 2+ words AND LastNM exists AND MiddleNM is empty
    elif last_clean and len(first_tokens) >= 2 and not middle_clean:
        first_clean = first_tokens[0]
        middle_clean = ' '.join(first_tokens[1:])
        indicators['MOVED_MIDDLE_FROM_FIRST'] = True
        indicators['needs_name_review'] = True
    
    # Case 4: FirstNM has 2+ words AND LastNM exists AND MiddleNM exists
    # Leave as-is (assume compound first name)
    elif last_clean and len(first_tokens) >= 2 and middle_clean:
        first_clean = ' '.join(first_tokens)
    
    # Check for duplicate last name in first name tokens
    elif last_clean and first_tokens:
        if first_tokens[-1].upper() == last_clean.upper():
            first_tokens = first_tokens[:-1]
            if len(first_tokens) > 1 and not middle_clean:
                middle_clean = ' '.join(first_tokens[1:])
                first_clean = first_tokens[0]
            elif first_tokens:
                first_clean = first_tokens[0]
            else:
                first_clean = ''
            indicators['REMOVED_DUPLICATE_LAST_TOKEN'] = True
    
    # Convert empty strings to NaN
    first_clean = first_clean.strip() if first_clean else np.nan
    middle_clean = middle_clean.strip() if middle_clean else np.nan
    last_clean = last_clean.strip() if last_clean else np.nan
    suffix_clean = suffix_clean.strip() if suffix_clean else np.nan
    
    # Final empty check
    if pd.notna(first_clean) and first_clean == '':
        first_clean = np.nan
    if pd.notna(middle_clean) and middle_clean == '':
        middle_clean = np.nan
    if pd.notna(last_clean) and last_clean == '':
        last_clean = np.nan
    if pd.notna(suffix_clean) and suffix_clean == '':
        suffix_clean = np.nan
    
    return first_clean, middle_clean, last_clean, suffix_clean, indicators


def process_name_tokens(first_nm: str, middle_nm: str, last_nm: str) -> Tuple[str, str, str, dict]:
    """
    Legacy wrapper for cross-column name separation.
    Returns (first_clean, middle_clean, last_clean, indicators_dict)
    """
    first_clean, middle_clean, last_clean, _, indicators = process_name_redistribution(
        first_nm, middle_nm, last_nm, None
    )
    return first_clean, middle_clean, last_clean, indicators


def validate_name_presence(first_nm: str, last_nm: str) -> Tuple[bool, dict]:
    """
    Validate that both first and last name are present.
    Returns (is_valid, indicators_dict)
    """
    indicators = {
        'MISSING_FIRST_NAME': False,
        'MISSING_LAST_NAME': False,
        'MISSING_BOTH_NAMES': False
    }
    
    has_first = pd.notna(first_nm) and str(first_nm).strip() != ''
    has_last = pd.notna(last_nm) and str(last_nm).strip() != ''
    
    if not has_first and not has_last:
        indicators['MISSING_BOTH_NAMES'] = True
        return False, indicators
    
    if not has_first:
        indicators['MISSING_FIRST_NAME'] = True
        return False, indicators
    
    if not has_last:
        indicators['MISSING_LAST_NAME'] = True
        return False, indicators
    
    return True, indicators


# =============================================================================
# DATA QUALITY FLAGS
# =============================================================================

def generate_quality_flags(row: pd.Series) -> dict:
    """
    Generate record-level data quality flags.
    """
    flags = {}
    
    # Check for valid name
    has_first = pd.notna(row.get('FirstNM_clean'))
    has_last = pd.notna(row.get('LastNM_clean'))
    flags['HAS_VALID_NAME'] = has_first and has_last
    
    # Check for valid DOB
    flags['HAS_VALID_DOB'] = pd.notna(row.get('BirthDT_clean'))
    
    # Check for valid SSN
    flags['HAS_VALID_SSN'] = pd.notna(row.get('SSN_clean'))
    
    # Check for contact info
    has_phone = any(pd.notna(row.get(f'Phone{i}_clean')) for i in ['Primary', '01', '02', '03'])
    has_email = pd.notna(row.get('Email_clean'))
    flags['HAS_CONTACT_INFO'] = has_phone or has_email
    
    # Check for valid address
    has_addr = pd.notna(row.get('AddressLine1_clean'))
    has_city = pd.notna(row.get('CityNM_clean'))
    has_state = pd.notna(row.get('StateCD_clean'))
    has_zip = pd.notna(row.get('ZipCD_base'))
    flags['HAS_VALID_ADDRESS'] = has_addr and has_city and has_state and has_zip
    
    return flags


def calculate_completeness_score(row: pd.Series, key_fields: List[str]) -> float:
    """
    Calculate record completeness score.
    """
    populated = sum(1 for f in key_fields if pd.notna(row.get(f)))
    return populated / len(key_fields) if key_fields else 0.0


def generate_composite_quality_score(row: pd.Series) -> Tuple[float, dict]:
    """
    Generate composite data quality score.
    Returns (score, derived_flags)
    """
    weights = {
        'HAS_VALID_NAME': 0.25,
        'HAS_VALID_DOB': 0.20,
        'HAS_VALID_SSN': 0.20,
        'HAS_CONTACT_INFO': 0.15,
        'HAS_VALID_ADDRESS': 0.20
    }
    
    flags = generate_quality_flags(row)
    score = sum(weights[k] * (1 if flags[k] else 0) for k in weights)
    
    derived_flags = {
        'LOW_QUALITY_RECORD': score < 0.4,
        'HIGH_CONFIDENCE_DEMOGRAPHICS': score >= 0.8,
        'PARTIAL_IDENTITY_RECORD': 0.4 <= score < 0.6,
        'LIKELY_TEST_RECORD': not row.get('ValidRecord', True)
    }
    
    return score, derived_flags


# =============================================================================
# MAIN TRANSFORMATION FUNCTION
# =============================================================================

def transform_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply all transformations to the MDM_Population dataset.
    """
    result = df.copy()
    
    # Initialize ValidRecord column
    result['ValidRecord'] = True
    
    # First Name
    if 'FirstNM' in result.columns:
        cleaned = result['FirstNM'].apply(lambda x: clean_first_name(x, True))
        result['FirstNM_clean'] = cleaned.apply(lambda x: x[0])
        result['ValidRecord'] = result['ValidRecord'] & cleaned.apply(lambda x: x[1])
    
    # Last Name
    if 'LastNM' in result.columns:
        cleaned = result['LastNM'].apply(lambda x: clean_last_name(x, True))
        result['LastNM_clean'] = cleaned.apply(lambda x: x[0])
        result['ValidRecord'] = result['ValidRecord'] & cleaned.apply(lambda x: x[1])
    
    # Middle Name
    if 'MiddleNM' in result.columns:
        cleaned = result['MiddleNM'].apply(lambda x: clean_middle_name(x, True))
        result['MiddleNM_clean'] = cleaned.apply(lambda x: x[0])
        result['ValidRecord'] = result['ValidRecord'] & cleaned.apply(lambda x: x[1])
    
    # Suffix
    if 'SuffixNM' in result.columns:
        result['SuffixNM_clean'] = result['SuffixNM'].apply(clean_suffix)
    
    # Cross-column name processing with full redistribution
    if all(col in result.columns for col in ['FirstNM_clean', 'LastNM_clean']):
        middle_col = 'MiddleNM_clean' if 'MiddleNM_clean' in result.columns else None
        suffix_col = 'SuffixNM_clean' if 'SuffixNM_clean' in result.columns else None
        
        def process_names(row):
            first = row.get('FirstNM_clean', np.nan)
            middle = row.get(middle_col, np.nan) if middle_col else np.nan
            last = row.get('LastNM_clean', np.nan)
            suffix = row.get(suffix_col, np.nan) if suffix_col else np.nan
            return process_name_redistribution(first, middle, last, suffix)
        
        name_results = result.apply(process_names, axis=1)
        result['FirstNM_clean'] = name_results.apply(lambda x: x[0])
        result['MiddleNM_clean'] = name_results.apply(lambda x: x[1])
        result['LastNM_clean'] = name_results.apply(lambda x: x[2])
        if suffix_col:
            result['SuffixNM_clean'] = name_results.apply(lambda x: x[3])
        result['INFERRED_LAST_FROM_FIRST'] = name_results.apply(lambda x: x[4].get('INFERRED_LAST_FROM_FIRST', False))
        result['MOVED_MIDDLE_FROM_FIRST'] = name_results.apply(lambda x: x[4].get('MOVED_MIDDLE_FROM_FIRST', False))
        result['CAMELCASE_SPLIT'] = name_results.apply(lambda x: x[4].get('CAMELCASE_SPLIT', False))
        result['EXTRACTED_SUFFIX'] = name_results.apply(lambda x: x[4].get('EXTRACTED_SUFFIX', False))
        result['needs_name_review'] = name_results.apply(lambda x: x[4].get('needs_name_review', False))
        result['REMOVED_DUPLICATE_LAST_TOKEN'] = name_results.apply(lambda x: x[4].get('REMOVED_DUPLICATE_LAST_TOKEN', False))
        
        # Validate name presence - mark invalid if first or last name missing
        def validate_names(row):
            return validate_name_presence(row.get('FirstNM_clean'), row.get('LastNM_clean'))
        
        name_validation = result.apply(validate_names, axis=1)
        result['ValidRecord'] = result['ValidRecord'] & name_validation.apply(lambda x: x[0])
        result['MISSING_FIRST_NAME'] = name_validation.apply(lambda x: x[1].get('MISSING_FIRST_NAME', False))
        result['MISSING_LAST_NAME'] = name_validation.apply(lambda x: x[1].get('MISSING_LAST_NAME', False))
        result['MISSING_BOTH_NAMES'] = name_validation.apply(lambda x: x[1].get('MISSING_BOTH_NAMES', False))
    
    # Date of Birth
    if 'BirthDT' in result.columns:
        dob_results = result['BirthDT'].apply(clean_birth_date)
        result['BirthDT_clean'] = dob_results.apply(lambda x: x[0])
        result['INVALID_DOB'] = dob_results.apply(lambda x: x[1].get('INVALID_DOB', False))
        result['DEFAULT_DOB_PATTERN'] = dob_results.apply(lambda x: x[1].get('DEFAULT_DOB_PATTERN', False))
        result['FUTURE_DOB'] = dob_results.apply(lambda x: x[1].get('FUTURE_DOB', False))
        
        # Derive year, month, day
        result['BirthYear'] = pd.to_datetime(result['BirthDT_clean'], errors='coerce').dt.year
        result['BirthMonth'] = pd.to_datetime(result['BirthDT_clean'], errors='coerce').dt.month
        result['BirthDay'] = pd.to_datetime(result['BirthDT_clean'], errors='coerce').dt.day
    
    # SSN
    if 'SSN' in result.columns:
        ssn_results = result['SSN'].apply(clean_ssn)
        result['SSN_clean'] = ssn_results.apply(lambda x: x[0])
        result['SSN_Last4'] = result['SSN_clean'].apply(extract_last_4_ssn)
        result['is_missing_SSN'] = ssn_results.apply(lambda x: x[1].get('is_missing_SSN', False))
        result['is_junk_SSN'] = ssn_results.apply(lambda x: x[1].get('is_junk_SSN', False))
        result['is_invalid_SSN'] = ssn_results.apply(lambda x: x[1].get('is_invalid_SSN', False))
        result['PADDED_SSN'] = ssn_results.apply(lambda x: x[1].get('PADDED_SSN', False))
    
    # Address Line 1
    if 'AddressLine1' in result.columns:
        addr1_results = result['AddressLine1'].apply(lambda x: clean_address(x, True))
        result['AddressLine1_clean'] = addr1_results.apply(lambda x: x[0])
        result['ValidRecord'] = result['ValidRecord'] & addr1_results.apply(lambda x: x[1])
        result['IS_POBOX'] = addr1_results.apply(lambda x: x[2].get('IS_POBOX', False))
        result['HAS_TRAPPED_UNIT'] = addr1_results.apply(lambda x: x[2].get('HAS_TRAPPED_UNIT', False))
        result['INVALID_ADDRESS'] = addr1_results.apply(lambda x: x[2].get('INVALID_ADDRESS', False))
        result['HOMELESS_ADDRESS'] = addr1_results.apply(lambda x: x[2].get('HOMELESS_ADDRESS', False))
    
    # Address Line 2
    if 'AddressLine2' in result.columns:
        addr2_results = result['AddressLine2'].apply(lambda x: clean_address(x, True))
        result['AddressLine2_clean'] = addr2_results.apply(lambda x: x[0])
        result['ValidRecord'] = result['ValidRecord'] & addr2_results.apply(lambda x: x[1])
    
    # City
    if 'CityNM' in result.columns:
        city_results = result['CityNM'].apply(clean_city)
        result['CityNM_clean'] = city_results.apply(lambda x: x[0])
        result['HAS_NUMERIC_CITY'] = city_results.apply(lambda x: x[1].get('HAS_NUMERIC_CITY', False))
        result['SUSPICIOUS_CITY_LENGTH'] = city_results.apply(lambda x: x[1].get('SUSPICIOUS_CITY_LENGTH', False))
    
    # ZIP Code
    if 'ZipCD' in result.columns:
        zip_results = result['ZipCD'].apply(clean_zip)
        result['ZipCD_base'] = zip_results.apply(lambda x: x[0])
        result['ZipCD_ext'] = zip_results.apply(lambda x: x[1])
        result['INVALID_ZIP'] = zip_results.apply(lambda x: x[2].get('INVALID_ZIP', False))
        result['JUNK_ZIP'] = zip_results.apply(lambda x: x[2].get('JUNK_ZIP', False))
    
    # State
    if 'StateCD' in result.columns:
        state_results = result['StateCD'].apply(clean_state)
        result['StateCD_clean'] = state_results.apply(lambda x: x[0])
        result['INVALID_STATE'] = state_results.apply(lambda x: x[1].get('INVALID_STATE', False))
        result['NUMERIC_STATE_ARTIFACT'] = state_results.apply(lambda x: x[1].get('NUMERIC_STATE_ARTIFACT', False))
    
    # Country
    if 'CountryNM' in result.columns:
        country_results = result['CountryNM'].apply(clean_country)
        result['CountryNM_clean'] = country_results.apply(lambda x: x[0])
        result['IS_FOREIGN_COUNTRY'] = country_results.apply(lambda x: x[1].get('IS_FOREIGN_COUNTRY', False))
        result['MAPPED_NUMERIC_COUNTRY'] = country_results.apply(lambda x: x[1].get('MAPPED_NUMERIC_COUNTRY', False))
        
        # Infer US country from valid state
        if 'StateCD_clean' in result.columns:
            mask = result['CountryNM_clean'].isna() & result['StateCD_clean'].notna()
            result.loc[mask, 'CountryNM_clean'] = 'US'
            result['INFERRED_US_COUNTRY'] = mask
    
    # Phone Numbers
    phone_cols = ['PrimaryPhoneNBR', 'Phone01NBR', 'Phone02NBR', 'Phone03NBR']
    phone_clean_names = ['PhonePrimary_clean', 'Phone01_clean', 'Phone02_clean', 'Phone03_clean']
    
    for orig_col, clean_col in zip(phone_cols, phone_clean_names):
        if orig_col in result.columns:
            phone_results = result[orig_col].apply(clean_phone)
            result[clean_col] = phone_results.apply(lambda x: x[0])
    
    # Create unified phone set
    if any(col in result.columns for col in phone_clean_names):
        def get_unique_phones(row):
            phones = set()
            for col in phone_clean_names:
                if col in row.index and pd.notna(row[col]):
                    phones.add(row[col])
            return list(phones) if phones else np.nan
        result['AllPhones'] = result.apply(get_unique_phones, axis=1)
    
    # Email
    if 'Email' in result.columns:
        email_results = result['Email'].apply(clean_email)
        result['Email_clean'] = email_results.apply(lambda x: x[0])
        result['is_missing_Email'] = email_results.apply(lambda x: x[1].get('is_missing_Email', False))
        result['is_junk_Email'] = email_results.apply(lambda x: x[1].get('is_junk_Email', False))
        result['is_invalid_Email'] = email_results.apply(lambda x: x[1].get('is_invalid_Email', False))
    
    # Sex at Birth
    if 'SexAtBirthDSC' in result.columns:
        sex_results = result['SexAtBirthDSC'].apply(clean_sex_at_birth)
        result['SexAtBirthDSC_clean'] = sex_results.apply(lambda x: x[0])
        result['UNKNOWN_SEX'] = sex_results.apply(lambda x: x[1].get('UNKNOWN_SEX', False))
        result['INVALID_SEX_VALUE'] = sex_results.apply(lambda x: x[1].get('INVALID_SEX_VALUE', False))
    
    # Generate quality flags
    quality_results = result.apply(lambda row: generate_composite_quality_score(row), axis=1)
    result['QUALITY_SCORE'] = quality_results.apply(lambda x: x[0])
    result['LOW_QUALITY_RECORD'] = quality_results.apply(lambda x: x[1].get('LOW_QUALITY_RECORD', False))
    result['HIGH_CONFIDENCE_DEMOGRAPHICS'] = quality_results.apply(lambda x: x[1].get('HIGH_CONFIDENCE_DEMOGRAPHICS', False))
    result['PARTIAL_IDENTITY_RECORD'] = quality_results.apply(lambda x: x[1].get('PARTIAL_IDENTITY_RECORD', False))
    result['LIKELY_TEST_RECORD'] = quality_results.apply(lambda x: x[1].get('LIKELY_TEST_RECORD', False))
    
    # Per-record quality indicators
    quality_flags = result.apply(generate_quality_flags, axis=1)
    result['HAS_VALID_NAME'] = quality_flags.apply(lambda x: x.get('HAS_VALID_NAME', False))
    result['HAS_VALID_DOB'] = quality_flags.apply(lambda x: x.get('HAS_VALID_DOB', False))
    result['HAS_VALID_SSN'] = quality_flags.apply(lambda x: x.get('HAS_VALID_SSN', False))
    result['HAS_CONTACT_INFO'] = quality_flags.apply(lambda x: x.get('HAS_CONTACT_INFO', False))
    result['HAS_VALID_ADDRESS'] = quality_flags.apply(lambda x: x.get('HAS_VALID_ADDRESS', False))
    
    # Completeness metrics
    key_fields = ['FirstNM_clean', 'LastNM_clean', 'BirthDT_clean', 'SSN_clean', 
                  'AddressLine1_clean', 'CityNM_clean', 'StateCD_clean', 'ZipCD_base']
    key_fields = [f for f in key_fields if f in result.columns]
    
    result['PCT_FIELDS_POPULATED'] = result.apply(
        lambda row: calculate_completeness_score(row, key_fields), axis=1
    )
    result['PCT_FIELDS_NULL'] = 1 - result['PCT_FIELDS_POPULATED']
    
    return result
