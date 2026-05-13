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
    Clean SSN field.
    Returns (cleaned_ssn, indicators_dict)
    """
    indicators = {
        'INVALID_SSN': False,
        'JUNK_SSN': False,
        'PADDED_SSN': False
    }
    
    if pd.isna(value):
        return np.nan, indicators
    
    # Remove non-numeric characters
    ssn = re.sub(r'\D', '', str(value))
    
    # Left-pad short SSNs
    if len(ssn) in (7, 8):
        ssn = ssn.zfill(9)
        indicators['PADDED_SSN'] = True
    
    # Must be exactly 9 digits
    if len(ssn) != 9:
        indicators['INVALID_SSN'] = True
        return np.nan, indicators
    
    # Check for junk patterns
    junk_patterns = {'999999999', '000000000', '123456789'}
    if ssn in junk_patterns:
        indicators['JUNK_SSN'] = True
        return np.nan, indicators
    
    # Check for repeating digits
    if len(set(ssn)) == 1:
        indicators['JUNK_SSN'] = True
        return np.nan, indicators
    
    # Check for invalid prefixes
    area_number = ssn[:3]
    if area_number == '666' or (900 <= int(area_number) <= 999):
        indicators['INVALID_SSN'] = True
        return np.nan, indicators
    
    # Check for all-zero group or serial
    if ssn[3:5] == '00' or ssn[5:9] == '0000':
        indicators['INVALID_SSN'] = True
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
    Clean email field.
    Returns (cleaned_email, indicators_dict)
    """
    indicators = {
        'INVALID_EMAIL': False,
        'JUNK_EMAIL': False
    }
    
    if pd.isna(value):
        return np.nan, indicators
    
    text = str(value).lower().strip()
    
    # Standardize nulls
    if text.upper() in NULL_VALUES or text == '':
        return np.nan, indicators
    
    # Check for placeholder emails (check both full strings and prefixes)
    if text in PLACEHOLDER_EMAILS:
        indicators['JUNK_EMAIL'] = True
        return np.nan, indicators
    
    # Check prefix placeholders
    prefix_placeholders = {'test@', 'noemail@', 'noemai@'}
    for prefix in prefix_placeholders:
        if text.startswith(prefix):
            indicators['JUNK_EMAIL'] = True
            return np.nan, indicators
    
    # Must contain @
    if '@' not in text:
        indicators['INVALID_EMAIL'] = True
        return np.nan, indicators
    
    # Basic email validation
    email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    if not re.match(email_pattern, text):
        indicators['INVALID_EMAIL'] = True
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

def process_name_tokens(first_nm: str, middle_nm: str, last_nm: str) -> Tuple[str, str, str, dict]:
    """
    Cross-column name separation and standardization.
    Returns (first_clean, middle_clean, last_clean, indicators_dict)
    """
    indicators = {
        'INFERRED_LAST_FROM_FIRST': False,
        'REMOVED_DUPLICATE_LAST_TOKEN': False
    }
    
    first_clean = first_nm if pd.notna(first_nm) else ''
    middle_clean = middle_nm if pd.notna(middle_nm) else ''
    last_clean = last_nm if pd.notna(last_nm) else ''
    
    # Tokenize first name
    first_tokens = first_clean.split() if first_clean else []
    
    # Handle missing last name - infer from first name tokens
    if not last_clean and len(first_tokens) > 1:
        last_clean = first_tokens[-1]
        first_tokens = first_tokens[:-1]
        if len(first_tokens) > 1:
            middle_clean = ' '.join(first_tokens[1:])
            first_clean = first_tokens[0]
        else:
            first_clean = first_tokens[0] if first_tokens else ''
        indicators['INFERRED_LAST_FROM_FIRST'] = True
    
    # Check for duplicate last name in first name tokens
    elif last_clean and first_tokens:
        if first_tokens[-1].upper() == last_clean.upper():
            first_tokens = first_tokens[:-1]
            if len(first_tokens) > 1:
                middle_clean = ' '.join(first_tokens[1:]) if not middle_clean else middle_clean
                first_clean = first_tokens[0]
            elif first_tokens:
                first_clean = first_tokens[0]
            else:
                first_clean = ''
            indicators['REMOVED_DUPLICATE_LAST_TOKEN'] = True
    
    # Convert empty strings to NaN
    first_clean = first_clean if first_clean else np.nan
    middle_clean = middle_clean if middle_clean else np.nan
    last_clean = last_clean if last_clean else np.nan
    
    return first_clean, middle_clean, last_clean, indicators


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
    
    # Cross-column name processing
    if all(col in result.columns for col in ['FirstNM_clean', 'LastNM_clean']):
        middle_col = 'MiddleNM_clean' if 'MiddleNM_clean' in result.columns else None
        
        def process_names(row):
            first = row.get('FirstNM_clean', np.nan)
            middle = row.get(middle_col, np.nan) if middle_col else np.nan
            last = row.get('LastNM_clean', np.nan)
            return process_name_tokens(first, middle, last)
        
        name_results = result.apply(process_names, axis=1)
        result['FirstNM_clean'] = name_results.apply(lambda x: x[0])
        result['MiddleNM_clean'] = name_results.apply(lambda x: x[1])
        result['LastNM_clean'] = name_results.apply(lambda x: x[2])
        result['INFERRED_LAST_FROM_FIRST'] = name_results.apply(lambda x: x[3].get('INFERRED_LAST_FROM_FIRST', False))
        result['REMOVED_DUPLICATE_LAST_TOKEN'] = name_results.apply(lambda x: x[3].get('REMOVED_DUPLICATE_LAST_TOKEN', False))
    
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
        result['last_4_SSN'] = result['SSN_clean'].apply(extract_last_4_ssn)
        result['INVALID_SSN'] = ssn_results.apply(lambda x: x[1].get('INVALID_SSN', False))
        result['JUNK_SSN'] = ssn_results.apply(lambda x: x[1].get('JUNK_SSN', False))
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
        result['INVALID_EMAIL'] = email_results.apply(lambda x: x[1].get('INVALID_EMAIL', False))
        result['JUNK_EMAIL'] = email_results.apply(lambda x: x[1].get('JUNK_EMAIL', False))
    
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
