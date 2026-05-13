### Updated Transformation Rules — SSN and Email Fields
### Source: A2 Blocking Scheme Feasibility Assessment Findings
### Date: May 13, 2026
### Status: Pending colleague review and incorporation into transformations_log.md

---

### FIELD: SSN (Social Security Number)

#### Transformation Rule SSN-1: Strip Formatting Characters
Remove all non-digit characters from the raw SSN field before any
validation is applied. Characters to strip: hyphens (-), spaces ( ),
dots (.), and parentheses ( ).
Example: "123-45-6789" → "123456789"

#### Transformation Rule SSN-2: Length Validation
After stripping formatting, the SSN must be exactly 9 digits in length.
Any value that does not meet this requirement after stripping is nullified
and flagged with is_invalid_SSN = True.

#### Transformation Rule SSN-3: Invalid Area Number Prefix — 000 and 666
Any SSN whose first three digits (the area number) are "000" or "666"
is nullified. The Social Security Administration has never assigned area
numbers 000 or 666. These values are guaranteed to be data entry errors
or system-generated placeholders.

#### Transformation Rule SSN-4: Invalid Area Number Prefix — 9XX Range
Any SSN whose first three digits begin with "9" is nullified. The 9XX
range is reserved for Individual Taxpayer Identification Numbers (ITINs),
which some EHR systems store in the SSN field. ITINs are not SSNs and
must not be used as patient matching identifiers in SSN-based blocking.

NOTE FOR ALLIANCE REVIEW: Flag this rule for explicit confirmation with
the Alliance team. If a meaningful portion of the patient population
consists of ITIN holders, this rule will reduce SSN-based blocking
coverage for those patients. Confirm that other blocking schemes
(B2, B3, B8) provide adequate recall coverage for this population
segment before finalizing this rule.

#### Transformation Rule SSN-5: Invalid Group Number — 00
The group number is the middle two digits of the SSN (positions 4–5).
Any SSN with a group number of "00" is nullified. The SSA has never
issued SSNs with a group number of 00. These are confirmed invalid values.

#### Transformation Rule SSN-6: Invalid Serial Number — 0000
The serial number is the last four digits of the SSN (positions 6–9).
Any SSN with a serial number of "0000" is nullified. The SSA has never
issued SSNs with a serial number of 0000. These are confirmed invalid values.

#### Transformation Rule SSN-7: Repeating Single Digit
Any SSN consisting entirely of a single repeated digit is nullified.
This covers all values where every position contains the same digit
(e.g. all nines, all ones, all zeros). These are confirmed junk
placeholder values.
Examples nullified: "000000000", "111111111", "999999999" (all variants)

#### Transformation Rule SSN-8: Known Sequential Patterns
The following specific sequential digit patterns are confirmed junk values
that appear in healthcare EHR systems and must be nullified:
  - "123456789" — ascending sequential
  - "987654321" — descending sequential
  - "123123123" — repeating three-digit block

#### Transformation Rule SSN-9: Known Exact Junk Values
The following specific SSN values are historically documented as
widely-circulated invalid or placeholder SSNs and must be nullified
regardless of whether they pass other validation rules:
  - "010101010" — alternating pattern
  - "090909090" — alternating pattern variant
  - "000000001" — incremental from zero, common system default
  - "999999998" — near-maximum, common system default
  - "111223333" — the Woolworth wallet card SSN, historically distributed
  - "219099999" — historically advertised SSN (Woolworth)
  - "457555462" — historically advertised SSN

#### Transformation Rule SSN-10: Flag Assignment
After all nullification rules above are applied, assign the following
boolean flag columns to every record:
  - is_missing_SSN = True if the raw SSN field was null or empty before
    any processing
  - is_junk_SSN = True if the raw SSN was non-null but was nullified by
    any of Rules SSN-2 through SSN-9
  - clean_SSN = the validated 9-digit string for records passing all rules,
    null for all others

#### Transformation Rule SSN-11: Last-4 Extraction
After clean_SSN is assigned, extract the last four digits of clean_SSN
into a separate column SSN_Last4. This column is used by Blocking Scheme
B9 (LastNM + FirstNM + SSN_Last4). SSN_Last4 is null for any record
where clean_SSN is null.

---

### FIELD: Email

#### Transformation Rule EMAIL-1: Lowercase Normalization
Convert all non-null email values to lowercase and strip leading and
trailing whitespace before any validation is applied. Email addresses
are case-insensitive by standard and must be normalized to lowercase
for reliable exact-match blocking.

#### Transformation Rule EMAIL-2: Format Validation — Presence of @
Any email value that does not contain the @ character after lowercase
normalization is nullified and flagged with is_invalid_Email = True.

#### Transformation Rule EMAIL-3: Format Validation — Domain Structure
Any email value that does not match the pattern of at least one character
before @, the @ symbol itself, at least one character, a dot, and at
least one character after the dot is nullified. This ensures the value
has a structurally valid local part and domain.
Pattern required: [one or more characters]@[one or more characters].[one
or more characters]

#### Transformation Rule EMAIL-4: Nullify Empty and Primitive Placeholders
After format validation, nullify any email value that is one of the
following primitive placeholder strings: "nan", "none", "null", "na",
or empty string. These represent null values that were coerced to strings
during data extraction from the source EHR system.

#### Transformation Rule EMAIL-5: Exact Known Junk Email Values
The following email addresses are confirmed system-generated or
clinic-default placeholder values identified in the dataset during the
A2 feasibility assessment. Each appeared in a significant number of
records and must be nullified:
  - noemail@noemail.com
  - noemail@email.com
  - no@email.com
  - none@none.com
  - unknown@unknown.com
  - unknown@email.com
  - test@test.com
  - test@example.com
  - donotreply@donotreply.com
  - noreply@noreply.com
  - email@email.com
  - patient@patient.com
  - none@noemail.com
  - na@na.com
  - null@null.com

NOTE: This list must be treated as a living list. After the full cleaning
pipeline runs, any email address appearing in more than 50 records in the
cleaned dataset should be reviewed and added to this list if it is
confirmed to be a clinic default or system placeholder. The A2 assessment
identified one surviving email shared by 91 records post-fix that
requires targeted review before the full pipeline run.

#### Transformation Rule EMAIL-6: Pattern-Based Junk Detection — Local Part
Any email whose local part (the portion before @) matches any of the
following patterns is nullified, regardless of the domain:
  - Begins with "noemail" or "no-email" or "no_email"
  - Begins with "noreply" or "no-reply"
  - Begins with "donotreply" or "do-not-reply"
  - Begins with "unknown"
  - Begins with "test@"
  - Begins with "patient@"
  - Begins with "none@"
  - Begins with "null@"
  - Begins with "na@"
  - Local part is 1 or 2 characters in total length (single or double
    character local parts are not valid personal email addresses and
    represent either data entry errors or system defaults)
  - Contains the substring "123456" (sequential digit pattern)

#### Transformation Rule EMAIL-7: Pattern-Based Junk Detection — Domain
Any email whose domain (the portion after @) matches any of the following
patterns is nullified, regardless of the local part:
  - @example.com, @example.org, @example.net (RFC reserved test domains)
  - @test.com, @test.org
  - @noemail.com
  - @noreply.com
  - @donotreply.com
  - @unknown.com
  - @123.com

#### Transformation Rule EMAIL-8: Flag Assignment
After all nullification rules above are applied, assign the following
boolean flag columns to every record:
  - is_missing_Email = True if the raw email field was null or empty
    before any processing
  - is_junk_Email = True if the raw email was non-null but was nullified
    by any of Rules EMAIL-2 through EMAIL-7
  - clean_Email = the validated lowercase email string for records passing
    all rules, null for all others

#### Transformation Rule EMAIL-9: Post-Cleaning Frequency Audit (QA Step)
After the full cleaning pipeline runs on the complete dataset, the
data engineering lead must run a frequency count on clean_Email and
review the top 20 most frequent values. Any clean_Email value appearing
in more than 50 records must be manually reviewed and either:
  (a) Added to the EMAIL-5 exact junk list if confirmed to be a clinic
      default or system placeholder, or
  (b) Documented as a known legitimate shared email (e.g. a social
      services organization email used as a patient contact) with a
      written justification in this transformation log.
This audit step is mandatory before the blocking pipeline runs and must
be signed off by the data engineering lead.