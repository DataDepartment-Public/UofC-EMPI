"""
Adversarial unit tests for the _parse_phone_set function.

THIS IS THE HIGHEST-PRIORITY TEST FILE IN THE ENTIRE SUITE.

_parse_phone_set is the most failure-prone component in blocking.py.
The cleaning pipeline serializes Phones_set as a Python string, and this
function must correctly parse it back to a native set. A silent failure
here causes B5 to generate zero candidate pairs — the exact bug that
occurred during the A3 Redundancy Analysis run, producing invalid results
that nearly led to removing B5 (the single highest-value block at 87%
unique pairs).

These tests are intentionally adversarial: they cover every serialization
format observed in the wild, plus edge cases designed to break naive
parsing approaches (malformed strings, mixed quote styles, nested
structures, whitespace variations).

RULE: If you encounter a new Phones_set format in production data that
_parse_phone_set does not handle, add a test case HERE first, then fix
the function. Test-first ensures the fix is permanent.
"""

import pytest
import numpy as np
import pandas as pd

from src.features.blocking import _parse_phone_set


class TestParsePhoneSetValidFormats:
    """Tests for correctly formatted phone set strings that MUST parse."""

    def test_python_set_literal_multi_element(self):
        """Standard Python set literal with multiple phone numbers."""
        result = _parse_phone_set("{'7735551234', '3125559876'}")
        assert isinstance(result, set)
        assert result == {"7735551234", "3125559876"}

    def test_python_set_literal_single_element(self):
        """Set literal with exactly one phone number."""
        result = _parse_phone_set("{'7735551234'}")
        assert isinstance(result, set)
        assert result == {"7735551234"}

    def test_python_list_literal(self):
        """List literal format (some serializers use lists, not sets)."""
        result = _parse_phone_set("['7735551234', '3125559876']")
        assert isinstance(result, set)
        assert result == {"7735551234", "3125559876"}

    def test_python_set_with_double_quotes(self):
        """Set literal using double quotes instead of single quotes."""
        result = _parse_phone_set('{"7735551234", "3125559876"}')
        assert isinstance(result, set)
        assert result == {"7735551234", "3125559876"}

    def test_three_element_set(self):
        """Set with three phone numbers."""
        result = _parse_phone_set("{'7735551234', '3125559876', '8475553333'}")
        assert isinstance(result, set)
        assert len(result) == 3
        assert "8475553333" in result

    def test_native_python_set_object(self):
        """If Phones_set is already a native set (not serialized), return it."""
        input_set = {"7735551234", "3125559876"}
        result = _parse_phone_set(input_set)
        assert result == input_set

    def test_native_python_list_object(self):
        """If Phones_set is already a native list, convert to set."""
        input_list = ["7735551234", "3125559876"]
        result = _parse_phone_set(input_list)
        assert isinstance(result, set)
        assert result == {"7735551234", "3125559876"}

    def test_native_list_with_duplicates(self):
        """Duplicate phones in a list should be deduplicated in the set."""
        result = _parse_phone_set(["7735551234", "7735551234", "3125559876"])
        assert isinstance(result, set)
        assert len(result) == 2

    def test_plain_single_number_string(self):
        """A plain phone number string without braces or brackets."""
        result = _parse_phone_set("7735551234")
        assert isinstance(result, set)
        assert result == {"7735551234"}


class TestParsePhoneSetEmptyAndNull:
    """Tests for inputs that represent empty or null phone data.
    All must return an empty set, never None or an exception."""

    @pytest.mark.parametrize("empty_input, description", [
        (None,          "Python None"),
        (np.nan,        "numpy NaN"),
        (float("nan"),  "float NaN"),
        (pd.NA,         "pandas NA"),
        ("",            "empty string"),
        ("nan",         "string 'nan'"),
        ("None",        "string 'None'"),
        ("set()",       "string 'set()'"),
        ("{}",          "string '{}'"),
        ("[]",          "string '[]'"),
        ("  ",          "whitespace only"),
        (" nan ",       "padded 'nan'"),
    ])
    def test_empty_inputs_return_empty_set(self, empty_input, description):
        """All null/empty representations must return an empty set."""
        result = _parse_phone_set(empty_input)
        assert isinstance(result, set), (
            f"Expected set for {description}, got {type(result)}"
        )
        assert len(result) == 0, (
            f"Expected empty set for {description}, got {result}"
        )

    def test_return_type_is_always_set(self):
        """Even on null input, the return type must be set (not None)."""
        result = _parse_phone_set(None)
        assert result is not None
        assert isinstance(result, set)


class TestParsePhoneSetEdgeCases:
    """Adversarial edge cases designed to break naive parsing."""

    def test_extra_whitespace_around_braces(self):
        """Whitespace around and between elements."""
        result = _parse_phone_set("  { '7735551234' ,  '3125559876' }  ")
        assert "7735551234" in result
        assert "3125559876" in result

    def test_mixed_quote_styles(self):
        """Some serializers mix single and double quotes."""
        result = _parse_phone_set("{'7735551234', \"3125559876\"}")
        assert len(result) == 2

    def test_numeric_type_in_native_set(self):
        """If phone numbers are stored as integers in a set."""
        result = _parse_phone_set({7735551234, 3125559876})
        assert isinstance(result, set)
        assert "7735551234" in result or 7735551234 in result

    def test_whitespace_within_phone_numbers(self):
        """Phones with internal whitespace should be stripped."""
        result = _parse_phone_set("{' 7735551234 ', ' 3125559876 '}")
        # After parsing and stripping, the phones should be clean
        cleaned = {p.strip() for p in result}
        assert "7735551234" in cleaned

    def test_empty_elements_filtered_out(self):
        """Empty strings within the set should not appear in output."""
        result = _parse_phone_set("{'7735551234', '', '  '}")
        assert "" not in result
        assert "  " not in result
        assert "7735551234" in result

    def test_non_string_non_collection_type(self):
        """Unexpected types (int, bool, float) return empty set."""
        assert _parse_phone_set(42) == set()
        assert _parse_phone_set(True) == set()
        assert _parse_phone_set(3.14) == set()

    def test_deeply_nested_structure_does_not_crash(self):
        """Malformed nested structure should not raise — return best effort."""
        result = _parse_phone_set("{{'7735551234'}}")
        assert isinstance(result, set)
        # May or may not parse the inner phone — must not crash

    def test_comma_separated_no_braces(self):
        """Plain comma-separated string without delimiters."""
        result = _parse_phone_set("7735551234,3125559876")
        assert isinstance(result, set)
        assert len(result) == 2

    def test_semicolon_separated_does_not_crash(self):
        """Semicolons instead of commas — should not crash."""
        result = _parse_phone_set("7735551234;3125559876")
        assert isinstance(result, set)
        # May return as one concatenated string — must not crash

    def test_very_long_phone_set(self):
        """A set with many phones should parse without issues."""
        phones = [f"773555{i:04d}" for i in range(50)]
        input_str = "{" + ", ".join(f"'{p}'" for p in phones) + "}"
        result = _parse_phone_set(input_str)
        assert isinstance(result, set)
        assert len(result) == 50


class TestParsePhoneSetConsistency:
    """Tests that verify consistency across different representations
    of the same data. These catch format-dependent bugs."""

    def test_set_string_equals_native_set(self):
        """String-serialized set and native set should produce identical output."""
        phones = {"7735551234", "3125559876"}
        from_string = _parse_phone_set(str(phones))
        from_native = _parse_phone_set(phones)
        assert from_string == from_native, (
            f"String format and native set produce different results: "
            f"string={from_string}, native={from_native}"
        )

    def test_list_string_equals_native_list(self):
        """String-serialized list and native list produce identical output."""
        phones = ["7735551234", "3125559876"]
        from_string = _parse_phone_set(str(phones))
        from_native = _parse_phone_set(phones)
        assert from_string == from_native

    def test_order_does_not_matter(self):
        """Phone set parsing should be order-independent."""
        result_a = _parse_phone_set("{'3125559876', '7735551234'}")
        result_b = _parse_phone_set("{'7735551234', '3125559876'}")
        assert result_a == result_b

    def test_idempotent_on_native_set(self):
        """Parsing a set that was already parsed should return identical result."""
        original = {"7735551234", "3125559876"}
        first_parse  = _parse_phone_set(original)
        second_parse = _parse_phone_set(first_parse)
        assert first_parse == second_parse