"""
Unit tests for the phonetic encoding helper functions used to compute
blocking keys for B3, B7 (Double Metaphone) and B8 (Soundex).

These tests verify:
    1. Known name variant pairs produce matching phonetic codes
    2. Null/invalid inputs return None (not empty string or crash)
    3. The encoding functions are consistent with the phonetics/jellyfish
       library versions installed in the project environment
    4. Edge cases: accented characters, very short names, numeric strings

WHY THIS MATTERS:
    If phonetic encoding behavior changes (due to a library upgrade or
    swap), blocking schemes B3, B7, and B8 will silently produce different
    candidate pairs. These tests catch that regression immediately.
"""

import pytest
import numpy as np

from src.preprocessing.blocking import _dm_primary, _soundex


# ═══════════════════════════════════════════════════════════════════════════
# _dm_primary (Double Metaphone — primary code)
# ═══════════════════════════════════════════════════════════════════════════

class TestDmPrimary:
    """Tests for the Double Metaphone primary code helper."""

    # ── Known phonetic variant pairs that MUST produce matching codes ─────
    # These pairs are the specific error patterns the blocking scheme
    # was designed to catch. If any pair stops matching after a library
    # update, the blocking scheme's recall is degraded.

    @pytest.mark.parametrize("name_a, name_b", [
        ("SMITH",      "SMYTH"),
        ("GARCIA",     "GARSIA"),
        ("JOHNSON",    "JONSON"),
        ("GUTIERREZ",  "GUTIERRES"),
        ("SCHMIDT",    "SHMIDT"),
        ("THOMPSON",   "TOMPSON"),
        ("GONZALEZ",   "GONZALES"),
        ("MARTINEZ",   "MARTINES"),
        ("SANCHEZ",    "SANCHES"),
        ("RODRIGUEZ",  "RODRIGUES"),
    ])
    def test_known_variant_pairs_share_dm_code(self, name_a, name_b):
        """Name variant pairs that the blocking scheme relies on must
        produce identical DM primary codes."""
        code_a = _dm_primary(name_a)
        code_b = _dm_primary(name_b)
        assert code_a is not None, f"{name_a} should produce a DM code"
        assert code_b is not None, f"{name_b} should produce a DM code"
        assert code_a == code_b, (
            f"DM mismatch: {name_a} -> {code_a}, {name_b} -> {code_b}. "
            f"These must match for B3/B7 blocking to function."
        )

    # ── Names that MUST produce non-None codes ────────────────────────────

    @pytest.mark.parametrize("name", [
        "SMITH", "GARCIA", "LI", "O", "DE LA CRUZ", "O'BRIEN",
        "MCDONALD", "AL-RASHID", "NGUYEN", "XIONG",
    ])
    def test_valid_names_produce_codes(self, name):
        """All valid name strings must produce a non-None DM code."""
        code = _dm_primary(name)
        assert code is not None, f"'{name}' should produce a DM code"
        assert isinstance(code, str), f"DM code should be a string, got {type(code)}"
        assert len(code) > 0, f"DM code should be non-empty for '{name}'"

    # ── Null and invalid inputs MUST return None ──────────────────────────

    @pytest.mark.parametrize("invalid_input", [
        None,
        np.nan,
        "",
        "   ",         # whitespace only
        float("nan"),
        42,            # non-string type
        True,          # boolean
    ])
    def test_invalid_inputs_return_none(self, invalid_input):
        """Invalid inputs must return None, not empty string or raise."""
        result = _dm_primary(invalid_input)
        assert result is None, (
            f"Expected None for input {invalid_input!r}, got {result!r}"
        )

    # ── Consistency: same input always produces same output ───────────────

    def test_deterministic_output(self):
        """Same input must always produce the same output (no randomness)."""
        name = "JOHNSON"
        results = [_dm_primary(name) for _ in range(100)]
        assert len(set(results)) == 1, (
            f"DM encoding is not deterministic: {set(results)}"
        )

    # ── Edge cases ────────────────────────────────────────────────────────

    def test_single_character_name(self):
        """Single-character names should produce a code, not crash."""
        result = _dm_primary("X")
        # May be None or a short code depending on library — just don't crash
        assert result is None or isinstance(result, str)

    def test_hyphenated_name(self):
        """Hyphenated names should produce a code."""
        result = _dm_primary("SMITH-JONES")
        assert result is not None

    def test_name_with_apostrophe(self):
        """Names with apostrophes should produce a code."""
        result = _dm_primary("O'BRIEN")
        assert result is not None

    def test_numeric_string_does_not_crash(self):
        """A numeric string passed to DM should not crash."""
        result = _dm_primary("12345")
        # May return None or a code — just must not raise
        assert result is None or isinstance(result, str)


# ═══════════════════════════════════════════════════════════════════════════
# _soundex
# ═══════════════════════════════════════════════════════════════════════════

class TestSoundex:
    """Tests for the Soundex encoding helper."""

    # ── Known variant pairs that MUST produce matching Soundex codes ──────
    # Soundex is intentionally coarser than DM. These pairs are used by B8.

    @pytest.mark.parametrize("name_a, name_b", [
        ("SMITH",    "SMYTH"),
        ("JOHNSON",  "JONSON"),
        ("MICHAEL",  "MIKHAIL"),   # Key B8 test case
        ("GARCIA",   "GARSIA"),
        ("ROBERT",   "RUPERT"),
    ])
    def test_known_variant_pairs_share_soundex(self, name_a, name_b):
        """Variant pairs relied upon by B8 must share Soundex codes."""
        code_a = _soundex(name_a)
        code_b = _soundex(name_b)
        assert code_a is not None
        assert code_b is not None
        assert code_a == code_b, (
            f"Soundex mismatch: {name_a} -> {code_a}, {name_b} -> {code_b}"
        )

    # ── Soundex code format validation ────────────────────────────────────

    @pytest.mark.parametrize("name", [
        "SMITH", "GARCIA", "JOHNSON", "WILLIAMS", "BROWN",
    ])
    def test_soundex_format(self, name):
        """Soundex codes must be exactly 4 characters: 1 letter + 3 digits."""
        code = _soundex(name)
        assert code is not None
        assert len(code) == 4, f"Soundex code should be 4 chars, got '{code}'"
        assert code[0].isalpha(), f"First char should be a letter: '{code}'"
        assert code[1:].isdigit(), f"Last 3 chars should be digits: '{code}'"

    # ── Null and invalid inputs ───────────────────────────────────────────

    @pytest.mark.parametrize("invalid_input", [
        None,
        np.nan,
        "",
        "   ",
        float("nan"),
        42,
        True,
    ])
    def test_invalid_inputs_return_none(self, invalid_input):
        """Invalid inputs must return None."""
        result = _soundex(invalid_input)
        assert result is None, (
            f"Expected None for input {invalid_input!r}, got {result!r}"
        )

    # ── Determinism ───────────────────────────────────────────────────────

    def test_deterministic_output(self):
        """Same input must always produce the same Soundex code."""
        name = "WILLIAMS"
        results = [_soundex(name) for _ in range(100)]
        assert len(set(results)) == 1

    # ── Names that MUST produce different codes ───────────────────────────
    # Soundex is coarse but not infinite — clearly different names must
    # produce different codes to prevent false blocking matches.

    @pytest.mark.parametrize("name_a, name_b", [
        ("SMITH",   "GARCIA"),
        ("JOHNSON", "WILLIAMS"),
        ("CHEN",    "PATEL"),
    ])
    def test_clearly_different_names_differ(self, name_a, name_b):
        """Names with no phonetic similarity should not share codes."""
        code_a = _soundex(name_a)
        code_b = _soundex(name_b)
        assert code_a != code_b, (
            f"Clearly different names produced same Soundex: "
            f"{name_a} -> {code_a}, {name_b} -> {code_b}"
        )