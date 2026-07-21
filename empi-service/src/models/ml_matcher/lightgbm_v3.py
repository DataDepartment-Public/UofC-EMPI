"""Concrete BYOF + BYOM implementation for the pluggable ML matcher (Stage 4.5)
— the LightGBM **v3 ambiguous-pair classifier**.

Ports the feature engineering of
`notebooks/ml_model/pair_classifier_lightgbm_ambiguous_v3.ipynb` (§5) into a
`FeatureBuilder`, and adapts the notebook's fitted model to the pipeline's
match-scorer contract.

Two pieces:

* ``V3FeatureBuilder`` — turns candidate pairs + cleaned records into the 12
  v3 features (3 categoricals + 9 numerics). No dependency on the FS matcher's
  ``fs_features``; features come straight from the cleaned attributes.
* ``MatchProbabilityAdapter`` — the notebook model predicts ``P(ambiguous)``
  as class 1 (positive = "route to review"), but the pipeline treats
  ``predict_proba(X)[:, 1]`` as ``match_probability`` and maps *high* scores to
  the ``auto_merge`` tier. This adapter swaps the two probability columns at
  serve time so column 1 is ``P(confident match) = 1 - P(ambiguous)``. The
  serialized model artifact pickles an instance of this class, so it must stay
  importable from this module for ``registry.load_model_artifact`` (joblib) to
  deserialize it.

HIPAA: no PHI is logged here (feature building runs silently; the matcher logs
aggregate tier counts only, mirroring the FS matcher).
"""

from __future__ import annotations

import ast
import re
from typing import Any

import jellyfish
import numpy as np
import pandas as pd

__all__ = ["V3FeatureBuilder", "MatchProbabilityAdapter", "FEATURE_COLS", "CATEGORICAL_FEATURES"]

# ── Feature roster (must match the notebook's training columns + order) ───────
MISSING, SAME, DIFFERENT = "missing", "same", "different"
COMPARE_LEVELS = [MISSING, SAME, DIFFERENT]

CATEGORICAL_FEATURES = ["sound_first", "sound_last", "cmp_street_num"]
NUMERIC_FEATURES = [
    "sim_jw_first", "sim_jw_last", "sim_jw_middle",
    "sim_lev_email", "sim_lev_address1", "addr_token_jaccard",
    "ssn_digit_frac", "sim_dob", "sim_phones",
]
# LightGBM was fit on this exact column order (cat_features + num_features).
FEATURE_COLS = CATEGORICAL_FEATURES + NUMERIC_FEATURES

# Cleaned columns the builder reads (suffixed _A/_B after the pair join).
_CLEAN_COLS = [
    "FirstNM_clean", "LastNM_clean", "MiddleNM_clean", "BirthDT_clean",
    "SSN_clean", "Email_clean", "AddressLine1_clean", "Phones_set",
]

_num_re = re.compile(r"\d+")
_NA_TOKENS = {"nan", "none", "<na>", "nat", "null"}


# ── Scalar helpers (ported verbatim from the v3 notebook §5) ──────────────────
def _norm(x: Any) -> str | None:
    """Value -> lowercased stripped str, or None for any missing kind."""
    if isinstance(x, str):
        s = x.strip().lower()
        return s if s and s not in _NA_TOKENS else None
    if x is None:
        return None
    try:
        if pd.isna(x):
            return None
    except (TypeError, ValueError):
        pass
    s = str(x).strip().lower()
    return s if s and s not in _NA_TOKENS else None


def _norm_series(s: pd.Series) -> pd.Series:
    return s.astype(object).map(_norm)


def _parse_set(value: Any) -> set:
    """Set-valued cell (phones) -> set of strings. Handles native
    array/list/set and the legacy stringified form."""
    if isinstance(value, (set, frozenset, list, tuple, np.ndarray)):
        return {str(p).strip() for p in value if str(p).strip()}
    if not isinstance(value, str):
        # non-str scalar (incl. NaN/None) -> empty
        try:
            if pd.isna(value):
                return set()
        except (TypeError, ValueError):
            pass
        return set()
    v = value.strip()
    if v in ("", "nan", "None", "set()", "{}", "[]"):
        return set()
    try:
        parsed = ast.literal_eval(v)
        if isinstance(parsed, (set, list, tuple)):
            return {str(p).strip() for p in parsed if str(p).strip()}
    except (ValueError, SyntaxError, TypeError):
        pass
    cleaned = v.strip("{}[]").replace("'", "").replace('"', "")
    return {p.strip() for p in cleaned.split(",") if p.strip()}


def _jw(x: Any, y: Any) -> float:
    if not (isinstance(x, str) and isinstance(y, str)):
        return np.nan
    return jellyfish.jaro_winkler_similarity(x, y)


def _lev_sim(x: Any, y: Any) -> float:
    if not (isinstance(x, str) and isinstance(y, str)):
        return np.nan
    m = max(len(x), len(y))
    return 1.0 - jellyfish.levenshtein_distance(x, y) / m if m else np.nan


def _metaphone(x: Any) -> str | None:
    s = _norm(x)
    if not s:
        return None
    return jellyfish.metaphone(s) or None


def _street_num(x: Any) -> str | None:
    s = _norm(x)
    if not s:
        return None
    m = _num_re.search(s)
    return m.group() if m else None


def _tok_jaccard(x: Any, y: Any) -> float:
    x, y = _norm(x), _norm(y)
    if not x or not y:
        return np.nan
    sx, sy = set(x.split()), set(y.split())
    return len(sx & sy) / len(sx | sy) if sx and sy else np.nan


def _ssn_frac(x: Any, y: Any) -> float:
    x, y = _norm(x), _norm(y)
    if not x or not y:
        return np.nan
    n = min(len(x), len(y))
    if n == 0:
        return np.nan
    return sum(1 for i in range(n) if x[i] == y[i]) / max(len(x), len(y))


def _phones_best_jw(av: Any, bv: Any) -> float:
    A, B = _parse_set(av), _parse_set(bv)
    if not A or not B:
        return np.nan
    return max(jellyfish.jaro_winkler_similarity(x, y) for x in A for y in B)


def _cmp_categorical(a_norm: pd.Series, b_norm: pd.Series) -> pd.Categorical:
    """same/different/missing categorical from two already-normalized series."""
    missing = a_norm.isna() | b_norm.isna()
    same = (a_norm == b_norm) & ~missing
    out = np.where(missing, MISSING, np.where(same, SAME, DIFFERENT))
    return pd.Categorical(out, categories=COMPARE_LEVELS)


class V3FeatureBuilder:
    """`FeatureBuilder` for the LightGBM v3 model.

    Returns a frame keyed by ``PATID_A``/``PATID_B`` with exactly the 12
    ``FEATURE_COLS`` (categoricals as ``category`` dtype with the training
    category ordering, numerics as float). ``fs_features`` is ignored. A
    missing ``MiddleNM_clean`` column degrades gracefully to a NaN
    ``sim_jw_middle`` (the notebook is ~96% NaN there anyway).
    """

    def build_features(
        self,
        candidate_pairs: pd.DataFrame,
        df_clean: pd.DataFrame,
        fs_features: pd.DataFrame | None = None,  # noqa: ARG002 - part of the Protocol
    ) -> pd.DataFrame:
        pairs = self._materialize_pairs(candidate_pairs, df_clean)
        feat = pd.DataFrame(index=pairs.index)

        # --- Names: Jaro-Winkler (missing MiddleNM_clean -> NaN column) ---
        for feat_name, col in [
            ("sim_jw_first", "FirstNM_clean"),
            ("sim_jw_last", "LastNM_clean"),
            ("sim_jw_middle", "MiddleNM_clean"),
        ]:
            feat[feat_name] = self._sim_column(pairs, col, _jw)

        # --- Email / address: normalized Levenshtein ---
        feat["sim_lev_email"] = self._sim_column(pairs, "Email_clean", _lev_sim)
        feat["sim_lev_address1"] = self._sim_column(pairs, "AddressLine1_clean", _lev_sim)

        # --- Names: phonetic (Metaphone) sound-alike match ---
        feat["sound_first"] = self._sound_cmp(pairs, "FirstNM_clean")
        feat["sound_last"] = self._sound_cmp(pairs, "LastNM_clean")

        # --- Address: street number exact + token Jaccard ---
        sn_a = self._side(pairs, "AddressLine1_clean", "A").map(_street_num)
        sn_b = self._side(pairs, "AddressLine1_clean", "B").map(_street_num)
        feat["cmp_street_num"] = _cmp_categorical(sn_a, sn_b)
        feat["addr_token_jaccard"] = self._pair_apply(
            pairs, "AddressLine1_clean", _tok_jaccard,
        )

        # --- SSN: fraction of position-wise matching digits ---
        feat["ssn_digit_frac"] = self._pair_apply(pairs, "SSN_clean", _ssn_frac)

        # --- DOB: normalized Levenshtein on the YYYYMMDD digit string ---
        dob_a = pd.to_datetime(self._side(pairs, "BirthDT_clean", "A"), errors="coerce")
        dob_b = pd.to_datetime(self._side(pairs, "BirthDT_clean", "B"), errors="coerce")
        sa = dob_a.dt.strftime("%Y%m%d")
        sb = dob_b.dt.strftime("%Y%m%d")
        feat["sim_dob"] = pd.Series(
            [_lev_sim(x, y) for x, y in zip(sa, sb)], index=pairs.index, dtype="float",
        )

        # --- Phones: best cross-pair Jaro-Winkler ---
        feat["sim_phones"] = pd.Series(
            [_phones_best_jw(av, bv) for av, bv in zip(
                self._side(pairs, "Phones_set", "A"), self._side(pairs, "Phones_set", "B"))],
            index=pairs.index, dtype="float",
        )

        # Finalize dtypes + column order (exactly what the model was fit on).
        for c in CATEGORICAL_FEATURES:
            feat[c] = pd.Categorical(feat[c], categories=COMPARE_LEVELS)
        for c in NUMERIC_FEATURES:
            feat[c] = feat[c].astype("float")

        out = feat[FEATURE_COLS].copy()
        out.insert(0, "PATID_B", pairs["PATID_B"].to_numpy())
        out.insert(0, "PATID_A", pairs["PATID_A"].to_numpy())
        return out.reset_index(drop=True)

    # ── internals ────────────────────────────────────────────────────────────
    @staticmethod
    def _materialize_pairs(candidate_pairs: pd.DataFrame, df_clean: pd.DataFrame) -> pd.DataFrame:
        """Join cleaned attributes onto both sides of each pair (mirrors the
        notebook's reindex-and-suffix pattern). Missing cleaned columns are
        skipped; missing PATIDs yield NaN attribute rows."""
        clean = df_clean
        if "PATID" in clean.columns:
            clean = clean.set_index("PATID")
        clean = clean[[c for c in _CLEAN_COLS if c in clean.columns]]
        clean = clean[~clean.index.duplicated(keep="first")]
        idx_a = candidate_pairs["PATID_A"].astype(str).values
        idx_b = candidate_pairs["PATID_B"].astype(str).values
        a = clean.add_suffix("_A").reindex(idx_a).reset_index(drop=True)
        b = clean.add_suffix("_B").reindex(idx_b).reset_index(drop=True)
        base = candidate_pairs[["PATID_A", "PATID_B"]].reset_index(drop=True)
        return pd.concat([base, a, b], axis=1)

    @staticmethod
    def _side(pairs: pd.DataFrame, col: str, side: str) -> pd.Series:
        """`<col>_<side>` if present, else an all-NaN series (missing column)."""
        name = f"{col}_{side}"
        if name in pairs.columns:
            return pairs[name]
        return pd.Series(np.nan, index=pairs.index, dtype="object")

    def _sim_column(self, pairs: pd.DataFrame, col: str, fn) -> pd.Series:
        a = _norm_series(self._side(pairs, col, "A"))
        b = _norm_series(self._side(pairs, col, "B"))
        return pd.Series([fn(x, y) for x, y in zip(a, b)], index=pairs.index, dtype="float")

    def _pair_apply(self, pairs: pd.DataFrame, col: str, fn) -> pd.Series:
        return pd.Series(
            [fn(x, y) for x, y in zip(self._side(pairs, col, "A"), self._side(pairs, col, "B"))],
            index=pairs.index, dtype="float",
        )

    def _sound_cmp(self, pairs: pd.DataFrame, col: str) -> pd.Categorical:
        a = self._side(pairs, col, "A").map(_metaphone)
        b = self._side(pairs, col, "B").map(_metaphone)
        return _cmp_categorical(a, b)


class MatchProbabilityAdapter:
    """Wraps the notebook's fitted classifier so ``predict_proba(X)[:, 1]`` is
    ``P(confident match) = 1 - P(ambiguous)``.

    The inner model was trained with class 1 = *ambiguous*, so its
    ``predict_proba`` returns ``[P(match), P(ambiguous)]``. This adapter swaps
    the columns to ``[P(ambiguous), P(match)]`` so the pipeline (which reads
    column 1 as ``match_probability`` and maps high → ``auto_merge``) sees the
    match-scorer semantics. Also reorders input columns to the inner model's
    training feature order when known, so column ordering can't silently break
    predictions.
    """

    def __init__(self, model: Any):
        self.model = model

    def fit(self, X, y):  # noqa: D401 - present for MLModel Protocol completeness
        """Serve-only adapter; retraining is out of scope. Kept so the wrapper
        still satisfies the `MLModel` Protocol."""
        raise NotImplementedError(
            "MatchProbabilityAdapter is a serve-only wrapper; fit the inner "
            "model directly (see the v3 notebook) and wrap the fitted model."
        )

    def _align(self, X):
        names = getattr(self.model, "feature_name_", None)
        if names is not None and isinstance(X, pd.DataFrame) and set(names).issubset(X.columns):
            return X[list(names)]
        return X

    def predict_proba(self, X):
        proba = np.asarray(self.model.predict_proba(self._align(X)))
        if proba.ndim == 2 and proba.shape[1] == 2:
            return proba[:, ::-1]  # [P(match), P(amb)] -> [P(amb), P(match)]
        return proba
