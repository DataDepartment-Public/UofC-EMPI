"""Typo-tolerant q-gram blocking pass for the eMPI stacked blocker.

The 8-block scheme (`src/preprocessing/blocking.py`) anchors every name+DOB block
on the **last name**, so a last-name typo that breaks both Soundex and Double
Metaphone escapes all of them at once (the documented 212-pair residual). This
pass closes that gap: it scores a **character-n-gram TF-IDF cosine** on the full
name, restricted to within a cheap coarse block (exact birth date by default), and
emits any pair above a similarity threshold. Restricting the O(g²) cosine to small
within-DOB groups keeps it sub-quadratic and production-viable.

It is the second leg of the stacked blocker — the output is unioned with the
8-block candidate set and then pruned by meta-blocking
(`src/preprocessing/meta_blocking.py`). See `stacked_blocking.run_stacked_blocking`
and `docs/Blocking-Guide.md`.

Validated recommended config (run `real_20260620`): `block_kind="fulldob"`, char
(2,4)-grams, cosine ≥ 0.30 → recovers the residual the 8-block scheme misses at
~0.8% extra candidates. The threshold is pending gold-label confirmation.

OUTPUT SCHEMA (matches the blocking stage):
    PATID_A, PATID_B  — canonical order (PATID_A < PATID_B)
    source_blocks     — the pass label (default "QGRAM")
    n_blocks          — 1 (this pass is a single signal)
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from src.config import settings
from src.preprocessing.blocking import _filter_valid_records

logger = logging.getLogger(__name__)

QGRAM_SOURCE_LABEL = "QGRAM"

_COL_PATID = "PATID"
_COL_FIRST = "FirstNM_clean"
_COL_LAST = "LastNM_clean"
_COL_BIRTH = "BirthDT_clean"

_PAIR_COLUMNS = ["PATID_A", "PATID_B", "source_blocks", "n_blocks"]


def _name_identity(df: pd.DataFrame) -> np.ndarray:
    """Lower-cased "first last" identity string per record (blank where missing)."""
    first = df.get(_COL_FIRST, pd.Series("", index=df.index)).fillna("").astype(str)
    last = df.get(_COL_LAST, pd.Series("", index=df.index)).fillna("").astype(str)
    return (first + " " + last).str.lower().str.strip().to_numpy()


def _coarse_block_keys(df: pd.DataFrame, block_kind: str) -> np.ndarray:
    """The coarse block each record joins for the within-block cosine.

    `fulldob` groups by exact birth date (small, precise groups); `birthyear`
    groups by year (broader recall, larger groups). Records with a null key are
    excluded (returned key is an empty string).
    """
    dob = pd.to_datetime(df.get(_COL_BIRTH), errors="coerce")
    if block_kind == "fulldob":
        keys = dob.dt.strftime("%Y-%m-%d")
    elif block_kind == "birthyear":
        keys = dob.dt.year.astype("Int64").astype(str)
    else:
        raise ValueError(
            f"Unknown qgram_block_kind {block_kind!r} (use 'fulldob' or 'birthyear')."
        )
    return keys.fillna("").replace({"<NA>": "", "nan": "", "NaT": ""}).to_numpy()


def _empty_pairs() -> pd.DataFrame:
    return pd.DataFrame(columns=_PAIR_COLUMNS)


def run_qgram_blocking(
    df_clean: pd.DataFrame,
    *,
    block_kind: str | None = None,
    threshold: float | None = None,
    ngram_range: tuple[int, int] | None = None,
    min_df: int | None = None,
    source_label: str = QGRAM_SOURCE_LABEL,
) -> pd.DataFrame:
    """Emit typo-tolerant candidate pairs via within-block char-n-gram cosine.

    Parameters
    ----------
    df_clean : pd.DataFrame
        Cleaned dataset (PATID + `*_clean` columns). `valid_record == False`
        rows are filtered out first, mirroring the 8-block scheme.
    block_kind, threshold, ngram_range, min_df : optional
        Override the configured defaults (`settings.qgram_*`). `ngram_range`
        defaults to `(qgram_ngram_min, qgram_ngram_max)`.
    source_label : str
        Value written to `source_blocks` for these pairs (default "QGRAM").

    Returns
    -------
    pd.DataFrame
        Candidate pairs in the standard blocking schema (canonical order).
    """
    block_kind = block_kind or settings.qgram_block_kind
    threshold = settings.qgram_threshold if threshold is None else threshold
    min_df = settings.qgram_min_df if min_df is None else min_df
    if ngram_range is None:
        ngram_range = (settings.qgram_ngram_min, settings.qgram_ngram_max)

    records = _filter_valid_records(df_clean)
    if len(records) < 2 or _COL_PATID not in records.columns:
        return _empty_pairs()

    names = _name_identity(records)
    patids = records[_COL_PATID].to_numpy()
    keys = _coarse_block_keys(records, block_kind)

    vectorizer = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=ngram_range,
        min_df=min_df,
        lowercase=True,
        dtype=np.float32,
    )
    try:
        matrix = vectorizer.fit_transform(names)  # L2-normalized → dot == cosine
    except ValueError:
        # Empty vocabulary (e.g. every n-gram below min_df on a tiny input).
        logger.warning("q-gram blocking: empty vocabulary; no candidate pairs.")
        return _empty_pairs()

    groups: dict[str, list[int]] = {}
    for i, key in enumerate(keys):
        if key:
            groups.setdefault(key, []).append(i)

    rows_a: list[np.ndarray] = []
    rows_b: list[np.ndarray] = []
    for idx in groups.values():
        if len(idx) < 2:
            continue
        sub = matrix[idx]
        sim = (sub @ sub.T).toarray()
        iu, ju = np.triu_indices(len(idx), k=1)
        keep = sim[iu, ju] >= threshold
        if not keep.any():
            continue
        members = np.asarray(idx)
        rows_a.append(members[iu[keep]])
        rows_b.append(members[ju[keep]])

    if not rows_a:
        logger.info(
            "q-gram blocking (%s, >=%.2f): 0 candidate pairs from %d records.",
            block_kind, threshold, len(records),
        )
        return _empty_pairs()

    left = patids[np.concatenate(rows_a)]
    right = patids[np.concatenate(rows_b)]
    # Canonical ordering PATID_A < PATID_B; drop any degenerate self-pairs.
    a = np.minimum(left, right)
    b = np.maximum(left, right)
    pairs = pd.DataFrame({"PATID_A": a, "PATID_B": b})
    pairs = pairs[pairs["PATID_A"] != pairs["PATID_B"]].drop_duplicates()
    pairs["source_blocks"] = source_label
    pairs["n_blocks"] = 1

    logger.info(
        "q-gram blocking (%s, >=%.2f): %d candidate pairs from %d records.",
        block_kind, threshold, len(pairs), len(records),
    )
    return pairs[_PAIR_COLUMNS].reset_index(drop=True)
