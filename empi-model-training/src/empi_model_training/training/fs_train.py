"""Fellegi-Sunter matcher training — Azure ML job entry point.

    uv run python -m empi_model_training.training.fs_train \
        --cleaned-index PATH --silver-labels PATH [--promote]

**Independent implementation.** This repo does not import or call into
`empi-service` (a deliberate project decision — see `CLAUDE.md`). This
script reproduces the comparison structure and supervised training
procedure documented in
`empi-service/docs/FS-Matcher-Production-Guide.md` closely enough that the
resulting model JSON is loadable by empi-service's own
`FSMatcher.load_settings()`/`score()` — but the code is maintained here
independently, not shared.

Training needs only two inputs: cleaned records and labeled pairs. Unlike
serving (which scores a large `non_matches` candidate pool), m/u estimation
never needs a real candidate-pairs table — Splink's
`estimate_m_from_pairwise_labels`/`estimate_u_using_random_sampling` work
against separately-registered tables. An empty placeholder table is
registered purely so the settings' blocking-rule SQL resolves.
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import mlflow
import pandas as pd

from empi_model_training.training import registry_utils

logger = logging.getLogger("empi_model_training.fs_train")

MODEL_NAME = "fs_matcher"

# ── Column names (must match empi-service's Data-Contract.md cleaned schema) ──
COL_PATID = "PATID"
COL_FIRST_NM = "FirstNM_clean"
COL_LAST_NM = "LastNM_clean"
COL_BIRTH_DT = "BirthDT_clean"
COL_DOB_STR = "_dob_str"
COL_SSN = "SSN_clean"
COL_EMAIL = "Email_clean"
COL_SEX = "SexAtBirthDSC_clean"
COL_PHONES_RAW = "Phones_set"
COL_PHONES_ARRAY = "Phones_array"
COL_ADDRESS1 = "AddressLine1_clean"

CANDIDATE_PAIRS_TABLE = "candidate_pairs"
LABELS_TABLE = "silver_labels"

CANDIDATE_PAIRS_BLOCKING_RULE = f"""
EXISTS (
    SELECT 1 FROM {CANDIDATE_PAIRS_TABLE} cp
    WHERE cp.PATID_A = l.{COL_PATID} AND cp.PATID_B = r.{COL_PATID}
)
"""

_NAME_DOB = (
    f"l.{COL_FIRST_NM} = r.{COL_FIRST_NM} AND l.{COL_LAST_NM} = r.{COL_LAST_NM} "
    f"AND l.{COL_DOB_STR} = r.{COL_DOB_STR}"
)
PRIOR_RULES: list[str] = [
    f"l.{COL_SSN} = r.{COL_SSN} AND l.{COL_DOB_STR} = r.{COL_DOB_STR} AND l.{COL_SSN} IS NOT NULL",
    f"{_NAME_DOB} AND l.{COL_EMAIL} = r.{COL_EMAIL} AND l.{COL_EMAIL} IS NOT NULL",
    f"{_NAME_DOB} AND len(list_intersect(l.{COL_PHONES_ARRAY}, r.{COL_PHONES_ARRAY})) >= 1",
    f"{_NAME_DOB} AND l.{COL_SEX} = r.{COL_SEX} AND l.{COL_SEX} IS NOT NULL",
    f"{_NAME_DOB} AND l.{COL_ADDRESS1} = r.{COL_ADDRESS1} AND l.{COL_ADDRESS1} IS NOT NULL",
]
DEFAULT_PRIOR_RECALL = 0.9


# ── Comparisons (7 two-level: null / exact / all-other) ─────────────────────
def _comparison_to_dict(creator: Any) -> dict[str, Any]:
    from splink import SettingsCreator

    settings = (
        SettingsCreator(
            link_type="dedupe_only", unique_id_column_name=COL_PATID, comparisons=[creator]
        )
        .get_settings("duckdb")
        .as_dict()
    )
    comparison: dict[str, Any] = settings["comparisons"][0]
    return comparison


def _phones_comparison() -> dict[str, Any]:
    return {
        "output_column_name": "Phones",
        "comparison_levels": [
            {
                "sql_condition": (
                    f"{COL_PHONES_ARRAY}_l IS NULL OR {COL_PHONES_ARRAY}_r IS NULL "
                    f"OR len({COL_PHONES_ARRAY}_l) = 0 OR len({COL_PHONES_ARRAY}_r) = 0"
                ),
                "label_for_charts": "Phones null/empty",
                "is_null_level": True,
            },
            {
                "sql_condition": (
                    f"len(list_intersect({COL_PHONES_ARRAY}_l, {COL_PHONES_ARRAY}_r)) >= 1"
                ),
                "label_for_charts": "Shared phone (intersection >= 1)",
            },
            {"sql_condition": "ELSE", "label_for_charts": "All other comparisons"},
        ],
    }


def build_comparisons(include_address: bool = True) -> list[dict[str, Any]]:
    import splink.comparison_library as cl

    comparisons = [
        _comparison_to_dict(cl.ExactMatch(COL_FIRST_NM).configure(term_frequency_adjustments=True)),
        _comparison_to_dict(cl.ExactMatch(COL_LAST_NM).configure(term_frequency_adjustments=True)),
        _comparison_to_dict(cl.ExactMatch(COL_DOB_STR)),
        _comparison_to_dict(cl.ExactMatch(COL_SSN)),
        _comparison_to_dict(cl.ExactMatch(COL_EMAIL).configure(term_frequency_adjustments=True)),
        _phones_comparison(),
    ]
    if include_address:
        comparisons.append(_comparison_to_dict(cl.ExactMatch(COL_ADDRESS1)))
    return comparisons


def build_settings(include_address: bool = True) -> dict[str, Any]:
    from splink import SettingsCreator

    settings = (
        SettingsCreator(
            link_type="dedupe_only",
            unique_id_column_name=COL_PATID,
            # A raw SQL string is a valid blocking rule per Splink's own docs
            # (resolved to a rule internally) even though the stub's
            # BlockingRuleCreator|dict union doesn't spell that out.
            blocking_rules_to_generate_predictions=[CANDIDATE_PAIRS_BLOCKING_RULE],  # type: ignore[list-item]
            comparisons=[],
            retain_intermediate_calculation_columns=True,
            retain_matching_columns=True,
        )
        .get_settings("duckdb")
        .as_dict()
    )
    settings["comparisons"] = build_comparisons(include_address)
    result: dict[str, Any] = settings
    return result


# ── Derived columns ──────────────────────────────────────────────────────────
_NA_TOKENS = {"nan", "none", "<na>", "nat", "null", "", "set()", "{}", "[]"}


def _parse_phone_set(value: Any) -> set[str]:
    """Cleaned phone cell -> set of phone strings. Handles a native
    set/list/tuple or the legacy stringified-repr form."""
    import ast

    if isinstance(value, (set, frozenset, list, tuple)):
        return {str(p).strip() for p in value if str(p).strip()}
    if not isinstance(value, str):
        try:
            if pd.isna(value):
                return set()
        except (TypeError, ValueError):
            pass
        return set()
    v = value.strip()
    if v.lower() in _NA_TOKENS:
        return set()
    try:
        parsed = ast.literal_eval(v)
        if isinstance(parsed, (set, list, tuple)):
            return {str(p).strip() for p in parsed if str(p).strip()}
    except (ValueError, SyntaxError, TypeError):
        pass
    cleaned = v.strip("{}[]").replace("'", "").replace('"', "")
    return {p.strip() for p in cleaned.split(",") if p.strip()}


#: Inert placeholder columns on the *main records* frame (not the pairs
#: table) so Splink's blocking-rule column harvesting resolves the
#: `cp.PATID_A`/`cp.PATID_B` references in CANDIDATE_PAIRS_BLOCKING_RULE
#: without erroring — Splink's settings validation checks these identifiers
#: against every registered table, including the main one. Without them,
#: Linker construction fails with "Table l does not have a column named
#: PATID_B". Never populated with real values; PATID_A/PATID_B always come
#: from the candidate_pairs table or from predict()'s min/max(PATID_l/r).
_PAIR_SHIM_COLUMNS = ("PATID_A", "PATID_B")


def prepare_model_input(df_clean: pd.DataFrame) -> pd.DataFrame:
    """Add the derived columns Splink reads (`_dob_str`, `Phones_array`) plus
    the pair-shim placeholder columns."""
    out = df_clean.copy()
    dob = pd.to_datetime(out[COL_BIRTH_DT], errors="coerce")
    out[COL_DOB_STR] = dob.dt.strftime("%Y-%m-%d")
    if COL_PHONES_RAW in out.columns:
        out[COL_PHONES_ARRAY] = out[COL_PHONES_RAW].apply(lambda v: sorted(_parse_phone_set(v)))
    else:
        out[COL_PHONES_ARRAY] = [[] for _ in range(len(out))]
    for shim in _PAIR_SHIM_COLUMNS:
        if shim not in out.columns:
            out[shim] = pd.NA
    return out


# ── Training ─────────────────────────────────────────────────────────────────
def _empty_candidate_pairs() -> pd.DataFrame:
    return pd.DataFrame(
        {"PATID_A": pd.Series([], dtype="object"), "PATID_B": pd.Series([], dtype="object")}
    )


def train_fs_model(
    df_model: pd.DataFrame,
    train_labels: pd.DataFrame,
    label_col: str,
    u_max_pairs: float,
    seed: int,
    include_address: bool = True,
) -> dict[str, Any]:
    """Estimate lambda (from PRIOR_RULES), m (supervised, from `train_labels`),
    and u (random sampling). Returns the trained Splink settings dict."""
    from splink import DuckDBAPI, Linker

    db_api = DuckDBAPI()
    con = db_api._con
    con.register(CANDIDATE_PAIRS_TABLE, _empty_candidate_pairs())

    linker = Linker(df_model, build_settings(include_address), db_api=db_api)

    logger.info(
        "Seeding match-prevalence prior from %d deterministic rule(s) (recall=%.2f)",
        len(PRIOR_RULES),
        DEFAULT_PRIOR_RECALL,
    )
    linker.training.estimate_probability_two_random_records_match(
        deterministic_matching_rules=list(PRIOR_RULES), recall=DEFAULT_PRIOR_RECALL
    )

    positives = train_labels[train_labels[label_col].astype(int) == 1]
    positives = positives.rename(
        columns={"PATID_A": f"{COL_PATID}_l", "PATID_B": f"{COL_PATID}_r"}
    )[[f"{COL_PATID}_l", f"{COL_PATID}_r"]]
    logger.info("Registering %d positive labels for m-training", len(positives))
    linker.table_management.register_table(positives, LABELS_TABLE, overwrite=True)
    linker.training.estimate_m_from_pairwise_labels(LABELS_TABLE)

    linker.training.estimate_u_using_random_sampling(max_pairs=u_max_pairs, seed=seed)

    trained_settings: dict[str, Any] = linker.misc.save_model_to_json()
    return trained_settings


def score_pairs(
    df_model: pd.DataFrame,
    pairs: pd.DataFrame,
    trained_settings: dict[str, Any],
) -> pd.DataFrame:
    """Score `pairs` (PATID_A/PATID_B) through a trained model — used only for
    held-out evaluation here, restricted to the records the pairs reference
    (evaluation-scale, not the O(records^2) serving path)."""
    from splink import DuckDBAPI, Linker

    referenced = set(pairs["PATID_A"].astype(str)) | set(pairs["PATID_B"].astype(str))
    subset = df_model[df_model[COL_PATID].astype(str).isin(referenced)]

    db_api = DuckDBAPI()
    db_api._con.register(CANDIDATE_PAIRS_TABLE, pairs[["PATID_A", "PATID_B"]])
    linker = Linker(subset, trained_settings, db_api=db_api)
    predictions = linker.inference.predict().as_pandas_dataframe()

    l_col, r_col = f"{COL_PATID}_l", f"{COL_PATID}_r"
    a = predictions[[l_col, r_col]].min(axis=1)
    b = predictions[[l_col, r_col]].max(axis=1)
    predictions.insert(0, "PATID_A", a)
    predictions.insert(1, "PATID_B", b)
    return predictions


def classify(
    df_predictions: pd.DataFrame, auto_merge_threshold: float, review_floor: float
) -> pd.DataFrame:
    out = df_predictions.copy()
    p = out["match_probability"]
    tier = pd.Series("no_match", index=out.index, dtype="object")
    tier = tier.mask(p >= review_floor, "human_review")
    tier = tier.mask(p >= auto_merge_threshold, "auto_merge")
    out["classification_tier"] = tier
    return out


def _binary_metrics(y_true: pd.Series, y_pred_pos: pd.Series) -> dict[str, float | int]:
    tp = int(((y_true == 1) & y_pred_pos).sum())
    fp = int(((y_true == 0) & y_pred_pos).sum())
    fn = int(((y_true == 1) & ~y_pred_pos).sum())
    tn = int(((y_true == 0) & ~y_pred_pos).sum())
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
    }


def _stratified_split(
    labels: pd.DataFrame, label_col: str, test_size: float, seed: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    test_parts = [
        grp.sample(frac=test_size, random_state=seed) for _, grp in labels.groupby(label_col)
    ]
    test = pd.concat(test_parts)
    train = labels.drop(test.index)
    return train.reset_index(drop=True), test.reset_index(drop=True)


def _load_silver_labels(path: Path, label_col: str) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={"PATID_A": str, "PATID_B": str})
    df[label_col] = df[label_col].astype(int)
    return df


def _canonicalize_pairs(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["PATID_A"], out["PATID_B"] = (
        out[["PATID_A", "PATID_B"]].min(axis=1),
        out[["PATID_A", "PATID_B"]].max(axis=1),
    )
    return out


def _merge_reviewer_labels(
    labels: pd.DataFrame, reviewer_labels_path: Path | None, label_col: str
) -> pd.DataFrame:
    """Fold in reviewer-confirmed labels (empi-service's
    scripts/export_reviewer_labels.py output -- always a `reviewer_label`
    column) alongside the primary label source. Reviewer labels win for any
    pair present in both: a human reviewer's confirmation in the dashboard
    outranks a deterministic-rule proxy. No-op if not given."""
    if reviewer_labels_path is None:
        return labels
    reviewer = _load_silver_labels(reviewer_labels_path, "reviewer_label")
    reviewer = reviewer.rename(columns={"reviewer_label": label_col})
    combined = pd.concat(
        [_canonicalize_pairs(labels), _canonicalize_pairs(reviewer)], ignore_index=True
    )
    return combined.drop_duplicates(subset=["PATID_A", "PATID_B"], keep="last").reset_index(
        drop=True
    )


@dataclass
class TrainResult:
    model_path: Path
    meta: dict[str, Any]


def run(args: argparse.Namespace) -> TrainResult:
    settings_module_dir = Path(args.model_dir)
    settings_module_dir.mkdir(parents=True, exist_ok=True)

    df_clean = pd.read_parquet(args.cleaned_index)
    df_clean[COL_PATID] = df_clean[COL_PATID].astype(str)
    df_model = prepare_model_input(df_clean)

    silver = _load_silver_labels(args.silver_labels, args.label_col)
    if args.reviewer_labels is not None:
        before = len(silver)
        silver = _merge_reviewer_labels(silver, args.reviewer_labels, args.label_col)
        logger.info(
            "merged reviewer labels from %s: %d -> %d rows (reviewer wins on overlap)",
            args.reviewer_labels, before, len(silver),
        )
    train_labels, test_labels = _stratified_split(
        silver, args.label_col, args.test_size, args.split_seed
    )
    logger.info(
        "records=%d silver=%d (pos=%d) -> train=%d test=%d",
        len(df_clean),
        len(silver),
        int((silver[args.label_col] == 1).sum()),
        len(train_labels),
        len(test_labels),
    )

    trained_settings = train_fs_model(
        df_model,
        train_labels,
        args.label_col,
        args.u_max_pairs,
        args.split_seed,
        args.include_address,
    )

    ts = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    model_path = settings_module_dir / f"fs_model_{ts}.json"
    model_path.write_text(json.dumps(trained_settings, indent=2, default=str))
    logger.info("Wrote trained model -> %s", model_path)

    predictions = score_pairs(df_model, test_labels[["PATID_A", "PATID_B"]], trained_settings)
    classified = classify(predictions, args.auto_merge_threshold, args.review_floor)
    canon = test_labels.copy()
    canon["PATID_A"], canon["PATID_B"] = (
        canon[["PATID_A", "PATID_B"]].min(axis=1),
        canon[["PATID_A", "PATID_B"]].max(axis=1),
    )
    merged = classified.merge(
        canon[["PATID_A", "PATID_B", args.label_col]], on=["PATID_A", "PATID_B"], how="inner"
    )
    y_true = merged[args.label_col]
    tier = merged["classification_tier"]
    test_metrics = {
        "n_test_pairs": int(len(merged)),
        "n_test_positives": int((y_true == 1).sum()),
        "metrics_auto_merge": _binary_metrics(y_true, tier == "auto_merge"),
        "metrics_merge_or_review": _binary_metrics(
            y_true, tier.isin(["auto_merge", "human_review"])
        ),
    }
    logger.info("Held-out auto_merge metrics: %s", test_metrics["metrics_auto_merge"])

    meta = {
        "model_name": MODEL_NAME,
        "model_file": model_path.name,
        "created_utc": datetime.now(UTC).isoformat(),
        "data_version": args.data_version or Path(args.cleaned_index).stem,
        "silver_labels_path": str(args.silver_labels),
        "seed": args.split_seed,
        "u_max_pairs": args.u_max_pairs,
        "thresholds": {"auto_merge": args.auto_merge_threshold, "review_floor": args.review_floor},
        "split": {
            "test_size": args.test_size,
            "n_train": int(len(train_labels)),
            "n_train_positives": int((train_labels[args.label_col] == 1).sum()),
            "n_test": int(len(test_labels)),
        },
        "test_metrics": test_metrics,
    }
    registry_utils.meta_path_for(model_path).write_text(json.dumps(meta, indent=2))
    logger.info("Wrote model meta -> %s", registry_utils.meta_path_for(model_path))

    if args.promote or args.force_promote:
        try:
            reason = registry_utils.promote(
                model_path, settings_module_dir, args.deploy_gate_margin, force=args.force_promote
            )
            logger.info("PROMOTED: %s", reason)
        except registry_utils.DeployGateError as exc:
            logger.error("NOT PROMOTED: %s", exc)
            raise

    return TrainResult(model_path=model_path, meta=meta)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--cleaned-index", type=Path, required=True, help="Cleaned records parquet.")
    p.add_argument(
        "--silver-labels",
        type=Path,
        required=True,
        help="Silver-labels CSV (PATID_A, PATID_B, label).",
    )
    p.add_argument("--label-col", default="silver_label")
    p.add_argument(
        "--reviewer-labels",
        type=Path,
        default=None,
        help="Optional reviewer-confirmed labels CSV (PATID_A, PATID_B, reviewer_label) from "
        "empi-service's scripts/export_reviewer_labels.py. Wins over --silver-labels for any "
        "pair present in both.",
    )
    p.add_argument("--test-size", type=float, default=0.2)
    p.add_argument("--split-seed", type=int, default=42)
    p.add_argument("--data-version", default=None)
    p.add_argument("--u-max-pairs", type=float, default=1e6)
    p.add_argument("--include-address", action="store_true", default=True)
    p.add_argument("--auto-merge-threshold", type=float, default=0.95)
    p.add_argument("--review-floor", type=float, default=0.40)
    p.add_argument("--deploy-gate-margin", type=float, default=0.02)
    p.add_argument("--model-dir", type=Path, default=Path("models/fs"))
    p.add_argument("--promote", action="store_true")
    p.add_argument("--force-promote", action="store_true")
    return p.parse_args(argv)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    )
    args = parse_args()

    mlflow.set_experiment("empi-fs-matcher")
    with mlflow.start_run(run_name="fs-matcher-train") as run_ctx:
        mlflow.log_params(
            {
                "cleaned_index": str(args.cleaned_index),
                "u_max_pairs": args.u_max_pairs,
                "split_seed": args.split_seed,
                "test_size": args.test_size,
                "auto_merge_threshold": args.auto_merge_threshold,
                "review_floor": args.review_floor,
            }
        )
        result = run(args)
        auto_merge = result.meta["test_metrics"]["metrics_auto_merge"]
        mlflow.log_metrics(
            {f"auto_merge_{k}": v for k, v in auto_merge.items() if isinstance(v, int | float)}
        )
        mlflow.log_artifact(str(result.model_path), artifact_path="model")
        mlflow.log_artifact(
            str(registry_utils.meta_path_for(result.model_path)), artifact_path="model"
        )

        if mlflow.get_tracking_uri().startswith("azureml"):
            mlflow.register_model(f"runs:/{run_ctx.info.run_id}/model", "empi-fs-matcher")
        else:
            logger.info(
                "Not running under Azure ML tracking -- skipping model registry registration."
            )


if __name__ == "__main__":
    main()
