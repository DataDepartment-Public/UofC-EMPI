"""
tests/regression/test_fellegi_sunter_baseline.py

Regression / snapshot-style checks for the Fellegi-Sunter Splink baseline
(implementation plan Section 10.3, the m/u weight-inversion guardrail from
Section 6.4).

These guard against silent training regressions: if a future change to
build_settings(), the EM training sessions, or the synthetic dataset causes
the model's learned m/u parameters to become degenerate or to invert (m < u
on a field that should be discriminating), these tests catch it even though
the end-to-end classification (tested in tests/integration) might still
happen to "pass" by coincidence.

As real-data calibration progresses (plan Section 8), add snapshot-style
assertions here for specific synthetic pairs' expected score ranges so drift
is caught early.
"""

from models.experiments.fs_splink_baseline import fellegi_sunter_baseline as fs


def test_trained_parameters_are_sane(fs_df_clean, fs_candidate_pairs_path):
    """
    After training, m/u parameters must be finite and the discriminating
    SSN-exact level must satisfy m >= u (an exact SSN match is more likely
    among true matches than among random pairs). A violation would indicate a
    weight inversion (plan Section 6.4).
    """
    # Single-CPU sandbox cap (see FS_U_MAX_PAIRS_SANDBOX in tests/conftest.py
    # and NEW ISSUE A in fellegi_sunter_baseline.py); VM default is 1e6.
    _, diag = fs.run_fs_baseline(
        fs_candidate_pairs_path, fs_df_clean,
        u_max_pairs=1e4, return_diagnostics=True,
    )
    records = diag.get("parameter_records", [])
    assert records, "diagnostics should expose per-level parameter records"

    # All m and u probabilities are finite and within [0, 1].
    for rec in records:
        for key in ("m_probability", "u_probability"):
            val = rec.get(key)
            if val is not None:
                assert 0.0 <= val <= 1.0, f"{key}={val} out of range in {rec}"

    # Locate the SSN exact-match level and assert m >= u.
    ssn_exact = [
        r for r in records
        if r.get("comparison_name") == "SSN"
        and r.get("m_probability") is not None
        and r.get("u_probability") is not None
        and "exact" in str(r.get("label_for_charts", "")).lower()
    ]
    if ssn_exact:  # present once SSN comparison trained
        rec = ssn_exact[0]
        assert rec["m_probability"] >= rec["u_probability"], (
            f"SSN exact-match weight inverted: m={rec['m_probability']} "
            f"u={rec['u_probability']}"
        )


def test_no_weight_inversions_on_high_precision_levels(
    fs_df_clean, fs_candidate_pairs_path,
):
    """
    Plan R1 guardrail: the exact-match level of every comparison that has one
    should satisfy m >= u after training. This is the class of bug that the
    original Phase A diagnostic surfaced (FirstNM JW>=0.7, DOB within-1-year,
    Email JW>0.88 username, ZIP first-3-digit prefix all had m < u). If a
    future settings change re-introduces a non-discriminating exact level,
    this test catches it.
    """
    _, diag = fs.run_fs_baseline(
        fs_candidate_pairs_path, fs_df_clean,
        u_max_pairs=1e4, return_diagnostics=True,
    )
    records = diag.get("parameter_records", []) or []
    inversions: list[str] = []
    for rec in records:
        label = str(rec.get("label_for_charts", "")).lower()
        m = rec.get("m_probability")
        u = rec.get("u_probability")
        if not isinstance(m, (int, float)) or not isinstance(u, (int, float)):
            continue
        if "exact" in label and m < u:
            inversions.append(
                f"{rec.get('comparison_name')}/{rec.get('label_for_charts')}: "
                f"m={m:.4g} u={u:.4g}"
            )
    assert not inversions, (
        "Weight inversion on exact-match level(s): " + "; ".join(inversions)
    )