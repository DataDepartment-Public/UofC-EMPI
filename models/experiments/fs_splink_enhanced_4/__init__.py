"""fs_splink_enhanced_4 — structural conflict-vs-missing FS experiment.

Enhanced_4 is enhanced_3 plus explicit multi-level conflict-vs-missing
comparisons: for every identity field the comparison distinguishes

    null (no evidence)  →  exact / near-match (positive evidence)
                        →  explicit hard-conflict (negative evidence)
                        →  else (weak residual)

so a conflicting SSN, DOB year off by more than 1, or clearly different first
name all contribute negative Bayes factors rather than falling through to the
weakly-informative "else" level that enhanced_3 uses throughout.

Corroboration (requiring agreement on ≥2 independent fields before auto-merge)
and the Household_discount composite handle the precision side; m stays purely
supervised from the real-cohort silver labels exactly as enhanced_3.
"""
