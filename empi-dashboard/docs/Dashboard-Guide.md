# Entity Matching Dashboard — Functional Specification

Functional requirements for the **Entity Matching Dashboard**: the reviewer-facing
UI on top of the eMPI entity-resolution pipeline. It surfaces match results and
KPIs, lets authorized users review/merge/unmatch patient records, and explains
individual model decisions.

> **Scope:** UI/UX and behavioral requirements only. The matching logic itself is
> covered by [Blocking-Guide.md](../../empi-service/docs/Blocking-Guide.md) and
> [Deterministic-Rules-Guide.md](../../empi-service/docs/Deterministic-Rules-Guide.md)
> (backend docs, `empi-service/docs/`).
>
> **Branding:** colors, typography, and logo per
> [Alliance-Chicago-Branding.md](Alliance-Chicago-Branding.md).

## Information architecture

The application is a single-page app with **four top-level tabs** — three
reviewer-workflow tabs plus an operator-facing Admin tab. Switching tabs
never reloads or leaves the application. The **Model Explanation** view is a
sub-page reached by clicking a match inside the Patient Registry tab — not a
top-level tab.

```
┌─ Top nav: [AllianceChicago logo] [company name]  "Entity Matching Dashboard" ─┐
│                                                                               │
│   [ Dashboard ]   [ Review Queue ]   [ Patient Registry ]   [ Admin ]         │
└───────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      └─ click a match ─▶ Model Explanation (sub-page)
```

| Surface | Purpose |
|---------|---------|
| **Dashboard** (tab) | KPIs, summary metric cards, and visual charts **only** — no interactive workflows |
| **Review Queue** (tab) | Candidate-grain triage of pending review pairs only — two-panel layout (queue left, dynamic explanation right); see §5 |
| **Patient Registry** (tab, `/dataset` route) | The full patient list, one row per distinct patient — resolved clusters by default, plus (under its "All" toggle) singleton records still pending review, findable but not actionable here; see §3 |
| **Admin** (tab, `/admin` route) | Live ML decision-threshold tuning — operator configuration, not a reviewer workflow; see §7 |
| **Model Explanation** (sub-page) | Per-pair "why was this a match?" detail; reached from a Patient Registry row, returns to the reviewer's prior Patient Registry view (search/filter/page), not necessarily that one row |

Patient Registry and Review Queue split the old single Dataset tab's two
concerns cleanly: Review Queue is where a reviewer decides on a pending
candidate (merge or dismiss); Patient Registry is where anyone browses (and
searches) the patient list, settled or not. A still-pending singleton is
findable in both places at once — under Patient Registry's "All" toggle,
badged **Needs review**, and in the Review Queue's "Needs review" list —
but only decidable from the Review Queue. Once merged or dismissed, it
disappears from the Review Queue and shows up as a resolved entry in
Patient Registry's default view too.

---

## 1. Navigation & General Structure

| ID | Requirement |
|----|-------------|
| FR-1 | Top navigation bar displays the AllianceChicago logo, company name, and the title **"Entity Matching Dashboard"**. |
| FR-2 | Four tabs, in order: **Dashboard**, **Review Queue**, **Patient Registry**, **Admin** (§7 — operator configuration, not a reviewer workflow). |
| FR-3 | Users can switch between tabs without leaving the application (no full reload). |

---

## 2. Dashboard Page — KPIs & Visualizations Only

The Dashboard is **strictly** KPIs, summary metric cards, and charts. Merge/
dismiss workflows live on the Review Queue tab; browsing the resolved result
lives on the Patient Registry tab (§3).

### 2.1 Summary metrics & KPIs

| ID | Requirement |
|----|-------------|
| FR-4 | Total number of patient records loaded into the entity-resolution pipeline. |
| FR-5 | Total count of identified duplicate patient clusters. |
| FR-6 | Total number and/or percentage of matched patient records. |
| FR-7 | Number of patient records that currently require manual review. |
| FR-8 | Number of patient records classified as invalid. |
| FR-9 | Count of manually merged and manually unmerged patients. |
| FR-10 | Auto-match rate = automatic matches ÷ total candidate matches. |

### 2.2 Match status & performance visualizations

| ID | Requirement |
|----|-------------|
| FR-11 | Bar chart of patient counts across **Auto-match**, **Needs review**, **No match**. |
| FR-12 | Dynamically update the match-status chart based on the active dataset or selected filters. |
| FR-13 | Color-code categories — Auto-match: **green**, Needs review: **teal**, No match: **cyan**. Blues/greens only for these non-interactive status indicators; the four reviewer action buttons (Merge, Undo, Unmerge, Not a match) and system-error messages keep the original gold/red so they still read as actionable/alerting against the calmer status palette. |
| FR-14 | Display model performance metrics over time. **Limited to label-free metrics — auto-match rate and review rate.** Precision/recall are out of scope (no ground-truth labels available to compute them). |
| FR-15 | Visualize the auto-match-rate and review-rate trend as a line/bar chart with a metric-card group. |
| FR-16 | Show the model version or training run used to generate the current match results. |
| FR-17 | Show the exact match-confidence threshold values used to classify records. |

---

## 3. Patient Registry Page — Final, Resolved Patients Only

The Patient Registry (`/dataset` route) defaults to **resolved clusters
only** — automatically matched, manually merged, or standalone with nothing
pending. Its "2+ records" / "All" toggle (an implementation-level
`min_members` filter, not an `origin` one) is what actually keeps a
pending-review candidate out of the default view — every such candidate is
a singleton, so it's excluded there regardless of origin, and only surfaces
under "All", badged **Needs review**, so a data steward searching by
name/DOB/PATID can actually *find* a record even while it's still pending.
That's the one exception to "resolved only": finding it here is not
deciding it — merge/dismiss on a pending candidate only ever happens on the
Review Queue tab (§5). The only action available on an already-final
cluster here is **Unmerge**, for correcting one that turns out to be wrong.

### 3.1 Registry view

| ID | Requirement |
|----|-------------|
| FR-18 | Display the current master patient list (not raw source data). Under the default "2+ records" view, this is exactly the resolved list — automatically matched, manually merged, or standalone with nothing pending; a pending-review candidate is always a singleton so it's excluded here regardless. The "All" toggle additionally surfaces those singleton pending candidates, badged **Needs review**, so they're findable by search — deciding one (merge/dismiss) still only happens on the Review Queue tab. |
| FR-19 | Include every resolved status: automatically matched, manually merged, and standalone records with no pending candidates. |
| FR-20 | Render every patient as an expandable row, whether it folds in multiple entries or stands alone. |
| FR-21 | Expanding a row reveals its constituent entries — the raw records folded into this final patient — for traceability, not pending candidates. |
| FR-22 | The table displays exactly: **Patient Name**, **Masked SSN**, **Birthdate**, **# of entries**, **Last updated**. Master Patient ID is de-emphasized to a secondary line under the name (reviewers scan by identity, not internal ID) but always shown in full in unmerge confirmations. |
| FR-23 | Instantly refresh the table when a candidate elsewhere is merged into (or split out of) a record shown here. |
| FR-24 | Search & filter by: Master Patient ID, patient name, masked SSN, birthdate. |
| FR-25 | Preserve historical match decisions so changes to the registry are traceable over time. |

### 3.2 Correcting a final cluster

| ID | Requirement |
|----|-------------|
| FR-26 | Show an **Unmerge** control per entry within an expanded, multi-entry row. |
| FR-27 | Allow authorized users to split a wrongly-merged entry back out into its own standalone record. |
| FR-28 | Before any unmerge, show a confirmation naming exactly which record will be split out. |
| FR-29 | After an unmerge: append the split-out record to the registry as its own entry and update the original record's entry count. |

### 3.3 Merge audit log

| ID | Requirement |
|----|-------------|
| FR-30 | Record every manual or automatic merge action in an immutable audit log. |
| FR-31 | Each audit entry tracks: User ID, timestamp, patient IDs merged, previous & new match status, final master patient ID. |
| FR-52 | A merge or unmerge entry that hasn't already been reversed shows an **Undo** action; a reversed entry shows "Undone" instead. Undoing a merge unmerges every affected patid back out into its own standalone record; undoing an unmerge re-merges the patid into the mid it was split from. Undo is itself a new `merge`/`unmerge` audit-log entry (`undo_of` pointing at the entry it reverses) — the trail is append-only, nothing is deleted or rewritten. |
| FR-53 | Audit entries are never deleted, including reversed ones — undoing an action adds a new row rather than removing the original. |

---

## 4. Model Explanation (sub-page)

Reached by clicking a match inside a Patient Registry row's expanded dropdown;
back-navigation (FR-38) restores the reviewer's prior Patient Registry view
rather than returning to that specific row.

| ID | Requirement |
|----|-------------|
| FR-32 | Each patient-match item inside the dataset dropdown is clickable. |
| FR-33 | Clicking a match navigates to the **Model Explanation** page for that specific decision. |
| FR-34 | The page shows: patient IDs compared (de-emphasized, secondary line), the deterministic rule fired (if any) and its fixed confidence, the model/git version, and — when the pair was scored by the ML pipeline — a SHAP waterfall (FR-36/37a) from `GET /explanations/{model}/{patid_a}/{patid_b}`. A pair resolved purely by a deterministic rule (never reaching the gate/ML matcher) shows that rule as its explanation instead of a fabricated score. |
| FR-35 | ~~Display a human-readable explanation of why the pair/cluster was predicted a match.~~ **Removed** — the "Plain-language summary" card was dropped per reviewer feedback: it restated the structured feature-comparison table (FR-37) in prose without adding information, and added visual clutter to the page reviewers scan quickly. |
| FR-36 | Feature-level evaluations explicitly indicate which traits **increased** vs. **decreased** match probability. |
| FR-37 | Show how the score was calculated via a structured per-feature comparison table (feature, Patient A, Patient B, result) — deterministic-rule fields, always shown. |
| FR-37a | When the pair was scored by the non-match gate or ML matcher, additionally show an exact TreeSHAP waterfall (`ShapWaterfall.tsx`) of that model's feature contributions in log-odds, with the model's decision threshold marked — real, persisted per-run explanation data, never recomputed on the fly (see `empi-service/docs/Explanations-Guide.md`). |
| FR-38 | Back-navigation from the Model Explanation page uses browser history (`router.back()`), returning the reviewer to their exact prior Patient Registry state (search, filters, page) rather than a fresh view re-filtered to the one pair they came from. |

**Example feature-comparison table (FR-37):**

| Feature | Patient A | Patient B | Result |
|---------|-----------|-----------|--------|
| SSN | \*\*\*-\*\*-1234 | \*\*\*-\*\*-1234 | Exact match |
| Birthdate | 01/15/1980 | 01/15/1980 | Exact match |
| Last name | Smith | Smyth | High similarity |
| Phone | Same | Same | Exact match |
| ZIP code | 13902 | 13902 | Exact match |
| Email / First name | example value | example value | Similarity metric |

---

## 5. Review Queue (tab)

A dedicated, candidate-grain workflow surface — separate from the Dataset
tab's full-record browse table — for working through pending review
candidates one pair at a time. Added per reviewer feedback: the Dataset
tab's cluster-grain rows meant a cluster with several pending candidates
only appeared once, forcing expand-and-hunt instead of a queue a reviewer
could work through top to bottom.

| ID | Requirement |
|----|-------------|
| FR-42 | Two-panel layout: a candidate list on the left, a dynamic detail/explanation panel on the right. Selecting a candidate updates the right panel in place — no page navigation. |
| FR-43 | The left panel is **candidate-grain**: one row per pending pair (`patid_a`/`patid_b`), not per cluster — the same cluster can surface multiple rows if it has multiple pending candidates. |
| FR-44 | Left panel shows, per row: both patients' names, birthdate, masked SSN, a match-confidence percentage (rule confidence, falling back to the ML matcher's score when no rule fired — no separate rule/ML-tier tag), and a "+N in cluster" indicator when either side already belongs to a multi-member cluster. |
| FR-45 | Left panel supports: a **Needs review** / **Already reviewed** toggle, a confidence-range filter, and a name/DOB search. Default sort is confidence descending, with unconfirmed (no-rule) candidates sorted last. |
| FR-46 | A candidate counts as "reviewed" once it is either merged (both sides now share a master record) or explicitly dismissed (FR-49). |
| FR-47 | Right panel shows: both patients' names and de-emphasized IDs, a confidence/rule/predicted-class/cluster-context summary, a **pipeline trail** (FR-48), and the full field-by-field feature comparison (§4's FR-37 table). |
| FR-48 | The pipeline trail shows the value's path through the pipeline: Raw → Cleaned → Deterministic rule → ML signal (labeled **Non-match gate** or **ML matcher**, whichever actually scored the pair). All four stages show real data. The ML signal stage reads `GET /explanations/{model}/{patid_a}/{patid_b}` and shows the model's score, tier, and run id; a pair the deterministic rules resolved without reaching either model (or that the gate dropped before the ML matcher saw it) shows an honest "Not scored by the ML pipeline" state — never a fabricated score. The Fellegi-Sunter/Splink matcher (Stage 4 in the backend pipeline) is intentionally not part of this trail — it remains an audit-only candidate/feature generator kept in the backend for lineage, not a reviewer-facing decision signal (see `empi-service/docs/FS-Matcher-Production-Guide.md`). |
| FR-49 | A **"Not a match"** action lets a reviewer explicitly dismiss a candidate as a false positive. Recorded as an audit-log entry (`action=dismiss`); the candidate moves to "Already reviewed" and does not reappear in the default queue. |
| FR-50 | A **"Search manually for a different record"** action lets a reviewer propose a match blocking never surfaced as a candidate, sharing the same search/compare/merge flow as FR-27. |
| FR-51 | SSN fields support a reveal-in-place toggle (`SsnReveal.tsx`), sourced from `GET /records/{patid}/ssn-clean` — the pipeline-normalized SSN, a **separate** endpoint from the "View raw data" drawer's `GET /records/{patid}/raw`, preferred specifically so a reviewer sanity-checks against the value the matching engine actually trusted, not raw source-text noise. Every fetch from either endpoint is written to the backend audit log, but as two distinct actions (`view_ssn_clean` vs `view_raw`) for PHI-access accountability; these are compliance records, not reviewer decisions, so they're excluded from the Merge audit log table above (§3.3) and only queryable directly against the database. |

## 6. System Update & Sync Behavior

| ID | Requirement |
|----|-------------|
| FR-39 | If a new patient matches no existing cluster, instantly generate a standalone, unmatched master record. |
| FR-40 | On merge or reject/unmatch, instantly update the corresponding master records or isolate them into distinct master clusters. |
| FR-41 | Any dataset workflow action automatically refreshes the whole application — updating Dashboard metrics, resetting match-status charts, and refreshing dataset list views (now also the Review Queue). |

---

## 7. Admin (tab)

Operator configuration, not a reviewer workflow — no audit-log entry is written for a threshold change (§3.3's audit log is reviewer decisions on patient data; this is pipeline tuning). Explicitly built with **no authentication**; access control for this tab is out of scope for the dashboard build and is expected to be handled by whoever operationalizes the deployment.

| ID | Requirement |
|----|-------------|
| FR-54 | Show the three live ML decision thresholds — gate threshold, ML auto-merge threshold, FS review floor — with their current values and a one-line description of what each controls. |
| FR-55 | Let an operator edit and save all three thresholds; saving applies immediately to the running backend and persists across a restart (`empi-service/data/config/thresholds.json`), but only affects scoring done after the change — it never rewrites tiers a prior run already published. |
| FR-56 | Show a success or error toast on save. |

---

## Open items (REVIEW)

- **FR-10** — confirm the auto-match-rate denominator (total candidate matches vs. total records).
- **FR-14 / FR-15** — performance trend is limited to **auto-match rate** and **review rate** (both derivable from classification counts alone). Precision/recall are excluded because no ground-truth labels exist to compute them; revisit if gold labels become available. See [Deterministic-Rules-Guide.md](../../empi-service/docs/Deterministic-Rules-Guide.md).
- **FR-30 / FR-31 / FR-52 / FR-53** — confirm audit-log retention, immutability guarantees, and access policy (including for the `view_raw`/`view_ssn_clean` PHI-access entries, FR-51).
- **FR-54–56 (Admin tab)** — gated the same way as every other tab: Entra ID sign-in (Easy Auth) in front of the whole dashboard app in Azure (`terraform/auth.tf`). The `/admin/*` FastAPI routes themselves still carry no role-based authorization internally — their protection is the backend having no public ingress at all, not a per-route check. See `docs/Application-Architecture.md` §"Identity / auth" and `empi-service/src/api/routers/admin.py`'s own docstring.
