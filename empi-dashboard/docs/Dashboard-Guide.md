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

The application is a single-page app with **three top-level tabs**. Switching
tabs never reloads or leaves the application. The **Model Explanation** view
is a sub-page reached by clicking a match inside the Dataset tab — not a
top-level tab.

```
┌─ Top nav: [AllianceChicago logo] [company name]  "Entity Matching Dashboard" ─┐
│                                                                               │
│   [ Dashboard ]   [ Review Queue ]   [ Patient Registry ]                     │
└───────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      └─ click a match ─▶ Model Explanation (sub-page)
```

| Surface | Purpose |
|---------|---------|
| **Dashboard** (tab) | KPIs, summary metric cards, and visual charts **only** — no interactive workflows |
| **Review Queue** (tab) | Candidate-grain triage of pending review pairs only — two-panel layout (queue left, dynamic explanation right); see §5 |
| **Patient Registry** (tab, `/dataset` route) | The final, resolved patient list **only** — one row per distinct patient, no in-progress candidates; see §3 |
| **Model Explanation** (sub-page) | Per-pair "why was this a match?" detail; reached from a Patient Registry row, returns to it |

Patient Registry and Review Queue split the old single Dataset tab's two
concerns cleanly: Review Queue is where a reviewer decides on a pending
candidate (merge or dismiss); Patient Registry is where anyone browses the
settled result. A record only appears in one place at a time — once merged
or dismissed, it disappears from the Review Queue and (if applicable) shows
up as a resolved entry in Patient Registry.

---

## 1. Navigation & General Structure

| ID | Requirement |
|----|-------------|
| FR-1 | Top navigation bar displays the AllianceChicago logo, company name, and the title **"Entity Matching Dashboard"**. |
| FR-2 | Three main tabs, in order: **Dashboard**, **Review Queue**, **Patient Registry**. |
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
| FR-13 | Color-code categories — Auto-match: **green**, Needs review: **yellow/orange**, No match: **red**. |
| FR-14 | Display model performance metrics over time. **Limited to label-free metrics — auto-match rate and review rate.** Precision/recall are out of scope (no ground-truth labels available to compute them). |
| FR-15 | Visualize the auto-match-rate and review-rate trend as a line/bar chart with a metric-card group. |
| FR-16 | Show the model version or training run used to generate the current match results. |
| FR-17 | Show the exact match-confidence threshold values used to classify records. |

---

## 3. Patient Registry Page — Final, Resolved Patients Only

The Patient Registry (`/dataset` route) shows **only resolved clusters** —
automatically matched, manually merged, or standalone with nothing pending
(`origin` != `review`). Anything still awaiting a match decision lives
entirely on the Review Queue tab (§5); a data steward browsing the registry
should never see in-progress work. Merge/dismiss actions on pending
candidates happen on Review Queue — the only action available here is
**Unmerge**, for correcting an already-final cluster that turns out to be
wrong.

### 3.1 Registry view

| ID | Requirement |
|----|-------------|
| FR-18 | Display the current, resolved master patient list (not raw source data) — excludes anything still in the Review Queue. |
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

---

## 4. Model Explanation (sub-page)

Reached by clicking a match inside a Dataset dropdown; returns to the originating
row in the Dataset tab.

| ID | Requirement |
|----|-------------|
| FR-32 | Each patient-match item inside the dataset dropdown is clickable. |
| FR-33 | Clicking a match navigates to the **Model Explanation** page for that specific decision. |
| FR-34 | The page shows: patient IDs compared (de-emphasized, secondary line), overall match probability and final predicted class, match threshold and model version used, prediction date/time, and a Fellegi waterfall graph (if feasible). |
| FR-35 | ~~Display a human-readable explanation of why the pair/cluster was predicted a match.~~ **Removed** — the "Plain-language summary" card was dropped per reviewer feedback: it restated the structured feature-comparison table (FR-37) in prose without adding information, and added visual clutter to the page reviewers scan quickly. |
| FR-36 | Feature-level evaluations explicitly indicate which traits **increased** vs. **decreased** match probability. |
| FR-37 | Show how the score was calculated via a structured per-feature comparison table (feature, Patient A, Patient B, result). |
| FR-38 | Provide back-navigation from the Model Explanation page to the selected cluster in the Dataset tab. |

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
| FR-44 | Left panel shows, per row: both patients' names, birthdate, masked SSN, match-confidence badge, rule tag (or "Blocking only — no rule" if unconfirmed), and a "+N in cluster" indicator when either side already belongs to a multi-member cluster. |
| FR-45 | Left panel supports: a **Needs review** / **Already reviewed** toggle, a confidence-range filter, and a name/DOB search. Default sort is confidence descending, with unconfirmed (no-rule) candidates sorted last. |
| FR-46 | A candidate counts as "reviewed" once it is either merged (both sides now share a master record) or explicitly dismissed (FR-49). |
| FR-47 | Right panel shows: both patients' names and de-emphasized IDs, a confidence/rule/predicted-class/cluster-context summary, a **pipeline trail** (FR-48), the full field-by-field feature comparison (§4's FR-37 table), and the FS matcher signal card (§4 scope note). |
| FR-48 | The pipeline trail shows the value's path through the pipeline: Raw → Cleaned → Deterministic rule → FS matcher signal → ML model. The first four stages show real data; the ML/GBT stage is a labeled "not yet in production" placeholder — no fabricated score, since no such model is deployed (see `empi-service/docs/FS-Matcher-Production-Guide.md`). |
| FR-49 | A **"Not a match"** action lets a reviewer explicitly dismiss a candidate as a false positive. Recorded as an audit-log entry (`action=dismiss`); the candidate moves to "Already reviewed" and does not reappear in the default queue. |
| FR-50 | A **"Search manually for a different record"** action lets a reviewer propose a match blocking never surfaced as a candidate, sharing the same search/compare/merge flow as FR-27. |
| FR-51 | SSN fields support a reveal-in-place toggle, sourced from the same raw-record data the "View raw data" drawer already fetches — no separate PII exposure surface. |

## 6. System Update & Sync Behavior

| ID | Requirement |
|----|-------------|
| FR-39 | If a new patient matches no existing cluster, instantly generate a standalone, unmatched master record. |
| FR-40 | On merge or reject/unmatch, instantly update the corresponding master records or isolate them into distinct master clusters. |
| FR-41 | Any dataset workflow action automatically refreshes the whole application — updating Dashboard metrics, resetting match-status charts, and refreshing dataset list views (now also the Review Queue). |

---

## Open items (REVIEW)

- **FR-10** — confirm the auto-match-rate denominator (total candidate matches vs. total records).
- **FR-14 / FR-15** — performance trend is limited to **auto-match rate** and **review rate** (both derivable from classification counts alone). Precision/recall are excluded because no ground-truth labels exist to compute them; revisit if gold labels become available. See [Deterministic-Rules-Guide.md](../../empi-service/docs/Deterministic-Rules-Guide.md).
- **FR-30 / FR-31** — confirm audit-log retention, immutability guarantees, and access policy.
- **FR-34** — the Fellegi-Sunter/Splink matcher is built and in production (Stage 4, `empi-service/docs/FS-Matcher-Production-Guide.md`), but deliberately runs as an audit-only candidate/feature generator for a future GBT, not as a scored decision on any reviewed pair — so the waterfall graph this FR describes has no real data to show yet. The Model Explanation page and the Review Queue's pipeline trail (FR-48) now do surface the FS match probability/tier where it exists (candidates scored via incremental scoring; null for full-batch-published candidates), always labeled "audit-only — does not decide matches," never presented as the page's predicted class.
- **FR-48's ML/GBT stage** — genuinely not in production; the pipeline trail renders it as a placeholder only. Revisit once a GBT model is trained and deployed downstream of the FS matcher.
