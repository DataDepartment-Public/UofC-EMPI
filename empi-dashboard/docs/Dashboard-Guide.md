# Entity Matching Dashboard — Functional Specification

Functional requirements for the **Entity Matching Dashboard**: the reviewer-facing
UI on top of the eMPI entity-resolution pipeline. It surfaces match results and
KPIs, lets authorized users review/merge/unmatch patient records, and explains
individual model decisions.

> **Scope:** UI/UX and behavioral requirements only. The matching logic itself is
> covered by [Blocking-Guide.md](Blocking-Guide.md) and
> [Deterministic-Rules-Guide.md](Deterministic-Rules-Guide.md).
>
> **Branding:** colors, typography, and logo per
> [Alliance-Chicago-Branding.md](Alliance-Chicago-Branding.md).
>
> **Demo:** an interactive mock implementation lives at
> [demo/dashboard-demo.html](../demo/dashboard-demo.html).

## Information architecture

The application is a single-page app with **two top-level tabs**. Switching tabs
never reloads or leaves the application. The **Model Explanation** view is a
sub-page reached by clicking a match inside the Dataset tab — not a top-level tab.

```
┌─ Top nav: [AllianceChicago logo] [company name]  "Entity Matching Dashboard" ─┐
│                                                                               │
│   [ Dashboard ]   [ Dataset ]                                                 │
└───────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      └─ click a match ─▶ Model Explanation (sub-page)
```

| Surface | Purpose |
|---------|---------|
| **Dashboard** (tab) | KPIs, summary metric cards, and visual charts **only** — no interactive workflows |
| **Dataset** (tab) | Full matched dataset with embedded expandable-row review, merge, and unmatch workflows |
| **Model Explanation** (sub-page) | Per-pair "why was this a match?" detail; reached from a Dataset dropdown, returns to it |

---

## 1. Navigation & General Structure

| ID | Requirement |
|----|-------------|
| FR-1 | Top navigation bar displays the AllianceChicago logo, company name, and the title **"Entity Matching Dashboard"**. |
| FR-2 | Two main tabs: **Dashboard** and **Dataset**. |
| FR-3 | Users can switch between Dashboard and Dataset without leaving the application (no full reload). |

---

## 2. Dashboard Page — KPIs & Visualizations Only

The Dashboard is **strictly** KPIs, summary metric cards, and charts. All
interactive workflows, dropdown lists, and merge controls live on the Dataset page.

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

## 3. Dataset Page — Matched Dataset & Matching Workflow

The Dataset tab houses the full matched dataset and embeds the interactive review,
matching, and merge workflows. Every patient record — clustered or standalone — is
accessible here via an expandable-row dropdown interface.

### 3.1 Dataset view & dropdown integration

| ID | Requirement |
|----|-------------|
| FR-18 | Display the full matched dataset of current master patient records (not raw source data). |
| FR-19 | Include records of all resolution statuses: automatically matched, manually approved, manually merged, left unmatched. |
| FR-20 | Render every patient record as an expandable row/dropdown, whether part of a duplicate cluster or standalone. |
| FR-21 | Expanding a row reveals every underlying record flagged as a potential match / duplicate candidate. |
| FR-22 | The master table and expanded views display: Patient ID / Master Patient ID, patient name, masked SSN, birthdate, match-confidence score, match status / row origin (automatic, manual review, manual merge, no-match), key matching features used, last-updated timestamp. |
| FR-23 | Append a new row whenever a new patient is processed, and instantly refresh the table when a match is created, a merge is approved, or a user unmatches records. |
| FR-24 | Search & filter by: Master Patient ID, patient name, masked SSN, birthdate, match status, merge status, last-updated date. |
| FR-25 | Preserve historical match decisions so changes to the dataset are traceable over time. |

### 3.2 Merge & action workflow

| ID | Requirement |
|----|-------------|
| FR-26 | Show a **Merge** button for suggested clusters and a **remove/unmerge** control within the dropdown panel. |
| FR-27 | Allow authorized users to merge two or more records directly from the dropdown into a single master record. |
| FR-28 | Before any permanent merge, show a confirmation naming exactly which records will be combined. |
| FR-29 | After a merge: update the cluster status, append the new master record to the dataset, and update/remove the prior unmerged duplicate suggestion. |

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
| FR-34 | The page shows: patient IDs compared, overall match probability and final predicted class, match threshold and model version used, prediction date/time, and a Fellegi waterfall graph (if feasible). |
| FR-35 | Display a human-readable explanation of why the pair/cluster was predicted a match. |
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

## 5. System Update & Sync Behavior

| ID | Requirement |
|----|-------------|
| FR-39 | If a new patient matches no existing cluster, instantly generate a standalone, unmatched master record. |
| FR-40 | On merge or reject/unmatch, instantly update the corresponding master records or isolate them into distinct master clusters. |
| FR-41 | Any dataset workflow action automatically refreshes the whole application — updating Dashboard metrics, resetting match-status charts, and refreshing dataset list views. |

---

## Open items (REVIEW)

- **FR-10** — confirm the auto-match-rate denominator (total candidate matches vs. total records).
- **FR-14 / FR-15** — performance trend is limited to **auto-match rate** and **review rate** (both derivable from classification counts alone). Precision/recall are excluded because no ground-truth labels exist to compute them; revisit if gold labels become available. See [Deterministic-Rules-Guide.md](Deterministic-Rules-Guide.md).
- **FR-30 / FR-31** — confirm audit-log retention, immutability guarantees, and access policy.
- **FR-34** — confirm feasibility of the Fellegi–Sunter waterfall graph (probabilistic/Splink model still in research).
