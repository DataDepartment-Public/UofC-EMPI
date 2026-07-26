# eMPI — Infra Hardening & MLOps To-Do

Tracking doc for the post-review hardening pass so this survives across
sessions. Originating context: an architecture review done ahead of a client
presentation (see [`docs/Architecture-Diagram.md`](docs/Architecture-Diagram.md))
surfaced a set of infrastructure gaps and an MLOps maturity question. This
file is the backlog for closing them.

## Track C — Client presentation refresh (2026-07-25)

Tracks A and B are done (see below) but the client-facing artifacts that
describe them are not — both predate this session's infra/MLOps work
entirely and currently show the *original* topology (public everything, no
auth, no Azure ML). Started this track specifically so it survives a
session break.

### C1 — Redraw docs/Architecture-Diagram.md
**Status: done.** Restructured into 4 sections matching the deck outline
(Overview, Networking, MLOps/AML, CI/CD) with 5 Mermaid diagrams total.
Verified two ways: rendered every diagram with `@mermaid-js/mermaid-cli`
(no syntax errors) and visually inspected the PNG output for legibility —
not just "didn't crash."
Update the Mermaid diagrams to reflect the actual current state:
- VNet + private endpoints + no public backend ingress (A4)
- Entra ID SSO on the dashboard (A5)
- Azure ML workspace (managed network isolation) + training/CI flow (B1-B4)
- Observability (A1) and Postgres backup/HA (A2) worth a mention
Root `README.md`'s inline ASCII diagram is also stale (same root cause) —
lower priority than the dedicated diagram doc, but flag if touching this
area again.

### C2 — Branded PPTX architecture deck
**Status: done.** `docs/eMPI-Architecture-Deck.pptx` (15 slides: title,
agenda, then divider + 1-3 content slides per section for all 4 sections,
closing — the Overview section grew a 3rd slide, see B5). Build script +
inputs kept in `docs/deck-build/` (see its README) so this is
editable/regeneratable in a future session, not just a binary handed over.
- Branding verified directly from the template's XML — not guessed. Real
  values used: slide size 20×11.25in; brand colors `0060A0`/`18A2CC`/
  `009090`/`A8BA4E` (confirmed as a genuine 4-color rotation used across
  4 different template slides, and it visibly matches the AllianceChicago
  logo's own gradient); navy `0A3A5C` for cover-style backgrounds; fonts
  Outfit (headers) + Source Sans 3 (body); the actual logo image extracted
  from the template's media folder.
- **The 4 architecture diagrams (Overview, Networking, MLOps, CI/CD) are
  native pptx shapes, not embedded images.** Originally built by rendering
  Mermaid to PNG and picture-embedding it; rebuilt per explicit feedback
  ("id rather you natively build the diagrams with python-pptx") so
  they're editable in PowerPoint. Every service box now names the real
  Azure service explicitly (e.g. "Azure Database for PostgreSQL", not a
  generic "database" icon) and pairs it with a bold plain-language role
  line, per the follow-up ask that an executive audience needs to
  understand the diagrams too, not just engineers. Toolkit
  (`svc_box`/`boundary`/`arrow`/`arrow_label`) lives at the top of
  `docs/deck-build/build_deck.py`. `docs/deck-build/diagrams/*.png`
  (the old Mermaid renders) were deleted — no longer referenced.
- No LibreOffice in this environment, so no rendered visual QA was
  possible (see `to-do.md`'s "Decisions already made" — this was raised
  with the user, who chose python-pptx over installing LibreOffice).
  Substituted a written shape-aware geometry + WCAG contrast checker,
  committed at `docs/deck-build/qa_check.py` (details in
  `docs/deck-build/README.md`) — it opens the actual generated `.pptx`
  and checks real shape positions/sizes/colors, not source-code estimates.
  Caught and fixed real issues in the native-diagram rebuild: several
  layout-spacing risks, role-text boxes sized for one line receiving text
  long enough to wrap, and a genuine WCAG contrast failure — the brand's
  teal (`009090`) and olive/green (`A8BA4E`) both read fine as large text
  or fills but fall below 4.5:1 at the diagrams' small 10.5–13pt bold
  label size — fixed with darker text-safe variants (`TEAL_TEXT`/
  `CYAN_TEXT`/`OLIVE_TEXT`) used only where the math requires them.
- **Recommend an actual visual pass in PowerPoint/LibreOffice before
  presenting** — the computational checks are a real substitute for "will
  this crash or misrender," not a guarantee of "does this look good."
  **Superseded by C3 below** — the deck grew from 4 sections/15 slides to
  7 sections/25 slides; treat this entry as history, not current state.

### C3 — Three new sections + Networking/Overview deep-dives (2026-07-26)
**Status: done.** User request: "add a slide about the route for a new
record and the admin routes," explicit callouts on *why* the network is
segmented into separate subnets, plus whole new sections on data
architecture (with scalability/performance), observability/telemetry, and
FinOps — all in one large follow-up pass. Deck is now **25 slides across 7
sections**: Overview → Networking → MLOps → **Data Architecture (new)** →
CI/CD → **Observability & Telemetry (new)** → **FinOps (new)**. Agenda
slide's fixed `2.0in`/item spacing (built for exactly 4 items) had to
become adaptive to item count — see `add_agenda_slide`'s `gap`/`roomy`
logic in `docs/deck-build/build_deck.py`.

- **Overview gained a 2nd diagram** (`add_other_routes_diagram`): the two
  routes nothing dashboard-related calls — `POST /records/score` (external
  system scores one record) and `POST/GET /admin/models/*` (ops/CI puts a
  promoted model live) — each its own arrow-based flow, mirroring
  `add_api_routes_diagram`'s visual grammar so the two read as a pair.
  `add_api_routes_diagram` itself was **redesigned once already** earlier
  this session per feedback that its first pass (a 2-column
  "dashboard calls / backend serves" table per resource) read as a spec
  sheet, not a diagram — see that redesign's own note above if picking
  this back up.
- **Networking gained two bulleted slides**: "Why segmented into separate
  subnets?" (Postgres's delegation exclusivity is an Azure platform rule,
  not a design choice; App Service uses a different, incompatible
  delegation type; Private Endpoints share one subnet by convenience, not
  requirement; 4 DNS zones because each service publishes under its own
  fixed Microsoft-defined zone name) and "How everything actually talks to
  everything else" (HTTPS/TLS, AAD token auth for Postgres — no passwords,
  SMB for Azure Files mounts, split-horizon private DNS, OIDC for GitHub
  Actions). Both grounded directly in `terraform/networking.tf`'s own
  comments, not invented — quoted almost verbatim in `docs/
  Architecture-Diagram.md`'s mirrored §2 additions.
- **New §4 Data Architecture** (`add_data_flow_diagram` +
  `add_bulleted_slide` for scalability/performance): a 4-stage data-flow
  diagram (Raw data → Clean & find candidates → Match → Cluster & publish)
  finally brings the pipeline's own logic into the deck natively (it was
  previously only in `docs/Architecture-Diagram.md`'s Mermaid, never
  ported to pptx). Scalability content is honest about a real constraint:
  the backend runs as a single instance today because `src/api/jobs.py`'s
  job-status registries are in-process dicts — a deliberate scope decision
  from Track A, not an oversight, surfaced here rather than hidden.
- **New §6 Observability & Telemetry** (`add_observability_diagram`):
  fan-in/fan-out diagram — 3 resources' diagnostic settings → Log
  Analytics + App Insights → 7 baseline alerts → ops action group. Every
  number on this slide (30-day retention, 7 alerts, severities 0-2) is
  read directly from `terraform/monitoring.tf`, not approximated — this is
  the first time A1's work has appeared in the client deck at all.
- **New §7 FinOps** (`add_finops_diagram`): a hand-built cost-breakdown
  "table" (rows via `textbox`/`rect`, no native pptx table — matches how
  every other slide in this deck is built) totaling **~$150–225/mo
  estimated**, explicitly labeled "estimate, not a quote" — public Azure
  list pricing, no reservations, region-dependent, verify before
  budgeting. Grounded in the actual chosen SKUs (P0v3 App Service —
  reusing the exact $85-90/mo figure already agreed with the user in A3 —
  Burstable Postgres, Basic ACR, `Standard_DS3_v2` scale-to-zero AML
  compute). No live Azure Pricing API access was used or available; this
  is a best-effort estimate from known list-price tiers, not a quote.
- **Real geometry bugs found and fixed via `qa_check.py`** (same
  established workflow as every diagram before this): the record-scoring/
  admin-routes diagram's boundary label initially overlapped the slide
  header (pushed the two flow rows down and recomputed vertical budget);
  the FinOps table's alternating zebra-stripe row background
  (`LIGHTGRAY_BG`) failed WCAG contrast for its gray note text (4.42:1
  vs. the 4.5:1 normal-text threshold) — fixed by dropping the stripe for
  a plain thin divider line instead of inventing a new color variant for
  one slide; two bulleted-slide leadins were too long for their one-line
  boxes and got shortened. Final state: clean geometry/contrast/structural
  validation across all 25 slides.
- `docs/Architecture-Diagram.md` updated to mirror every addition above:
  new §4 Data Architecture (with its own Mermaid diagram), new §6
  Observability & Telemetry (with an alert table pulled straight from
  `monitoring.tf`), new §7 FinOps (with the same cost table), and the
  Networking/Overview deep-dive content folded into §1/§2. Old §4 CI/CD
  renumbered to §5. All Mermaid diagrams re-rendered via
  `@mermaid-js/mermaid-cli` to confirm no syntax errors, same verification
  standard as C1.

### C4 — Promote-model actually goes live; Scalability split out; honest data-lineage diagram (2026-07-26)
**Status: done.** Three pieces of feedback on C3's output, same day.

- **`promote-model.yml` now actually calls `POST /admin/models/reload`** —
  previously it only tagged "champion" and printed a warning that nothing
  else happened. Added a real `curl --fail` step right after tagging (new
  `BACKEND_URL` repo variable, documented in `terraform/README.md`'s
  variable table). **This will fail on the default GitHub-hosted runner,
  on purpose** — the backend has no public ingress (A4), so reaching it
  needs the same VNet connectivity already required for the artifact-copy
  step above it in the same job. Failing loudly here is correct: it means
  the promotion did **not** take effect, not that it silently did.
  `actionlint`-clean (same pre-existing cosmetic shellcheck style nit as
  before, no new issues). Deck's CI/CD diagram and bigblock updated to
  "call this out" per the explicit ask — the Azure ML box now reads
  "Champion tag + goes live" / "Also calls POST /admin/models/reload".
  `docs/Architecture-Diagram.md` §6 (CI/CD) updated to match, including
  the mermaid node text.
- **Scalability & Performance is now its own section (§5)**, not a
  subsection of Data Architecture — split out per explicit feedback.
  Renumbered everything after it: Data Architecture stays §4,
  Scalability & Performance is the new §5, CI/CD → §6, Observability → §7,
  FinOps → §8 (deck is now **27 slides across 8 sections**). Caught and
  fixed two stale internal cross-references in `docs/Architecture-Diagram.md`
  left over from the renumber (a "Section 4 covers how" that should've said
  "Section 6", and a FinOps "(Section 6)" alert reference that should've
  said "Section 7") — grep for `Section [0-9]` before trusting any
  future renumber in this doc.
- **New Data Architecture content, ground-truthed before drawing anything**
  (per explicit instruction not to invent this): the user asked for a
  slide on how reviewer-confirmed matches and retraining data connect.
  Spawned a research pass first rather than assume — finding, stated
  plainly on the new slide itself: **`audit_log` (reviewer merge/unmerge/
  dismiss) is NOT wired into either matcher's training labels today.** A
  repo-wide grep for `audit_log` outside `src/api/routers/audit.py` and its
  backends/tests returns nothing in `src/models/` or anywhere in
  `empi-model-training/` — it's a record-locking + audit-trail mechanism
  only. What actually IS used: **silver labels** (Stage-3 deterministic-
  rule confirmations, `data/silver_labels/*.csv`) and **gold labels** (a
  separate, small, hand-adjudicated set, `data/raw/gold_labels.csv` —
  explicitly documented elsewhere as "still a work in progress"), both
  staged into Azure ML by hand, same manual-copy constraint as everything
  else in Section 3. New `add_data_lineage_diagram()` in
  `docs/deck-build/build_deck.py` shows this real lineage, with the
  reviewer-action branch visually dead-ending (red box, "NOT used for
  retraining... a real gap, not built yet") rather than silently omitting
  it or fabricating a connection. Mirrored in `docs/Architecture-
  Diagram.md` §4 with the same mermaid diagram. **If reviewer-feedback-
  driven retraining is something worth actually building, that's new
  scope, not something this pass discovered already existed — flag to the
  user before assuming it's wanted.**
- Full re-QA after all three changes: `qa_check.py` clean (geometry +
  contrast) on the first rebuild for the new lineage diagram; `validate.py
  --original` passed; `markitdown` placeholder grep clean; all 8 mermaid
  blocks in `docs/Architecture-Diagram.md` re-rendered with no syntax
  errors.

### C5 — Built reviewer-confirmed-matches-to-retraining feature (2026-07-26)
**Status: done**, per an explicit plan-mode review with the user before any
code was written (`/Users/jasonclark/.claude/plans/linear-juggling-
rossum.md`). Follow-up to C4's honest finding that this connection didn't
exist — the user asked "is it as simple as pulling that data from
Postgres during retraining," which surfaced two real gaps (`audit_log`
doesn't capture enough to derive correct pairs; no training script merges
more than one label source) that got scoped and built rather than papered
over.

- **`audit_log` gained a `related_patids` column** — a membership snapshot
  captured **at write time**, not reconstructed later. `entity_member` only
  ever holds current state, so reconstructing "who was in this entity when
  the reviewer acted" after the fact is unreliable once anything else has
  touched that entity since. Both `merge` and `unmerge` handlers
  (`src/api/routers/audit.py`) already had the data needed in-memory at the
  moment of the event — `target["members"]` (merge, fetched before the
  mutation) and `remaining_patids` (unmerge, already computed, just never
  persisted) — so this was capturing an existing value, not adding new
  computation. Threaded through all 4 `insert_audit_log` implementations
  (`index_backend.py` Protocol + wrapper, `sql_backend.py`,
  `postgres_backend.py`, `parquet_backend.py`) and `AuditLogRow`. An
  **already-deployed** Postgres database needs a one-time manual
  `ALTER TABLE audit_log ADD COLUMN related_patids TEXT;` — no migration
  framework exists in this repo (`terraform/README.md` §7, new).
- **New `empi-service/scripts/export_reviewer_labels.py`** derives
  `(PATID_A, PATID_B, reviewer_label)` pairs: `merge` → positive pairs
  across `patids ∪ related_patids`; `unmerge` → negative pairs between the
  removed patid and `related_patids`; `dismiss` → a direct negative pair
  (already 2 patids, no entity involved). Pre-migration `unmerge` rows
  (`related_patids IS NULL`) are **skipped and counted**, not guessed at.
  Conflicting derivations of the same pair resolve last-write-wins by
  `ts_utc`. Verified two ways: 12 unit tests with synthetic fixtures, and a
  real end-to-end smoke test — seeded actual merge/unmerge/dismiss calls
  through the live FastAPI app against a temp SQLite DB, ran the real
  export logic against it, and hand-verified the output was exactly
  correct, including a genuine merge→unmerge conflict resolving correctly
  (P1-P3 kept its positive label since the unmerge never touched that
  pair; P1-P2 and P2-P3 correctly flipped to negative). Output written to
  `data/reviewer_labels/*.csv`, PHI-safe shape matching `data/silver_labels/`,
  gitignored the same way.
- **Wired into training** — `--reviewer-labels` added to
  `empi-service/src/models/fs_matcher/train.py`,
  `empi-model-training/training/fs_train.py`, and
  `empi-model-training/training/lightgbm_train.py`. Reviewer labels win on
  conflict (highest-trust — a live human decision, not a proxy). LightGBM's
  label scheme needed a real judgment call, not a mechanical copy: its
  target is `ambiguous_pair` genuinely being ambiguous, not just
  match/no-match, so reviewer-derived rows get `ambiguous_pair=True`
  unconditionally — by construction, a pair only ever reaches a reviewer
  action because something upstream already flagged it as uncertain,
  never a slam-dunk case a human would never see. **Not wired into
  `submit.py`** (the Azure ML job-submission entry point only takes one
  `--labels` reference) — works today for local/direct-CLI training only;
  extending `submit.py` + the AML component defs is flagged as a follow-up
  in `empi-model-training/README.md`, not silently assumed done.
- 19 new tests total across both repos (4 backend + 12 export-script + 4
  fs_matcher merge-logic in empi-service; 2 fs_train + 3 lightgbm_train in
  empi-model-training), all passing alongside the full existing suites
  (679 + 10). `ruff`/`mypy --strict` clean on every touched file — one real
  line-length violation caught and fixed in `lightgbm_train.py`, everything
  else flagged was pre-existing debt unrelated to this change.
- Deck + `docs/Architecture-Diagram.md` §4 updated to match the built
  reality — the "NOT used for retraining" dead-end is gone; all three
  label sources (silver, gold, reviewer) now visibly converge into the
  Azure ML staging step, with reviewer labels called out as highest-trust.

### C6 — Retry + database lifecycle decoupling (backend); deck resiliency content + section reorder (2026-07-26)
**Status: done.** Follow-up to an "any other architecture gaps" review:
two gaps closed in `empi-service`, then reflected in the client deck.

- **Background jobs now retry transient failures.** `run_pipeline_job` and
  `score_records_job` (`src/api/jobs.py`) each retry up to 3 attempts with
  exponential backoff before permanently failing — no new dependency
  (no tenacity), just a loop around the existing try/except.
- **Job status now survives a crash, not just success.** A durable
  `.status.json` file is written at every transition (queued/running/
  succeeded/failed, attempt, last error) under `data/runs/` — extending
  the existing `RunManifest` file pattern rather than adding a database
  table, per explicit instruction: **no new migration framework or schema
  changes** — that's a larger decision for infra/code owners, not
  something to introduce unilaterally here. `GET /runs/{run_id}` and
  `GET /records/score/{run_id}` (`runs.py`/`records.py`) fall back to that
  file once the in-memory registry is empty (the post-restart case), and a
  new `jobs.reconcile_interrupted_jobs()` runs at `lifespan` startup to
  mark anything left "running"/"queued" by a dead process as interrupted.
- **The app no longer creates or alters its own database schema.**
  `lifespan` and `build_index_backend()` both used to call `init_db()` —
  the latter on nearly every request, since it's the dependency behind
  almost every route. Both calls removed; schema setup is now a single
  explicit `scripts/init_db.py`, run deliberately once per environment.
  A read-only startup check logs a loud error if the schema looks missing
  instead of silently reconstructing it. This is the actual fix for the
  original ask — the database is now "disconnected and consumed," not
  reconstructed on every start or push, so reviewer decisions persist
  across a deploy.
- Manual crash-restart smoke test caught a real bug: `list_runs()`'s
  manifest glob (`run_*.json`) also matched `run_*.status.json` files
  (both end in `.json`), producing a mangled duplicate entry. Fixed by
  excluding `.status.json` names from the manifest glob, with a regression
  test. 693 tests passing (13 new), `ruff`/`mypy` clean on every touched
  file.
- **Deck updated to match**: new bigblock slide in the Scalability section
  ("What happens when something goes wrong" — retries, durable status,
  decoupled schema management), section renamed **Scalability,
  Performance & Resiliency**, and its stale "job status lives in memory"
  bullet corrected to reflect the durable-file addition.
- **Sections reordered** per explicit feedback to improve narrative flow:
  **CI/CD moved from §6 to §5** (right after Data Architecture, since its
  content — model promotion, code deploys — directly operationalizes what
  MLOps/Data Architecture just introduced), pushing **Scalability,
  Performance & Resiliency to §6**, now immediately followed by
  Observability at §7 — the two "how healthy is this in production"
  sections sit back-to-back instead of split by CI/CD. Observability (§7)
  and FinOps (§8) unchanged. All `Section N`/`§N` cross-references in
  `docs/Architecture-Diagram.md` audited and fixed (only two needed it —
  a forward reference from MLOps to the old §6, and the section header
  order list); `qa_check.py` clean, `markitdown` content dump clean.
- Updated the stale "Redis-backed job registry: descoped" decision below —
  still true for *multi-instance* scaling, but the specific "a crash loses
  all job state" complaint that decision used to leave open is now fixed
  by the durable status files above.
- **Follow-up edit, same day, direct on the .pptx, not via `build_deck.py`.**
  The user had since heavily hand-restructured the deck themselves outside
  this build script (28 slides → 22; shape names show a Google Slides
  round-trip) — regenerating from `build_deck.py` would have discarded
  that work, so this was a surgical python-pptx edit of two slides in
  place, explicitly told not to touch anything else. Reworded all 4 boxes
  on the resiliency slide to drop every before/after comparison ("now",
  "no longer", "used to", "instead of") and state only the current
  behavior. Bumped FinOps' smallest text (10.5pt row descriptions → 13pt,
  12pt bullets/header → 14pt, 13.5–14.5pt labels/costs → 15–16pt) — it was
  hard to read. `qa_check.py` confirms both edited slides are still clean
  (no new overflow/overlap); it also flags pre-existing issues on other
  slides from the user's own restructuring, correctly left alone per this
  instruction. **`build_deck.py` was intentionally left unrun and now
  reflects an older layout than the shipped `.pptx`** — do not regenerate
  from it without reconciling that drift first.
- **Second follow-up, same day: condensed Scalability from 2 slides to 1.**
  The 5-bullet "what's fast by design" slide and the 4-box resiliency
  slide merged into a single 6-box slide, per explicit instruction ("go
  from 4 → 6 boxes max"). Implementation note for next time this file is
  touched: done by deep-copying 2 of the 4 existing rect+heading+body
  shape groups (giving fresh, unique `cNvPr/@id`s — a naive deepcopy
  duplicates the id, which is invalid OOXML), then repositioning all 6
  into a 3×2 grid (`col_w=5.73in`, `row_h=3.55in`, matching the col_w
  already used elsewhere in `build_deck.py`) and recoloring on the same
  BLUE/CYAN/TEAL/OLIVE/BLUE/CYAN rotation (fill→text-color pairing
  — BLUE/TEAL get white text, CYAN/OLIVE get dark `#26292C` — pulled
  directly from the existing shapes' XML, not guessed). The old 5-bullet
  slide was then deleted via `<p:sldIdLst>` removal + `part.drop_rel()`.
  Content-wise, merged the former "single instance" + "headroom" bullets
  into one box, and "retries" + "job status survives a restart" into
  another, keeping the same no-before/after-comparison language from the
  prior edit. Deck is now **21 slides** (was 22, then 21 after this
  merge). `qa_check.py` re-run after: the new slide (now §6's sole content
  slide, deck slide 16) is clean; same pre-existing issues as before on
  other, untouched slides.

## Decisions already made (don't re-litigate without a reason)

- **PHI timeline:** real PHI is expected in this environment before long →
  full network/compliance hardening is in scope (VNet, private endpoints,
  customer-managed keys), not deferred.
- **Auth:** Microsoft Entra ID (Azure AD) SSO, via App Service's built-in
  **Easy Auth** (platform-level, sits in front of the container) — not an
  in-app OIDC library. This was chosen specifically so local dev needs zero
  changes (see next bullet).
- **Local-dev-first:** every change here must keep `docker-compose up` /
  `npm run dev` / `pytest` working with **zero Azure dependency**. Cloud
  pieces (Easy Auth, Azure Monitor, Azure ML) are additive layers on top of
  the same code, never a hard requirement to run or test locally.
- **Redis-backed job registry: still descoped, but partially superseded
  (C6).** `src/api/jobs.py`'s in-process `_REGISTRY`/`_SCORE_REGISTRY`
  dicts, and the resulting single-*instance* limit on the backend, are
  still **not** being changed — a shared store is still what scaling out
  to multiple instances would need, and the scale doesn't justify that
  yet. What C6 did fix: a crashed *single* process no longer loses all
  record of its in-flight jobs — status is now also written durably to
  disk and reconciled on restart. Revisit the shared-store question only
  if real multi-instance throughput needs surface.
- **MLOps scope:** Azure ML Workspace for **training + experiment tracking +
  model registry only**. Serving stays exactly as today — FastAPI loads a
  model artifact from `models/fs` / `models/ml` and scores in-process. No
  Azure ML managed online endpoints.

## Track A — Infrastructure hardening

Dependency order matters: A3 must land before A4 (VNet integration requires
a Premium App Service tier).

### A1 — Observability: Log Analytics, App Insights, baseline alerts
**Status: done.**
- `terraform/monitoring.tf`: Log Analytics workspace, workspace-based
  Application Insights, diagnostic settings (backend, dashboard, Postgres),
  an action group + baseline metric alerts (backend 5xx, backend/dashboard
  health check, App Service Plan CPU/memory, Postgres CPU/storage).
- Wires `APPLICATIONINSIGHTS_CONNECTION_STRING` into both App Services'
  `app_settings`.
- New variables: `log_retention_days`, `alert_notification_emails` (empty
  list by default — no personal address hardcoded; must be set in
  `terraform.tfvars` for alerts to actually notify anyone).
- No effect on local dev — purely additive Azure resources + an optional
  env var the app never requires to boot.

### A2 — Postgres backup/HA hardening
**Status: done, with one deferred decision.**
- `postgres_backup_retention_days` (default 35) and
  `postgres_geo_redundant_backup` (default `true`) added and wired into
  `azurerm_postgresql_flexible_server.main` — cheap disaster-recovery
  coverage.
- **Deferred:** zone-redundant HA is *not* enabled — it requires moving
  `postgres_sku` off the Burstable tier (Burstable doesn't support the
  `high_availability` block at all), a real cost jump. Decide this together
  with A3's App Service SKU conversation, since both are "leave the
  cheapest tier" decisions with the same underlying question: how much are
  we provisioning for before this is presentation-only vs. actually serving
  traffic.

### A3 — Right-size the App Service Plan off B1
**Status: done.**
- `var.app_service_sku` default changed `"B1"` → `"P0v3"` (confirmed with
  user given the real cost jump, ~$13/mo → ~$85-90/mo for the shared plan).
  Bump further to `P1v3`/`P2v3` if the App Service Plan CPU/memory alerts
  from A1 trip under real load.
- Unblocks A4 (VNet integration requires a Premium tier).

### A4 — VNet integration + private endpoints + customer-managed keys
**Status: done.**
- `terraform/networking.tf`: one VNet, three subnets (App Service VNet
  Integration, Postgres's dedicated delegated subnet, a shared Private
  Endpoint subnet), four private DNS zones + VNet links (postgres,
  storage-file, sites, vaultcore).
- Postgres: switched to VNet-injected private access (`delegated_subnet_id`
  + `private_dns_zone_id`), `public_network_access_enabled = false`, old
  "allow Azure services" firewall rule removed (not valid in this mode).
- Storage: `public_network_access_enabled = false` + a Private Endpoint for
  the `file` subresource (the SMB mounts app_service.tf uses).
- Backend: **no public ingress at all** (own inbound Private Endpoint +
  `public_network_access_enabled = false`) — confirmed with user despite
  the operational cost (see below). Also has outbound VNet Integration to
  reach Postgres/Storage privately.
- Dashboard: stays public (browsers need it), gets outbound VNet
  Integration so its existing calls to the backend's normal hostname
  resolve privately — no dashboard code change needed.
- `terraform/keyvault.tf`: private, RBAC-authorized Key Vault + one shared
  user-assigned identity (avoids a chicken-and-egg dependency vs. each
  resource's own system-assigned identity) + customer-managed RSA keys for
  both Storage and Postgres.
- **Real consequences, addressed:** `deploy-backend.yml`'s health check
  switched from curling the public `/health` URL to polling
  `az webapp show --query state` (ARM-plane, works regardless of network
  placement, but a weaker signal than an actual HTTP probe). The FS-matcher
  bootstrap step (`az webapp ssh`) also now needs VNet connectivity or a
  temporary `public_network_access_enabled` flip — documented in
  `terraform/README.md`. Direct `curl`/Postman debugging of the backend
  from a laptop no longer works without a VPN/bastion.
- Full writeup: `terraform/README.md` "Network isolation & encryption".
- Deferred: no NSGs added yet (default intra-VNet allow-all) — a reasonable
  future hardening step, not done here to control scope.

### A5 — Entra ID SSO for the dashboard
**Status: done.** (Scoped down from "both App Services" to dashboard-only
once A4 landed — see reasoning below.)
- `terraform/auth.tf`: new `azuread_application` + `azuread_service_principal`
  + `azuread_application_password` for the dashboard, mirroring the pattern
  already used for GitHub OIDC in `github_oidc.tf`.
- `app_service.tf`: `auth_settings_v2` (Easy Auth) on the dashboard's
  `azurerm_linux_web_app`, requiring authentication via that app
  registration.
- **Backend intentionally excluded**: it has no public ingress at all as of
  A4, so there's no browser-facing surface on it that needs its own login —
  its access control is network isolation, not authentication.
- App code: `empi-dashboard/src/lib/server-api.ts`'s new `reviewerId()`
  prefers the Easy-Auth-injected `X-MS-CLIENT-PRINCIPAL-NAME` header,
  falling back to the original hardcoded `REVIEWER_ID` when absent (always
  true locally, since there's no Azure platform in front of
  `docker-compose`/`npm run dev`). Verified: `tsc --noEmit` and `eslint`
  clean, and a local dev-server smoke test confirmed the fallback resolves
  without error (only the expected `ECONNREFUSED` from the backend not
  running locally).

## Track B — MLOps

Lives in `empi-model-training/` — nested inside this repo and pushed as
part of it (not a submodule, not a separate remote), but a **logically
independent codebase**: it does not import or call into `empi-service`'s
code, and vice versa. Two early direction changes worth knowing if you
didn't see them happen live:
- First built as a thin wrapper calling `empi-service`'s training code
  in-process (cross-repo import). **Reversed** — user directive: no
  cross-repo calls, period. Necessary logic (the FS comparison structure,
  the LightGBM feature engineering) is faithfully reimplemented
  independently instead; see `empi-model-training/CLAUDE.md`.
- First scaffolded as a true sibling repo (`~/Workspace/empi-model-training`,
  its own `git init`). **Corrected** — user only has the one GitHub remote
  and wants this pushed as part of it, so it's nested at
  `UofC-EMPI/empi-model-training/` with no `.git` of its own, tracked
  normally by this repo. **Do not add `empi-model-training/` to this
  repo's `.gitignore`** — it needs to be pushed, not hidden.

### B1 — Azure ML workspace
**Status: done.** `terraform/ml_workspace.tf`: workspace (reuses Track A's
Key Vault/App Insights/ACR, gets its own dedicated storage account),
managed-network isolation (`AllowOnlyApprovedOutbound` — chosen over
injecting a customer VNet + hand-written NSGs, which is a well-known
footgun this review couldn't verify against a live subscription anyway), a
small scale-to-zero compute cluster (`cpu-cluster`, DS3_v2, max 2 nodes).
**Consequence:** AML compute has no network path to the app's private
storage (different network boundary) — training data in / model artifact
out is an explicit staged step, not automatic. See
`empi-model-training/README.md` "Data movement."

**Remaining:** RBAC so the workspace/compute identities and the GitHub
Actions CI identity can actually do anything (storage access, ACR pull, key
vault access, AML data-plane permissions for the champion-promotion
workflow below) — not yet added to `ml_workspace.tf`.

### B2 — Training scripts (independent implementations)
**Status: done.** `empi-model-training/src/empi_model_training/training/`:
- `fs_train.py` — independent Splink-based FS matcher trainer (7
  comparisons, supervised m, u via random sampling, λ seeded from
  deterministic-rule priors — matches
  `empi-service/docs/FS-Matcher-Production-Guide.md`'s documented
  procedure). Needs only cleaned records + labeled pairs; no real
  candidate-pairs table required for training (only serving needs that).
- `lightgbm_train.py` — independent reimplementation of both the LightGBM
  v3 feature engineering (`empi-service`'s `V3FeatureBuilder`, ported) and
  the training loop (previously only a notebook) — same hyperparameters,
  same 60/20/20 stratified split, same plausible-pool population filter.
- `registry_utils.py` — small independent copy of the deploy-gate/
  `active.json` pattern, same file-naming convention as empi-service's
  `registry.py` so a promoted artifact drops into the serving share
  unchanged.
- Both scripts run with **zero Azure dependency** and are covered by real
  (not mocked) end-to-end pytest smoke tests against synthetic fixtures —
  a genuine Splink fit and a genuine LightGBM fit, which already caught one
  real bug (a missing Splink column-harvesting shim) before this ever got
  near Azure ML.
- MLflow logging is unconditional (works locally via the file-store
  fallback); model-registry registration is conditional on actually
  running under Azure ML tracking.

### B3 — Azure ML components + submission
**Status: done.** `src/empi_model_training/`:
- `components/{fs_train_component,lightgbm_train_component}.py` — AML
  `command()` component definitions. Each just shells out to the matching
  `training/*.py` CLI unchanged (`code=<src/>`, so the component's snapshot
  IS the package) — never a second copy of the training logic.
- `utils/azure_client.py` — shared `MLClient` builder
  (`AZURE_SUBSCRIPTION_ID`/`AZURE_RESOURCE_GROUP`/`AZURE_ML_WORKSPACE_NAME`
  env vars, sourced from `terraform output`).
- `utils/register_environment.py` — registers the training environment
  (`environment.yml` at the repo root) as an AML Environment asset. No
  Terraform resource exists for AML environments/datasets/models — they're
  workspace-internal, frequently-revised assets managed through the SDK,
  not infrastructure.
- `utils/register_dataset.py` — registers a local file/folder as a
  versioned AML Data asset (`azureml:name:version`) — the "data in" half of
  B1's data-movement step.
- `utils/preflight.py` — checks workspace/compute/environment/dataset all
  resolve *before* submitting; `submit.py` always runs this first.
- `submit.py` — the one entry point that actually submits a job
  (`--job fs|lightgbm --cleaned-index azureml:... --labels azureml:...
  --compute cpu-cluster [--promote]`).

### B4 — GitHub Action: promote to champion
**Status: done.** `.github/workflows/promote-model.yml` — no new OIDC trust
needed (federated credential trust in `github_oidc.tf` is scoped to the
repo/ref, not per-workflow, so the existing CI identity already covers a
new workflow in this same repo; it just needed the explicit "AzureML Data
Scientist" role added in B1's RBAC pass).
- `workflow_dispatch` with `model_name`/`model_version` inputs.
- Two jobs: `show-metrics` (unconditional — prints the candidate version's
  metadata and linked-run details into the run summary) then `promote`
  (gated by the `production` GitHub Environment, same human-approval
  pattern as `terraform-apply.yml` — a reviewer sees `show-metrics`'
  output before approving). On approval: clears any previous `champion` tag,
  tags the approved version.
- New repo variable: `ML_WORKSPACE_NAME` (documented in
  `terraform/README.md`'s GitHub Actions setup table).
- Validated with `actionlint` (clean beyond one cosmetic shellcheck style
  nit).
- **Explicitly NOT automated:** actually copying the champion artifact into
  the `empi-models` Azure Files share (so serving picks it up) — same
  private-storage consequence as the backend health check and the FS
  bootstrap step elsewhere in this repo. The workflow's last step prints a
  clear warning + pointer to the manual workaround rather than silently
  claiming success. **What no longer needs manual work after that copy:**
  see B5 — `POST /admin/models/reload` puts the copied artifact live with
  no restart, so the only remaining manual link is the copy itself.

### B5 — Model hot-reload: no-downtime promotion cutover
**Status: done.** (2026-07-26, requested alongside a client-deck ask to call
out the backend's actual API routes.) Closes the last piece of B4's
"explicitly not automated" gap — not the artifact copy itself, but the step
after it.

- **Ground truth checked first, and it wasn't what was assumed going in:**
  neither matcher was cached in memory before this — `FSMatcher.load_settings`
  and `registry.load_model_artifact` both re-read from disk on *every*
  pipeline run and every incremental `/records/score` call
  (`src/pipeline.py`, `src/api/ingest/incremental.py`). That means a
  promoted model already took effect on the very next call, zero code
  required. Said so plainly rather than "fixing" a staleness bug that
  didn't exist.
- `empi-service/src/models/model_cache.py` (new): adds the in-memory
  caching itself (avoids re-deserializing the same artifact on every call
  once there's real traffic), keyed on `(path, mtime)` so it
  self-invalidates the moment a promoted model's file changes — caching
  added without reintroducing the staleness it would normally cost.
  Wired into both FS matcher load sites and the one ML matcher load site.
- `POST /admin/models/reload` + `GET /admin/models/status`
  (`empi-service/src/api/routers/admin.py`, new): not reviewer-facing, no
  dashboard route calls these. Makes the cache-refresh moment immediate,
  synchronous, and observable (echoes the newly-resolved active model
  meta) instead of implicit and whenever-the-next-request-happens to be —
  call it right after copying a promoted artifact into Storage to confirm
  the swap actually took before calling the promotion done. No new
  per-route auth — matches every other route here, protected by the
  backend's lack of public ingress (A4), not an application check.
- Tests: `empi-service/tests/unit/models/test_model_cache.py` (cache
  hit/mtime-invalidation/explicit-invalidate behavior) +
  `TestAdmin` in `tests/integration/test_api.py` (the two routes via
  `TestClient`). Full suite (659 tests) green.
- **One real bug caught during testing, worth remembering:** the shared
  `test_settings` fixture in `test_api.py` doesn't isolate
  `fs_model_dir`/`ml_model_dir` — no earlier test needed to write model
  artifacts. My first test run before I noticed this wrote real files
  (`active.json`, `fs_model_1.json`) straight into the actual
  `empi-service/models/fs/` on disk. Cleaned up immediately; `TestAdmin`
  now has its own `_isolated_model_dirs` autouse fixture. If a future test
  in this file calls `fs_registry.promote()`/`ml_registry.promote()` for
  the first time, check it isolates these two settings fields too.
- Documented in `empi-service/docs/API-Design.md` (new "Admin (model
  hot-swap)" §3 subsection) and `docs/Architecture-Diagram.md` (new "API
  routes" subsection under §1, plus updates to §3 MLOps and §4 CI/CD's
  "known gap" paragraph — the artifact copy is still manual, the cutover
  after it no longer is).
- Deck (`docs/eMPI-Architecture-Deck.pptx`): new slide after the Overview
  diagram mapping every backend route to the dashboard BFF route that
  calls it (ground-truthed from `empi-dashboard/src/app/api/*/route.ts`'s
  actual `apiGet`/`apiPostJson`/`apiPostForm` call sites, including which
  routes the dashboard does *not* call). **Redesigned once already**, per
  feedback that the first pass (a 2-column "dashboard calls / backend
  serves" table per resource, `route_group()`) read as a spec sheet, not a
  diagram — replaced with an actual flow: Browser → Dashboard → Backend as
  three `svc_box` nodes with real arrows (a bold one for Dashboard→Backend,
  labeled "every call is proxied server-side — the browser never reaches
  the backend directly"), then the 6 call categories as a 3×2 grid of
  `svc_box` cards underneath (single technical route in each card's
  `detail` line, not two full columns), same "not called by the dashboard"
  footer kept as-is. `route_group()` was deleted (dead code once nothing
  called it). MLOps diagram gained a 5th box (`empi-service (backend)` —
  "Swaps in the new model" / "No restart — POST /admin/models/reload")
  after Azure Storage Account. Both diagrams passed `qa_check.py` clean on
  the geometry/contrast pass (one label-length fix needed along the way:
  "Starts using the new model instantly" wrapped to 2 lines in a 1-line
  box — shortened).

## Reference

- Full architecture + pipeline diagrams: `docs/Architecture-Diagram.md`
- API routes + admin/model-reload contract: `empi-service/docs/API-Design.md`
- Terraform: `terraform/`
- Gap list originally surfaced 2026-07-25.
