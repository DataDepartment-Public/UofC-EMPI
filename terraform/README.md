# eMPI Azure infrastructure

Terraform for the eMPI deployment: one resource group holding an Azure
Container Registry, two Linux App Services running the backend
(`empi-service`, FastAPI) and dashboard (`empi-dashboard`, Next.js) as
containers, a Storage Account (Azure Files) mounted into the backend for
`data/`, `models/`, `logs/`, and an Azure Database for PostgreSQL Flexible
Server that replaces the SQLite file as the resolved-output index in this
environment (`empi-service/src/api/backends/postgres_backend.py` — a
Postgres-flavored implementation of the same `IndexBackend` interface
`sql_backend.py` implements for SQLite; a document store like Cosmos DB was
considered and rejected because the schema is relational — FK joins,
multi-table transactions, `ON CONFLICT` upserts throughout). See the
comments at the top of `postgres.tf` and `app_service.tf` for more.

This doc is for whoever has Azure subscription access and is running this
for the first time. Everything after "Bootstrap" is one-time; day-to-day
changes go through the GitHub Actions workflows in `.github/workflows/`.

## What this does NOT do

- No Redis — the backend's job-status registry is in-process
  (`empi-service/src/api/jobs.py`), which is also why both App Services are
  pinned to a single instance (`always_on = true`, no autoscale). Don't
  raise the instance count without addressing that first. (Postgres solves
  the *data* side of scaling out; the job registry is a separate problem.)
- No automatic data migration from an existing local `empi.db` — the
  Postgres database starts empty. If you have real pipeline output in a
  local SQLite file you want carried over, that's a one-off export/import,
  not something this Terraform or `postgres_backend.py` does for you.

## Prerequisites

- Azure CLI (`az`), logged in (`az login`) as someone with:
  - **Owner** (or Contributor + User Access Administrator) on the target
    subscription — this config creates RBAC role assignments, which plain
    Contributor cannot do.
  - An Azure AD directory role that can create App Registrations —
    **Application Administrator** or **Cloud Application Administrator**.
    This is separate from, and not implied by, any subscription-level role.
- Terraform >= 1.9 (`brew install hashicorp/tap/terraform` or see
  terraform.io).
- Docker, if you want to test image builds locally before the first CI run.

## Bootstrap (one-time)

### 1. Create the state storage account

Terraform's own state has to live somewhere before this config can create
its own resource group — a small chicken-and-egg that gets solved with a
few plain `az` commands, outside Terraform:

```bash
az group create --name rg-empi-tfstate --location canadacentral

az storage account create \
  --name stempitfstate$RANDOM \
  --resource-group rg-empi-tfstate \
  --location canadacentral \
  --sku Standard_LRS \
  --min-tls-version TLS1_2

az storage container create \
  --name tfstate \
  --account-name <storage-account-name-from-above> \
  --auth-mode login
```

Grant yourself data-plane access on it (needed because the backend uses
Azure AD auth, not an account key — see `providers.tf`):

```bash
az role assignment create \
  --assignee "$(az ad signed-in-user show --query id -o tsv)" \
  --role "Storage Blob Data Contributor" \
  --scope "$(az storage account show -n <storage-account-name> -g rg-empi-tfstate --query id -o tsv)"
```

### 2. Configure and apply

```bash
cd terraform
cp backend.hcl.example backend.hcl        # fill in the storage account name from step 1
cp terraform.tfvars.example terraform.tfvars   # defaults are fine unless you need to change region/naming

terraform init -backend-config=backend.hcl
terraform plan   # review — this creates the resource group, networking (VNet,
                  # subnets, private endpoints, DNS zones), ACR, both App
                  # Services, the Postgres Flexible Server, storage accounts,
                  # Key Vault + customer-managed keys, the Azure ML workspace +
                  # compute cluster, Log Analytics + Application Insights +
                  # alerts, the Entra ID app registration for dashboard SSO,
                  # and the GitHub OIDC app registration + role assignments —
                  # 50+ resources; review the plan rather than skimming this list.
terraform apply
```

### 3. Let CI read/write state too

The state storage account from step 1 isn't managed by this Terraform config
(deliberately — see the chicken-and-egg note above), so the GitHub Actions
identity this config just created needs its own grant on it:

```bash
terraform output -raw github_actions_client_id   # note this

az role assignment create \
  --assignee <client-id-from-above> \
  --role "Storage Blob Data Contributor" \
  --scope "$(az storage account show -n <storage-account-name> -g rg-empi-tfstate --query id -o tsv)"
```

### 4. Wire up GitHub Actions

In the repo's **Settings -> Secrets and variables -> Actions**:

**Secrets:**

| Secret | Value |
|---|---|
| `AZURE_CLIENT_ID` | `terraform output -raw github_actions_client_id` |
| `AZURE_TENANT_ID` | `terraform output -raw github_actions_tenant_id` |
| `AZURE_SUBSCRIPTION_ID` | `terraform output -raw github_actions_subscription_id` |

**Variables:**

| Variable | Value |
|---|---|
| `RESOURCE_GROUP_NAME` | `terraform output -raw resource_group_name` |
| `ACR_LOGIN_SERVER` | `terraform output -raw container_registry_login_server` |
| `BACKEND_APP_NAME` | the app name Terraform used — see `local.backend_app_name` (`main.tf`) or `az webapp list -g <rg> -o table` |
| `DASHBOARD_APP_NAME` | same, for `local.dashboard_app_name` |
| `TF_STATE_RESOURCE_GROUP` | `rg-empi-tfstate` |
| `TF_STATE_STORAGE_ACCOUNT` | the storage account name from step 1 |
| `TF_STATE_CONTAINER` | `tfstate` |
| `TF_STATE_KEY` | `empi-prod.tfstate` (or whatever you put in `backend.hcl`) |
| `ML_WORKSPACE_NAME` | `terraform output -raw ml_workspace_name` — used by `promote-model.yml` |
| `BACKEND_URL` | `terraform output -raw backend_url` — used by `promote-model.yml`'s `POST /admin/models/reload` call. Only reachable from a runner with VNet connectivity (the backend has no public ingress) — that call fails loudly on the default GitHub-hosted runner by design; see the workflow's own comments. |

Create a `production` GitHub Environment (Settings -> Environments) with
required reviewers for a real manual-approval gate — both
`terraform-apply.yml` and `promote-model.yml`'s `promote` job already target
`environment: production`, so adding protection rules there covers both; no
workflow change needed.

### 5. First deploy

Push to `main` (or re-run `deploy-backend.yml` / `deploy-dashboard.yml` via
`workflow_dispatch`) to build and push the images and point the App Services
at them. The Terraform-provisioned apps start on a `:latest` placeholder tag
that doesn't exist in your ACR yet, so **the App Services will show
unhealthy until the first successful deploy workflow run** — expected, not a
bug.

**Required, not optional, on a brand-new Postgres instance:** the backend
no longer creates its own database schema on startup — it only connects,
and logs a loud error (without crashing) if the schema isn't there yet.
Run `python scripts/init_db.py --backend postgres` once, from something
with VNet connectivity (same access pattern as step 6 below), before the
app is actually usable. Re-run it whenever a code change adds a new column
to `_COLUMN_MIGRATIONS` (`empi-service/src/api/backends/postgres_backend.py`).

### 6. Bootstrap the FS matcher model (optional, can happen anytime after deploy)

The backend runs fine with no trained model (Stage 4 of the pipeline is
skipped with a log line — see
`empi-service/docs/FS-Matcher-Production-Guide.md` "Bootstrapping"). When
you're ready to train one against real data in the deployed environment, use
the Kudu/SSH console (`az webapp ssh --name <backend-app> --resource-group
<rg>`) or a one-off `az webapp` command to run:

```bash
python -m src.models.fs_matcher.train --promote
```

**Since A4 (network isolation, see below), `az webapp ssh` needs direct
network reachability to the backend's SCM/Kudu site, which no longer has
public ingress.** Run it from something with VNet connectivity (a jump
box/bastion, or a VNet-injected Cloud Shell), or — for this kind of
infrequent, one-off task — temporarily flip
`azurerm_linux_web_app.backend`'s `public_network_access_enabled` back to
`true` via a one-line override, `apply`, run the training command, then
revert and `apply` again. Not worth standing up permanent bastion
infrastructure for something this occasional.

### 7. Application-level schema changes (not Terraform's job)

`empi-service`'s app-level SQL schema (`entity`, `audit_log`, etc.) is
created by the app itself on startup (`CREATE TABLE IF NOT EXISTS` —
`src/api/backends/sql_backend.py`/`postgres_backend.py`), not by Terraform,
and there's no migration framework in this repo. A column *added* to an
existing table (e.g. `audit_log.related_patids`, added for
`scripts/export_reviewer_labels.py`) needs a one-time manual
`ALTER TABLE ... ADD COLUMN ...` against any **already-deployed** Postgres
database — `CREATE TABLE IF NOT EXISTS` only helps a brand-new one. Run it
the same way you'd reach the backend for anything else post-A4 (VNet
connectivity — bastion, jump box, or a temporary
`public_network_access_enabled` flip, same options as step 6 above).

## Day-to-day

- Push (or merge a PR) touching `empi-service/**` -> `deploy-backend.yml`
  builds, pushes to ACR, updates the App Service, waits for it to reach
  `Running` state (an ARM-plane check, not a `/health` curl — see the
  workflow's comment and "Network isolation & encryption" below for why).
- Push touching `empi-dashboard/**` -> `deploy-dashboard.yml`, same shape,
  checks `/api/health`.
- PR touching `terraform/**` -> `terraform-plan.yml` comments the plan on the
  PR automatically.
- Applying a `terraform/**` change -> manually run `terraform-apply.yml` from
  the Actions tab after reviewing the plan comment (see that workflow's
  header comment for why this isn't automatic on merge).

## Observability

Log Analytics + workspace-based Application Insights collect logs and
metrics from both App Services and Postgres (`monitoring.tf`). Baseline
alerts cover backend/dashboard health checks, backend 5xx rate, App Service
Plan CPU/memory, and Postgres CPU/storage. **Alerts have nowhere to go until
you set `alert_notification_emails` in `terraform.tfvars`** — it's an empty
list by default so no personal/team address is hardcoded into version
control. See `to-do.md` for the rest of the infra hardening backlog this is
part of (VNet/private endpoints, Entra ID SSO, Postgres backup/HA).

## Backup / disaster recovery

Postgres keeps 35 days of automated backups, geo-replicated to the region's
paired region by default (`postgres_backup_retention_days` /
`postgres_geo_redundant_backup` in `variables.tf`) — both are cheap to carry.
**Zone-redundant HA is deliberately not enabled**: it requires moving the
Postgres SKU off the Burstable tier, a real cost jump bundled with the App
Service SKU conversation in `to-do.md` A3, not something to default on
silently. `geo_redundant_backup_enabled` forces server recreation if changed
later, so it's set correctly from the first `apply`.

## Network isolation & encryption

Everything data-bearing is private-only (`networking.tf`, `keyvault.tf`,
plus the `public_network_access_enabled` flags in `postgres.tf`/
`storage.tf`/`app_service.tf`):

- One VNet with three subnets: outbound VNet Integration for both App
  Services, a dedicated delegated subnet for Postgres (its supported
  private-access model is subnet injection, not a Private Endpoint), and a
  shared subnet for Private Endpoints (Storage, Key Vault, and the backend's
  *inbound* side).
- **The backend has no public ingress at all** — reachable only from inside
  the VNet (in practice, from the dashboard, which has its own outbound VNet
  Integration into the same subnet). The dashboard itself stays public,
  since browsers need to reach it; its calls to the backend's normal
  `*.azurewebsites.net` hostname resolve privately via the linked private
  DNS zone, no code or config change needed on the dashboard side.
- Postgres and Storage are both `public_network_access_enabled = false`,
  reachable only over the VNet.
- Customer-managed encryption keys (Key Vault, RSA, wrap/unwrap only) for
  both Storage and Postgres, via one shared user-assigned identity
  (`azurerm_user_assigned_identity.cmk`) rather than each resource's own
  system-assigned identity — deliberately, to avoid a chicken-and-egg
  dependency between granting Key Vault access and the resource existing
  yet. The vault itself is also private-only, RBAC-authorized, with purge
  protection on (required for CMK use, and permanent once set).

**Real operational consequences of the backend having no public ingress,**
not just a config flip:
- GitHub Actions' post-deploy check can't curl `/health` from a GitHub-hosted
  runner (no VNet connectivity) — `deploy-backend.yml` now polls ARM for
  `Running` state instead, a weaker signal that the platform brought the app
  up, not that requests are succeeding. If self-hosted runners with VNet
  connectivity get added later, restore a real HTTP health check alongside
  this.
- `az webapp ssh` (used for the one-off FS model bootstrap above) also needs
  VNet connectivity now — see the workaround in that section.
- Debugging the backend directly (`curl`, Postman, browser) from a laptop no
  longer works without a VPN/bastion into the VNet.

## Entra ID SSO

The dashboard requires sign-in via App Service's built-in authentication
("Easy Auth", `auth_settings_v2` in `app_service.tf`) against a dedicated
Entra ID app registration (`auth.tf`) — a platform feature that sits in
front of the container, not an in-app OIDC library. **The backend does not
get its own sign-in**: it has no public ingress at all (A4), so there's no
browser-facing surface on it that needs one; its access control is network
isolation, not authentication.

Reviewer identity for the audit log now comes from the authenticated
principal (`X-MS-CLIENT-PRINCIPAL-NAME`, injected by Easy Auth before the
request reaches the container) rather than a hardcoded constant —
see `empi-dashboard/src/lib/server-api.ts`'s `reviewerId()`. Locally
(`docker-compose`, `npm run dev`), there is no Azure platform in front of
the container, so that header is never present and the code falls back to
the same hardcoded identity it always used — **no local dev workflow change
required.**

## Azure ML

`ml_workspace.tf` provisions an Azure ML Workspace (its own dedicated
storage account, reusing Track A's Key Vault/App Insights/ACR) plus a
small scale-to-zero compute cluster (`cpu-cluster`) — training and
experiment tracking only, no managed online endpoints; serving stays
exactly as today (backend loads a model artifact from `models/fs`/
`models/ml` and scores in-process). Network isolation uses AML's own
*managed* network feature (`AllowOnlyApprovedOutbound`) rather than
injecting a customer VNet, specifically to avoid hand-written NSG rules for
VNet-injected AML compute — a well-documented footgun this review couldn't
verify against a live subscription anyway.

**Consequence:** AML's compute has no network path to the app's own private
storage (different network boundary from `networking.tf`). Moving training
data in and a promoted model out is an explicit staged step, not automatic.

The actual training code, job submission, and environment/dataset
registration live in `empi-model-training/` — see that repo's own
`README.md`. This file only covers the infrastructure.

## Known follow-ups

- **The Postgres AAD administrator is the backend's own managed identity**
  (`postgres.tf`), not a separate least-privileged role — a deliberate
  simplification since only one app talks to this database today. If a
  second, less-trusted caller ever needs access, give it its own AAD role
  via `psql` rather than also making it an admin.
- **Azure Files access key lives in Terraform state** (`app_service.tf`'s
  `storage_account` blocks — App Service's container storage mounts don't
  support managed-identity auth). Restrict who can read this state; consider
  enabling state file encryption if the storage account doesn't already have
  it by default (it does, at rest, via Storage Service Encryption — this is
  about restricting *read access* to the state itself).
- ~~Reviewer identity is hardcoded~~ **Resolved (A4/A5):** the backend has
  no public ingress at all (network isolation is its access control), and
  the dashboard now requires Entra ID SSO (`auth.tf`) — reviewer identity
  comes from the authenticated principal, not a hardcoded constant. See
  "Entra ID SSO" below.
- **`app_service_sku = "P0v3"`** (1 core, 4GB RAM) is sized against the
  backend's pandas/Splink/LightGBM pipeline plus the VNet-integration
  requirement in A4 (`to-do.md`), not a measured production requirement —
  watch the CPU/memory alerts from `monitoring.tf` after a real run and bump
  to `P1v3`/`P2v3` if you see memory pressure or slow cold starts.
