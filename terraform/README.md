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

- No private networking / VNet integration — the backend and dashboard both
  get public HTTPS endpoints, with CORS on the backend locked to the
  dashboard's own origin. This matches the "synthetic/de-identified data,
  standard security baseline" decision behind this design. If real PHI ends
  up in this environment, revisit: private endpoints on Postgres/Storage, no
  public ingress on the backend, customer-managed encryption keys, and
  probably a different App Service tier (Premium v3, for VNet integration).
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
terraform plan   # review — this creates the resource group, ACR, both App
                  # Services, the Postgres Flexible Server, storage account,
                  # and the GitHub OIDC app registration + role assignments
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

Optionally, create a `production` GitHub Environment (Settings ->
Environments) with required reviewers if you want a manual-approval gate on
`terraform-apply.yml` — it already targets `environment: production`, so
adding protection rules there is enough; no workflow change needed.

### 5. First deploy

Push to `main` (or re-run `deploy-backend.yml` / `deploy-dashboard.yml` via
`workflow_dispatch`) to build and push the images and point the App Services
at them. The Terraform-provisioned apps start on a `:latest` placeholder tag
that doesn't exist in your ACR yet, so **the App Services will show
unhealthy until the first successful deploy workflow run** — expected, not a
bug.

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

## Day-to-day

- Push (or merge a PR) touching `empi-service/**` -> `deploy-backend.yml`
  builds, pushes to ACR, updates the App Service, waits for `/health`.
- Push touching `empi-dashboard/**` -> `deploy-dashboard.yml`, same shape,
  checks `/api/health`.
- PR touching `terraform/**` -> `terraform-plan.yml` comments the plan on the
  PR automatically.
- Applying a `terraform/**` change -> manually run `terraform-apply.yml` from
  the Actions tab after reviewing the plan comment (see that workflow's
  header comment for why this isn't automatic on merge).

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
- **Reviewer identity is hardcoded** (`reviewer.jclark` in
  `empi-dashboard/src/lib/server-api.ts`) — no real auth in front of either
  app today. Out of scope for this infra pass; flagging so it doesn't get
  mistaken for a deliberate access-control decision.
- **`app_service_sku = "B1"`** (1 core, 1.75GB RAM) is a starting point, not
  a measured requirement — the backend's pandas/splink pipeline can be
  memory-hungry on larger batches. Watch App Service metrics after a real
  run and bump the SKU (`P0v3`/`P1v3`) if you see memory pressure or slow
  cold starts.
