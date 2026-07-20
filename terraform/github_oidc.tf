# GitHub Actions <-> Azure trust via OIDC (Workload Identity Federation) — no
# client secret is ever stored as a GitHub Actions secret; GitHub's own OIDC
# token is exchanged for an Azure AD token at workflow run time.
#
# IMPORTANT — bootstrap ordering: creating an App Registration and federated
# credential requires an Azure AD *directory* role (e.g. Application
# Administrator or Cloud Application Administrator), which is separate from
# and not implied by any Azure *subscription* role (Owner/Contributor). The
# person running `terraform apply` for the first time needs both. After that
# first apply, the CI identity created here (github_actions) does NOT have
# permission to modify its own App Registration or federated credentials —
# by design, so a compromised/broadened CI workflow can't grant itself more
# trust. Routine `terraform plan`/`apply` runs from CI (see
# .github/workflows/terraform-*.yml) will still show these resources with no
# changes; if you ever need to add a trusted branch/environment here, that
# specific apply has to be run by a human with the Azure AD role, same as the
# initial bootstrap. See terraform/README.md.

resource "azuread_application" "github_actions" {
  display_name = "github-${replace(var.github_repository, "/", "-")}-${var.environment}"
}

resource "azuread_service_principal" "github_actions" {
  client_id = azuread_application.github_actions.client_id
}

resource "azuread_application_federated_identity_credential" "github_actions" {
  for_each = toset(var.github_oidc_subjects)

  application_id = "/applications/${azuread_application.github_actions.object_id}"
  display_name   = "github-${replace(each.value, "/", "-")}"
  description    = "Trusts GitHub Actions runs for ${var.github_repository}:${each.value}"
  audiences      = ["api://AzureADTokenExchange"]
  issuer         = "https://token.actions.githubusercontent.com"
  subject        = "repo:${var.github_repository}:${each.value}"
}

# Scoped to this one resource group only (never subscription-wide) — Owner
# rather than Contributor because this config also manages role assignments
# within the RG (ACR pull for the App Services, Cosmos data-plane access for
# the backend), and Contributor alone cannot create/modify RBAC. Tightening
# this to a narrower custom role is a reasonable future hardening step; Owner
# scoped to a single purpose-built RG is the pragmatic tradeoff for now.
resource "azurerm_role_assignment" "github_actions_rg_owner" {
  scope                = azurerm_resource_group.main.id
  role_definition_name = "Owner"
  principal_id         = azuread_service_principal.github_actions.object_id
}

# Explicit (if redundant with the Owner grant above) so `docker push`/`az acr
# login` access is auditable independent of the broader RG grant.
resource "azurerm_role_assignment" "github_actions_acr_push" {
  scope                = azurerm_container_registry.main.id
  role_definition_name = "AcrPush"
  principal_id         = azuread_service_principal.github_actions.object_id
}
