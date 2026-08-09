# Entra ID (Azure AD) SSO for the dashboard, via App Service's built-in
# authentication ("Easy Auth", auth_settings_v2) -- a platform feature that
# sits in front of the container, not an in-app OIDC library. Chosen
# specifically so local dev needs zero changes: there's no Azure platform in
# front of the container outside Azure, so `docker-compose up`/`npm run dev`
# keep using the dashboard's existing hardcoded-reviewer fallback
# (empi-dashboard/src/lib/server-api.ts) untouched. See to-do.md A5.
#
# The backend does NOT get its own Easy Auth registration -- it has no
# public ingress at all as of A4 (networking.tf), so there's no
# browser-facing surface on it that needs a login flow. Its access control
# is network isolation, not authentication; the dashboard is the only place
# a human actually signs in.

resource "azuread_application" "dashboard" {
  display_name     = "empi-${var.environment}-dashboard"
  sign_in_audience = "AzureADMyOrg"

  web {
    redirect_uris = ["https://${local.dashboard_hostname}/.auth/login/aad/callback"]
  }
}

# Required for the app registration to actually be usable for sign-in --
# azuread_application alone (an App Registration) doesn't create the
# corresponding Enterprise Application object a tenant needs to authorize
# sign-ins against. Mirrors the same two-resource pattern github_oidc.tf
# already uses for the CI identity.
resource "azuread_service_principal" "dashboard" {
  client_id = azuread_application.dashboard.client_id
}

resource "azuread_application_password" "dashboard" {
  application_id = "/applications/${azuread_application.dashboard.object_id}"
  display_name   = "app-service-easy-auth"
}
