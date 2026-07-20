output "resource_group_name" {
  value = azurerm_resource_group.main.name
}

output "backend_url" {
  description = "Public HTTPS URL of the empi-service (FastAPI) App Service."
  value       = "https://${local.backend_hostname}"
}

output "dashboard_url" {
  description = "Public HTTPS URL of the empi-dashboard (Next.js) App Service — this is the one to hand out to reviewers."
  value       = "https://${local.dashboard_hostname}"
}

output "container_registry_login_server" {
  value = azurerm_container_registry.main.login_server
}

output "postgres_fqdn" {
  description = "Postgres Flexible Server hostname. Unused by the running app until EMPI_INDEX_BACKEND=postgres is set and empi-service ships that backend — see postgres.tf and app_service.tf."
  value       = azurerm_postgresql_flexible_server.main.fqdn
}

output "storage_account_name" {
  value = azurerm_storage_account.main.name
}

# ---- Values needed as GitHub Actions repo secrets/variables ----------------
# See terraform/README.md "Wire up GitHub Actions" for exactly where these go.

output "github_actions_client_id" {
  description = "-> GitHub secret AZURE_CLIENT_ID"
  value       = azuread_application.github_actions.client_id
}

output "github_actions_tenant_id" {
  description = "-> GitHub secret AZURE_TENANT_ID"
  value       = data.azurerm_client_config.current.tenant_id
}

output "github_actions_subscription_id" {
  description = "-> GitHub secret AZURE_SUBSCRIPTION_ID"
  value       = data.azurerm_client_config.current.subscription_id
}
