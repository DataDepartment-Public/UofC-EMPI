# RBAC for the App Services' own system-assigned identities (created in
# app_service.tf). Split out here so the "who can access what" story reads in one
# place instead of being scattered across each resource file.

resource "azurerm_role_assignment" "backend_acr_pull" {
  scope                = azurerm_container_registry.main.id
  role_definition_name = "AcrPull"
  principal_id         = azurerm_linux_web_app.backend.identity[0].principal_id
}

resource "azurerm_role_assignment" "dashboard_acr_pull" {
  scope                = azurerm_container_registry.main.id
  role_definition_name = "AcrPull"
  principal_id         = azurerm_linux_web_app.dashboard.identity[0].principal_id
}
