resource "azurerm_container_registry" "main" {
  name                = "acr${var.project_name}${var.environment}${local.suffix}"
  resource_group_name = azurerm_resource_group.main.name
  location            = azurerm_resource_group.main.location
  sku                 = "Basic"

  # No admin username/password: App Service pulls via each app's managed identity
  # (see app_service.tf), and CI pushes via the GitHub OIDC identity's AcrPush role
  # (see github_oidc.tf). Nothing needs a stored registry credential.
  admin_enabled = false

  tags = local.tags
}
