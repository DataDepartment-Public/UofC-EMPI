resource "random_string" "suffix" {
  length  = 4
  special = false
  upper   = false
  numeric = true
}

locals {
  name_prefix = "${var.project_name}-${var.environment}"
  # Globally-unique Azure resources (ACR, Storage Account, Cosmos, App Service
  # hostnames) can't use hyphens or can't be guaranteed collision-free across all of
  # Azure by name alone, so they get this suffix appended.
  suffix = random_string.suffix.result
  tags   = var.tags

  # Public Azure App Service hostnames are deterministic (<name>.azurewebsites.net),
  # so the backend and dashboard apps can each be told the other's URL via these
  # locals instead of referencing each other's `default_hostname` attribute directly
  # — that cross-reference would make the two azurerm_linux_web_app resources depend
  # on each other and Terraform would reject it as a dependency cycle.
  backend_app_name   = "app-${local.name_prefix}-backend-${local.suffix}"
  dashboard_app_name = "app-${local.name_prefix}-dashboard-${local.suffix}"
  backend_hostname   = "${local.backend_app_name}.azurewebsites.net"
  dashboard_hostname = "${local.dashboard_app_name}.azurewebsites.net"
}

data "azurerm_client_config" "current" {}

resource "azurerm_resource_group" "main" {
  name     = "rg-${local.name_prefix}"
  location = var.location
  tags     = local.tags
}
