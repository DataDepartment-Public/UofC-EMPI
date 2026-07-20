# Azure Database for PostgreSQL Flexible Server — the target for the
# CosmosIndexBackend-that-almost-was; see empi-service/src/api/backends/
# postgres_backend.py for why relational fits this schema (FK joins,
# multi-table transactions, ON CONFLICT upserts) much better than a document
# store. Not wired into the running app until EMPI_INDEX_BACKEND=postgres is
# set (see app_service.tf) and empi-service ships that migration.
#
# AAD-only auth, no stored password: the backend App Service's own
# system-assigned identity is registered as the server's AAD administrator,
# so it can connect with a token from DefaultAzureCredential instead of a
# secret. This is a deliberate simplification — the app's identity being the
# *admin* (rather than a separate, lesser-privileged AAD role) avoids a
# `CREATE ROLE ... FROM EXTERNAL PROVIDER` SQL step that Terraform's azurerm
# provider can't issue directly (it needs to run over a DB connection, not
# ARM). Reasonable for "one app, one database"; if a second, less-trusted
# caller ever needs DB access, create it a narrower AAD role via psql instead
# of also making it an admin.
#
# Public network access + "allow all Azure services" firewall rule mirrors
# the no-VNet/standard-baseline tradeoff used everywhere else in this config
# (see terraform/README.md) — AAD auth is what actually gates who can log in.

resource "azurerm_postgresql_flexible_server" "main" {
  name                = "psql-${local.name_prefix}-${local.suffix}"
  resource_group_name = azurerm_resource_group.main.name
  location            = azurerm_resource_group.main.location

  version    = var.postgres_version
  sku_name   = var.postgres_sku
  storage_mb = var.postgres_storage_mb

  public_network_access_enabled = true

  authentication {
    active_directory_auth_enabled = true
    password_auth_enabled         = false
    tenant_id                     = data.azurerm_client_config.current.tenant_id
  }

  tags = local.tags
}

resource "azurerm_postgresql_flexible_server_active_directory_administrator" "backend" {
  server_name         = azurerm_postgresql_flexible_server.main.name
  resource_group_name = azurerm_resource_group.main.name
  tenant_id           = data.azurerm_client_config.current.tenant_id
  object_id           = azurerm_linux_web_app.backend.identity[0].principal_id
  principal_name      = local.backend_app_name
  principal_type      = "ServicePrincipal"
}

resource "azurerm_postgresql_flexible_server_firewall_rule" "allow_azure_services" {
  name             = "allow-azure-services"
  server_id        = azurerm_postgresql_flexible_server.main.id
  start_ip_address = "0.0.0.0"
  end_ip_address   = "0.0.0.0"
}

resource "azurerm_postgresql_flexible_server_database" "main" {
  name      = var.project_name
  server_id = azurerm_postgresql_flexible_server.main.id
  collation = "en_US.utf8"
  charset   = "UTF8"

  lifecycle {
    prevent_destroy = true
  }
}
