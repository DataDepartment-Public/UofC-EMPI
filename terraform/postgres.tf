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
# Private access via VNet injection into a dedicated delegated subnet
# (networking.tf's snet-postgres) -- Postgres Flexible Server's supported
# private-connectivity model is delegated-subnet injection, not a standard
# Private Endpoint (that's what Storage/the backend use instead; see
# networking.tf). public_network_access_enabled = false plus this means the
# database is reachable only from inside the VNet (i.e. from the backend,
# which has its own outbound VNet Integration -- app_service.tf). The old
# "allow all Azure services" firewall rule is gone entirely: firewall rules
# only apply in public-access mode, and don't exist in this mode at all.
#
# NOTE: delegated_subnet_id, private_dns_zone_id, and
# geo_redundant_backup_enabled all force server recreation if changed after
# the fact -- get this right before the first `terraform apply`.
#
# Zone-redundant HA is deliberately NOT enabled here: it requires moving off
# the Burstable SKU tier (var.postgres_sku) to General Purpose or better —
# Burstable doesn't support the `high_availability` block at all — which is
# roughly a 10x compute-cost jump before HA's own ~2x on top of that. That's
# a decision to make alongside the App Service SKU bump (see to-do.md A3),
# not something to default on silently. geo_redundant_backup_enabled below
# is a much cheaper way to get disaster-recovery coverage in the meantime.

resource "azurerm_postgresql_flexible_server" "main" {
  name                = "psql-${local.name_prefix}-${local.suffix}"
  resource_group_name = azurerm_resource_group.main.name
  location            = azurerm_resource_group.main.location

  version    = var.postgres_version
  sku_name   = var.postgres_sku
  storage_mb = var.postgres_storage_mb

  backup_retention_days        = var.postgres_backup_retention_days
  geo_redundant_backup_enabled = var.postgres_geo_redundant_backup

  delegated_subnet_id           = azurerm_subnet.postgres.id
  private_dns_zone_id           = azurerm_private_dns_zone.postgres.id
  public_network_access_enabled = false

  authentication {
    active_directory_auth_enabled = true
    password_auth_enabled         = false
    tenant_id                     = data.azurerm_client_config.current.tenant_id
  }

  # Customer-managed key (keyvault.tf) via the shared CMK identity -- same
  # pattern and same reasoning as the storage account's identity block.
  identity {
    type         = "UserAssigned"
    identity_ids = [azurerm_user_assigned_identity.cmk.id]
  }

  customer_managed_key {
    key_vault_key_id                  = azurerm_key_vault_key.postgres.id
    primary_user_assigned_identity_id = azurerm_user_assigned_identity.cmk.id
  }

  tags = local.tags

  # Azure requires the private DNS zone to already be linked to the target
  # VNet before a delegated-subnet server can be created in it -- Terraform
  # has no other way to infer that ordering from the attribute references
  # above alone.
  depends_on = [azurerm_private_dns_zone_virtual_network_link.postgres]
}

resource "azurerm_postgresql_flexible_server_active_directory_administrator" "backend" {
  server_name         = azurerm_postgresql_flexible_server.main.name
  resource_group_name = azurerm_resource_group.main.name
  tenant_id           = data.azurerm_client_config.current.tenant_id
  object_id           = azurerm_linux_web_app.backend.identity[0].principal_id
  principal_name      = local.backend_app_name
  principal_type      = "ServicePrincipal"
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
