# Persistent storage for the backend's SQLite file (data/empi.db, ~500MB in
# production-scale runs per the repo's own to-do notes), the FS/ML model
# artifacts (models/fs, models/ml), and pipeline intermediate output
# (data/raw, data/processed, etc.) — all runtime state the Dockerfile
# deliberately does not bake into the image (see empi-service/Dockerfile
# header comment). Mirrors docker-compose.yml's three named volumes.
#
# App Service's Azure Files mount for Linux containers authenticates with the
# storage account key, not a managed identity — that key ends up in Terraform
# state as a result. Make sure the backend (backend.hcl) points at a storage
# account with encryption-at-rest and restrict who can read this state.
#
# Public network access is off -- reachable only via the Private Endpoint in
# networking.tf (pe-*-storage-file), from the backend's VNet-integrated
# outbound path (app_service.tf). Account-key auth for the SMB mount is
# unaffected by this; it's independent of which network path reaches the
# endpoint.

resource "azurerm_storage_account" "main" {
  name                = "st${var.project_name}${var.environment}${local.suffix}"
  resource_group_name = azurerm_resource_group.main.name
  location            = azurerm_resource_group.main.location

  account_tier               = "Standard"
  account_replication_type   = "LRS"
  min_tls_version            = "TLS1_2"
  https_traffic_only_enabled = true

  public_network_access_enabled = false

  network_rules {
    default_action = "Deny"
    bypass         = ["AzureServices"]
  }

  # Customer-managed key (keyvault.tf) via the shared CMK identity -- using a
  # user-assigned identity here (not this resource's own system-assigned
  # one) is what avoids a chicken-and-egg dependency between the key vault
  # role grant and the storage account's own identity; see keyvault.tf.
  identity {
    type         = "UserAssigned"
    identity_ids = [azurerm_user_assigned_identity.cmk.id]
  }

  customer_managed_key {
    key_vault_key_id          = azurerm_key_vault_key.storage.id
    user_assigned_identity_id = azurerm_user_assigned_identity.cmk.id
  }

  tags = local.tags
}

resource "azurerm_storage_share" "data" {
  name               = "empi-data"
  storage_account_id = azurerm_storage_account.main.id
  quota              = 50 # GB — data/empi.db plus pipeline intermediates
}

resource "azurerm_storage_share" "models" {
  name               = "empi-models"
  storage_account_id = azurerm_storage_account.main.id
  quota              = 20 # GB — FS/ML matcher artifacts
}

resource "azurerm_storage_share" "logs" {
  name               = "empi-logs"
  storage_account_id = azurerm_storage_account.main.id
  quota              = 10 # GB
}
