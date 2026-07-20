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

resource "azurerm_storage_account" "main" {
  name                = "st${var.project_name}${var.environment}${local.suffix}"
  resource_group_name = azurerm_resource_group.main.name
  location            = azurerm_resource_group.main.location

  account_tier               = "Standard"
  account_replication_type   = "LRS"
  min_tls_version            = "TLS1_2"
  https_traffic_only_enabled = true

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
