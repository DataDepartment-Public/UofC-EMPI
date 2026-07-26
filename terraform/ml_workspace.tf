# Azure ML Workspace for training + experiment tracking + model registry
# (to-do.md B1) — training and experiment tracking only. Serving stays
# exactly as today: the backend loads a model artifact from the
# `empi-models` Azure Files share (storage.tf) and scores in-process. No
# Azure ML managed online endpoints.
#
# Network isolation uses AML's own *managed* VNet feature
# (managed_network block) rather than injecting a customer-supplied VNet --
# hand-written NSG rules for VNet-injected AML compute are a well-documented
# source of subtle misconfiguration, and AML's managed network gets
# equivalent isolation (AllowOnlyApprovedOutbound: only Azure-approved
# service traffic and explicitly-added private endpoints leave the
# workspace's network) without that risk.
#
# Deliberate consequence of this choice: AML's compute has no network path
# to the app's own private storage (storage.tf's Storage Account, behind its
# own private endpoint in a different network boundary — networking.tf).
# Moving training data in and the promoted model artifact out is an
# explicit, separate step (see empi-model-training/README.md), not
# automatic. In a PHI environment that is arguably the *correct* posture
# anyway -- data crossing a trust boundary should be a deliberate, auditable
# action -- not a reason to fall back to a fragile custom-VNet setup this
# review can't verify against a live Azure subscription.

resource "azurerm_storage_account" "ml" {
  name                = "st${var.project_name}${var.environment}ml${local.suffix}"
  resource_group_name = azurerm_resource_group.main.name
  location            = azurerm_resource_group.main.location

  account_tier             = "Standard"
  account_replication_type = "LRS"
  min_tls_version          = "TLS1_2"

  tags = local.tags
}

resource "azurerm_machine_learning_workspace" "main" {
  name                = "mlw-${local.name_prefix}"
  resource_group_name = azurerm_resource_group.main.name
  location            = azurerm_resource_group.main.location

  sku_name = "Basic"

  # Reuses Track A's Key Vault, Application Insights, and Container
  # Registry -- an AML workspace doesn't need dedicated ones of these, only
  # its own storage account (above), which has a specific folder structure
  # AML manages that isn't a great fit for sharing with the app's storage.
  key_vault_id            = azurerm_key_vault.main.id
  application_insights_id = azurerm_application_insights.main.id
  container_registry_id   = azurerm_container_registry.main.id
  storage_account_id      = azurerm_storage_account.ml.id

  identity {
    type = "SystemAssigned"
  }

  managed_network {
    isolation_mode = "AllowOnlyApprovedOutbound"
  }

  tags = local.tags
}

# Small, scale-to-zero cluster for both training jobs (to-do.md B1) --
# DS3_v2 is a reasonable default for the FS matcher's Splink/DuckDB fit and
# the LightGBM classifier; bump vm_size if a real training run needs more.
resource "azurerm_machine_learning_compute_cluster" "training" {
  name                          = "cpu-cluster"
  location                      = azurerm_resource_group.main.location
  machine_learning_workspace_id = azurerm_machine_learning_workspace.main.id
  vm_size                       = "Standard_DS3_v2"
  vm_priority                   = "Dedicated"

  scale_settings {
    min_node_count                       = 0
    max_node_count                       = 2
    scale_down_nodes_after_idle_duration = "PT15M"
  }

  identity {
    type = "SystemAssigned"
  }

  tags = local.tags
}

# ---- RBAC: what the workspace, compute, and CI actually need to work -------
#
# Without these, job submission fails opaquely (workspace can't write its
# own metadata/logs, compute can't pull the environment image or read/write
# training data, CI can't touch AML at all beyond what its broad RG-Owner
# grant already covers implicitly).

resource "azurerm_role_assignment" "ml_workspace_storage" {
  scope                = azurerm_storage_account.ml.id
  role_definition_name = "Storage Blob Data Contributor"
  principal_id         = azurerm_machine_learning_workspace.main.identity[0].principal_id
}

resource "azurerm_role_assignment" "ml_workspace_acr_pull" {
  scope                = azurerm_container_registry.main.id
  role_definition_name = "AcrPull"
  principal_id         = azurerm_machine_learning_workspace.main.identity[0].principal_id
}

resource "azurerm_role_assignment" "ml_workspace_keyvault" {
  scope                = azurerm_key_vault.main.id
  role_definition_name = "Key Vault Secrets Officer"
  principal_id         = azurerm_machine_learning_workspace.main.identity[0].principal_id
}

resource "azurerm_role_assignment" "ml_compute_storage" {
  scope                = azurerm_storage_account.ml.id
  role_definition_name = "Storage Blob Data Contributor"
  principal_id         = azurerm_machine_learning_compute_cluster.training.identity[0].principal_id
}

resource "azurerm_role_assignment" "ml_compute_acr_pull" {
  scope                = azurerm_container_registry.main.id
  role_definition_name = "AcrPull"
  principal_id         = azurerm_machine_learning_compute_cluster.training.identity[0].principal_id
}

# The GitHub Actions CI identity (github_oidc.tf) already has Owner on this
# resource group, which technically covers AML data-plane actions too --
# this is explicit for the same reason github_oidc.tf's AcrPush grant is:
# auditability independent of the broader RG grant, for the one workflow
# (to-do.md B4, empi-model-training's champion-promotion GitHub Action) that
# actually needs to touch AML.
resource "azurerm_role_assignment" "github_actions_ml_data_scientist" {
  scope                = azurerm_machine_learning_workspace.main.id
  role_definition_name = "AzureML Data Scientist"
  principal_id         = azuread_service_principal.github_actions.object_id
}
