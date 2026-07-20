resource "azurerm_service_plan" "main" {
  name                = "asp-${local.name_prefix}"
  resource_group_name = azurerm_resource_group.main.name
  location            = azurerm_resource_group.main.location
  os_type             = "Linux"
  sku_name            = var.app_service_sku

  tags = local.tags
}

# ---- Backend (empi-service, FastAPI, port 8000) ----------------------------
#
# Single instance is a hard requirement today, not a cost choice: the job
# registry backing GET /runs/{id} is an in-process dict (empi-service/src/api/
# jobs.py), so a second replica or worker would silently return 404s for runs
# it didn't start. Do not add autoscale/instance-count > 1 without first
# replacing that registry with a shared store (e.g. Redis).

resource "azurerm_linux_web_app" "backend" {
  name                = local.backend_app_name
  resource_group_name = azurerm_resource_group.main.name
  location            = azurerm_service_plan.main.location
  service_plan_id     = azurerm_service_plan.main.id

  identity {
    type = "SystemAssigned"
  }

  site_config {
    always_on         = true
    health_check_path = "/health"

    application_stack {
      # Placeholder tag — Terraform only sets the *initial* image so the app
      # exists; every real deploy updates the tag via `az webapp config
      # container set` in .github/workflows/deploy-backend.yml, which is why
      # this attribute is in lifecycle.ignore_changes below.
      docker_image_name = "${azurerm_container_registry.main.login_server}/empi-service:latest"
    }

    container_registry_use_managed_identity = true
  }

  app_settings = {
    WEBSITES_PORT = "8000"

    # Dashboard is a server-side BFF (src/lib/server-api.ts) — browsers never
    # call this API directly — so in the common case this is the only origin
    # that needs to be allowed. Pydantic-settings parses list[str] env vars as
    # JSON.
    EMPI_API_CORS_ORIGINS = jsonencode(concat(
      ["https://${local.dashboard_hostname}"],
      var.backend_extra_cors_origins,
    ))

    # Postgres connection — no password: postgres_backend.py authenticates
    # with an AAD token for this app's own managed identity (see postgres.tf).
    EMPI_INDEX_BACKEND = "postgres"
    EMPI_POSTGRES_HOST = azurerm_postgresql_flexible_server.main.fqdn
    EMPI_POSTGRES_DB   = azurerm_postgresql_flexible_server_database.main.name
    EMPI_POSTGRES_USER = local.backend_app_name
  }

  storage_account {
    name         = "data"
    type         = "AzureFiles"
    account_name = azurerm_storage_account.main.name
    share_name   = azurerm_storage_share.data.name
    access_key   = azurerm_storage_account.main.primary_access_key
    mount_path   = "/app/data"
  }

  storage_account {
    name         = "models"
    type         = "AzureFiles"
    account_name = azurerm_storage_account.main.name
    share_name   = azurerm_storage_share.models.name
    access_key   = azurerm_storage_account.main.primary_access_key
    mount_path   = "/app/models"
  }

  storage_account {
    name         = "logs"
    type         = "AzureFiles"
    account_name = azurerm_storage_account.main.name
    share_name   = azurerm_storage_share.logs.name
    access_key   = azurerm_storage_account.main.primary_access_key
    mount_path   = "/app/logs"
  }

  tags = local.tags

  lifecycle {
    ignore_changes = [site_config[0].application_stack]
  }
}

# ---- Dashboard (empi-dashboard, Next.js standalone, port 3000) -------------
#
# Stateless BFF — no storage mount. Only configured input is the backend's
# address (src/lib/server-api.ts's EMPI_API_URL).

resource "azurerm_linux_web_app" "dashboard" {
  name                = local.dashboard_app_name
  resource_group_name = azurerm_resource_group.main.name
  location            = azurerm_service_plan.main.location
  service_plan_id     = azurerm_service_plan.main.id

  identity {
    type = "SystemAssigned"
  }

  site_config {
    always_on         = true
    health_check_path = "/api/health"

    application_stack {
      docker_image_name = "${azurerm_container_registry.main.login_server}/empi-dashboard:latest"
    }

    container_registry_use_managed_identity = true
  }

  app_settings = {
    WEBSITES_PORT = "3000"
    EMPI_API_URL  = "https://${local.backend_hostname}"
  }

  tags = local.tags

  lifecycle {
    ignore_changes = [site_config[0].application_stack]
  }
}
