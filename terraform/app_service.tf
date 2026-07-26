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
#
# public_network_access_enabled = false + virtual_network_subnet_id together:
# no public ingress at all (only reachable via the inbound Private Endpoint
# in networking.tf, e.g. from the dashboard's own VNet integration below),
# and outbound traffic to Postgres/Storage routes over the VNet to their
# private endpoints instead of the public internet. See to-do.md A4 --
# this also means GitHub Actions' post-deploy health check can no longer
# curl this app's public URL directly (see deploy-backend.yml).

resource "azurerm_linux_web_app" "backend" {
  name                = local.backend_app_name
  resource_group_name = azurerm_resource_group.main.name
  location            = azurerm_service_plan.main.location
  service_plan_id     = azurerm_service_plan.main.id

  public_network_access_enabled = false
  virtual_network_subnet_id     = azurerm_subnet.app.id

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

    APPLICATIONINSIGHTS_CONNECTION_STRING = azurerm_application_insights.main.connection_string
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
#
# Stays publicly reachable (browsers need to reach it) -- but does get
# outbound VNet Integration into the same subnet as the backend, which is
# what lets its calls to https://<backend-hostname> (unchanged, still the
# public-looking *.azurewebsites.net URL) resolve to the backend's private
# IP instead of failing now that the backend has no public ingress. No code
# change needed on the dashboard side for this.

resource "azurerm_linux_web_app" "dashboard" {
  name                = local.dashboard_app_name
  resource_group_name = azurerm_resource_group.main.name
  location            = azurerm_service_plan.main.location
  service_plan_id     = azurerm_service_plan.main.id

  virtual_network_subnet_id = azurerm_subnet.app.id

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

    APPLICATIONINSIGHTS_CONNECTION_STRING = azurerm_application_insights.main.connection_string

    # Read by auth_settings_v2.active_directory_v2.client_secret_setting_name
    # below -- Easy Auth's documented convention is to read the provider
    # secret from an app setting by name, not accept it as a plan-visible
    # attribute directly.
    MICROSOFT_PROVIDER_AUTHENTICATION_SECRET = azuread_application_password.dashboard.value
  }

  # Entra ID SSO (auth.tf) -- see that file's header comment for why this is
  # on the dashboard only, not the backend.
  auth_settings_v2 {
    auth_enabled           = true
    default_provider       = "azureactivedirectory"
    unauthenticated_action = "RedirectToLoginPage"
    require_authentication = true

    active_directory_v2 {
      client_id                  = azuread_application.dashboard.client_id
      client_secret_setting_name = "MICROSOFT_PROVIDER_AUTHENTICATION_SECRET"
      tenant_auth_endpoint       = "https://login.microsoftonline.com/${data.azurerm_client_config.current.tenant_id}/v2.0"
    }

    login {
      token_store_enabled = true
    }
  }

  tags = local.tags

  lifecycle {
    ignore_changes = [site_config[0].application_stack]
  }
}
