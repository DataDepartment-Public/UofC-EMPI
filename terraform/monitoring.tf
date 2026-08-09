# Observability: nothing here existed before this file — the original
# Terraform pass shipped with no monitoring or alerting at all (see
# to-do.md A1). Log Analytics + workspace-based Application Insights collect
# logs/metrics from both App Services and Postgres; a small set of baseline
# metric alerts cover the failure modes this repo's own comments already
# flag as real risks (App Service Plan memory headroom against the
# pandas/Splink pipeline, the single-instance backend's health).
#
# Purely additive: no app code requires APPLICATIONINSIGHTS_CONNECTION_STRING
# to be set, and none of this exists in local/docker-compose dev.

resource "azurerm_log_analytics_workspace" "main" {
  name                = "log-${local.name_prefix}-${local.suffix}"
  resource_group_name = azurerm_resource_group.main.name
  location            = azurerm_resource_group.main.location
  sku                 = "PerGB2018"
  retention_in_days   = var.log_retention_days

  tags = local.tags
}

resource "azurerm_application_insights" "main" {
  name                = "appi-${local.name_prefix}-${local.suffix}"
  resource_group_name = azurerm_resource_group.main.name
  location            = azurerm_resource_group.main.location
  workspace_id        = azurerm_log_analytics_workspace.main.id
  application_type    = "web"

  tags = local.tags
}

# Empty by default (see variables.tf) so no personal/team email is
# hardcoded into version control — alerts fire into this action group either
# way, they just have nowhere to go until terraform.tfvars sets at least one
# address.
resource "azurerm_monitor_action_group" "ops" {
  name                = "ag-${local.name_prefix}-ops"
  resource_group_name = azurerm_resource_group.main.name
  short_name          = "empiops"

  dynamic "email_receiver" {
    for_each = { for idx, addr in var.alert_notification_emails : idx => addr }
    content {
      name          = "email-${email_receiver.key}"
      email_address = email_receiver.value
    }
  }

  tags = local.tags
}

# ---- Diagnostic settings: ship logs + metrics to Log Analytics ------------

resource "azurerm_monitor_diagnostic_setting" "backend" {
  name                       = "diag-backend"
  target_resource_id         = azurerm_linux_web_app.backend.id
  log_analytics_workspace_id = azurerm_log_analytics_workspace.main.id

  enabled_log {
    category_group = "allLogs"
  }

  enabled_metric {
    category = "AllMetrics"
  }
}

resource "azurerm_monitor_diagnostic_setting" "dashboard" {
  name                       = "diag-dashboard"
  target_resource_id         = azurerm_linux_web_app.dashboard.id
  log_analytics_workspace_id = azurerm_log_analytics_workspace.main.id

  enabled_log {
    category_group = "allLogs"
  }

  enabled_metric {
    category = "AllMetrics"
  }
}

resource "azurerm_monitor_diagnostic_setting" "postgres" {
  name                       = "diag-postgres"
  target_resource_id         = azurerm_postgresql_flexible_server.main.id
  log_analytics_workspace_id = azurerm_log_analytics_workspace.main.id

  enabled_log {
    category_group = "allLogs"
  }

  enabled_metric {
    category = "AllMetrics"
  }
}

# ---- Baseline alerts --------------------------------------------------------

resource "azurerm_monitor_metric_alert" "backend_5xx" {
  name                = "alert-backend-5xx"
  resource_group_name = azurerm_resource_group.main.name
  scopes              = [azurerm_linux_web_app.backend.id]
  description         = "Backend (empi-service) is returning server errors."
  severity            = 2
  frequency           = "PT5M"
  window_size         = "PT5M"

  criteria {
    metric_namespace = "Microsoft.Web/sites"
    metric_name      = "Http5xx"
    aggregation      = "Total"
    operator         = "GreaterThan"
    threshold        = 5
  }

  action {
    action_group_id = azurerm_monitor_action_group.ops.id
  }

  tags = local.tags
}

resource "azurerm_monitor_metric_alert" "backend_health" {
  name                = "alert-backend-health"
  resource_group_name = azurerm_resource_group.main.name
  scopes              = [azurerm_linux_web_app.backend.id]
  description         = "Backend /health check is failing."
  severity            = 0
  frequency           = "PT5M"
  window_size         = "PT5M"

  criteria {
    metric_namespace = "Microsoft.Web/sites"
    metric_name      = "HealthCheckStatus"
    aggregation      = "Average"
    operator         = "LessThan"
    threshold        = 1
  }

  action {
    action_group_id = azurerm_monitor_action_group.ops.id
  }

  tags = local.tags
}

resource "azurerm_monitor_metric_alert" "dashboard_health" {
  name                = "alert-dashboard-health"
  resource_group_name = azurerm_resource_group.main.name
  scopes              = [azurerm_linux_web_app.dashboard.id]
  description         = "Dashboard /api/health check is failing."
  severity            = 0
  frequency           = "PT5M"
  window_size         = "PT5M"

  criteria {
    metric_namespace = "Microsoft.Web/sites"
    metric_name      = "HealthCheckStatus"
    aggregation      = "Average"
    operator         = "LessThan"
    threshold        = 1
  }

  action {
    action_group_id = azurerm_monitor_action_group.ops.id
  }

  tags = local.tags
}

resource "azurerm_monitor_metric_alert" "plan_cpu" {
  name                = "alert-plan-cpu"
  resource_group_name = azurerm_resource_group.main.name
  scopes              = [azurerm_service_plan.main.id]
  description         = "App Service Plan CPU is running hot."
  severity            = 2
  frequency           = "PT5M"
  window_size         = "PT15M"

  criteria {
    metric_namespace = "Microsoft.Web/serverfarms"
    metric_name      = "CpuPercentage"
    aggregation      = "Average"
    operator         = "GreaterThan"
    threshold        = 80
  }

  action {
    action_group_id = azurerm_monitor_action_group.ops.id
  }

  tags = local.tags
}

resource "azurerm_monitor_metric_alert" "plan_memory" {
  name                = "alert-plan-memory"
  resource_group_name = azurerm_resource_group.main.name
  scopes              = [azurerm_service_plan.main.id]
  description         = <<-EOT
    App Service Plan memory is running hot -- the pandas/Splink/LightGBM
    pipeline is the likely cause. A sustained trip here is the signal to
    bump app_service_sku (see variables.tf and to-do.md A3), not just noise.
  EOT
  severity            = 1
  frequency           = "PT5M"
  window_size         = "PT15M"

  criteria {
    metric_namespace = "Microsoft.Web/serverfarms"
    metric_name      = "MemoryPercentage"
    aggregation      = "Average"
    operator         = "GreaterThan"
    threshold        = 85
  }

  action {
    action_group_id = azurerm_monitor_action_group.ops.id
  }

  tags = local.tags
}

resource "azurerm_monitor_metric_alert" "postgres_cpu" {
  name                = "alert-postgres-cpu"
  resource_group_name = azurerm_resource_group.main.name
  scopes              = [azurerm_postgresql_flexible_server.main.id]
  description         = "Postgres CPU is running hot."
  severity            = 2
  frequency           = "PT5M"
  window_size         = "PT15M"

  criteria {
    metric_namespace = "Microsoft.DBforPostgreSQL/flexibleServers"
    metric_name      = "cpu_percent"
    aggregation      = "Average"
    operator         = "GreaterThan"
    threshold        = 80
  }

  action {
    action_group_id = azurerm_monitor_action_group.ops.id
  }

  tags = local.tags
}

resource "azurerm_monitor_metric_alert" "postgres_storage" {
  name                = "alert-postgres-storage"
  resource_group_name = azurerm_resource_group.main.name
  scopes              = [azurerm_postgresql_flexible_server.main.id]
  description         = "Postgres storage is filling up."
  severity            = 1
  frequency           = "PT15M"
  window_size         = "PT30M"

  criteria {
    metric_namespace = "Microsoft.DBforPostgreSQL/flexibleServers"
    metric_name      = "storage_percent"
    aggregation      = "Average"
    operator         = "GreaterThan"
    threshold        = 80
  }

  action {
    action_group_id = azurerm_monitor_action_group.ops.id
  }

  tags = local.tags
}
