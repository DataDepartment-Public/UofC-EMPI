variable "project_name" {
  description = "Short name used as a prefix for all resource names. Must be lowercase alphanumeric (no hyphens) so it can also feed globally-unique names like the ACR and storage account."
  type        = string
  default     = "empi"

  validation {
    condition     = can(regex("^[a-z0-9]+$", var.project_name))
    error_message = "project_name must be lowercase letters and digits only."
  }
}

variable "environment" {
  description = "Deployment environment name, used in resource naming (e.g. \"prod\", \"staging\")."
  type        = string
  default     = "prod"
}

variable "location" {
  description = "Azure region for all resources."
  type        = string
  default     = "canadacentral"
}

variable "github_repository" {
  description = "GitHub repo in \"owner/name\" form, used to scope the OIDC federated credential trust so only workflows in this repo can assume the deploy identity."
  type        = string
  default     = "DataDepartment-Public/UofC-EMPI"
}

variable "github_oidc_subjects" {
  description = <<-EOT
    Federated credential subjects to trust, one per entry. Defaults to the `main`
    branch only, matching the single-environment/deploy-on-merge-to-main decision.
    Add more (e.g. "repo:OWNER/REPO:pull_request") only if CI needs to run
    `terraform plan` (not apply) from PRs against a different identity, or if a
    second environment is added later.
  EOT
  type        = list(string)
  default     = ["ref:refs/heads/main"]
}

variable "app_service_sku" {
  description = "App Service Plan SKU. B1 is the cheapest Linux tier that reliably supports custom containers; bump to P0v3/P1v3 for real traffic or if B1's 1.75GB RAM is tight for the pandas/splink pipeline."
  type        = string
  default     = "B1"
}

variable "postgres_version" {
  description = "PostgreSQL major version for the Flexible Server."
  type        = string
  default     = "16"
}

variable "postgres_sku" {
  description = "Postgres Flexible Server SKU. B_Standard_B1ms is the cheapest burstable tier — bump for real traffic."
  type        = string
  default     = "B_Standard_B1ms"
}

variable "postgres_storage_mb" {
  description = "Postgres Flexible Server storage size in MB."
  type        = number
  default     = 32768
}

variable "backend_extra_cors_origins" {
  description = <<-EOT
    Extra origins allowed to call the backend API directly, beyond the dashboard's
    own App Service origin (which is always added automatically). Normally empty:
    the dashboard is a BFF and proxies server-side, so browsers never call the
    backend directly today. Populate this only if that changes.
  EOT
  type        = list(string)
  default     = []
}

variable "tags" {
  description = "Common tags applied to all resources."
  type        = map(string)
  default = {
    project    = "empi"
    managed_by = "terraform"
  }
}
