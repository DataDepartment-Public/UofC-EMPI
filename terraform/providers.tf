terraform {
  required_version = ">= 1.9.0"

  required_providers {
    azurerm = {
      source  = "hashicorp/azurerm"
      version = "~> 4.15"
    }
    azuread = {
      source  = "hashicorp/azuread"
      version = "~> 3.0"
    }
    random = {
      source  = "hashicorp/random"
      version = "~> 3.6"
    }
    http = {
      source  = "hashicorp/http"
      version = "~> 3.4"
    }
  }

  # resource_group_name / storage_account_name / container_name / key are
  # deliberately omitted here and supplied at `terraform init` time via
  # `-backend-config=backend.hcl` (gitignored) or equivalent -backend-config
  # flags in CI, because the storage account that holds this state is itself
  # infrastructure someone with subscription access has to create once before
  # this config's own resource group exists. See terraform/README.md
  # "Bootstrap". use_azuread_auth authenticates to the state storage account
  # with the caller's Azure AD identity (local `az login` or GitHub OIDC)
  # instead of a storage account key, so no extra secret is needed for state
  # access beyond what's already required to manage the rest of the infra.
  backend "azurerm" {
    use_azuread_auth = true
  }
}

provider "azurerm" {
  features {
    resource_group {
      prevent_deletion_if_contains_resources = false
    }
  }
}

provider "azuread" {}
