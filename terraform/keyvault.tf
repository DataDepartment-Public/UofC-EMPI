# Customer-managed encryption keys for Storage and Postgres (to-do.md A4).
# Both already get Microsoft-managed encryption at rest by default; CMK adds
# key-rotation control and the ability to revoke data access by disabling
# the key -- the piece a client security questionnaire usually asks for by
# name once PHI is involved.
#
# One shared user-assigned identity is used for both, rather than a separate
# UAMI per resource -- a deliberate scope simplification (both resources
# already trust the same environment; splitting them buys little here) and
# specifically NOT each resource's own system-assigned identity, to avoid
# the chicken-and-egg problem that would otherwise create: granting the key
# vault role needs the identity to exist first, but a system-assigned
# identity is only created *with* its resource. The UAMI exists
# independently, so it can be granted access before Storage/Postgres are
# even created.

resource "azurerm_user_assigned_identity" "cmk" {
  name                = "id-${local.name_prefix}-cmk"
  resource_group_name = azurerm_resource_group.main.name
  location            = azurerm_resource_group.main.location

  tags = local.tags
}

# The current `terraform apply` caller's own public IP -- see the
# network_acls comment below for why this is needed at all. Any plain-text
# "what's my IP" service works; ipify is a long-standing, widely-used one
# for exactly this Terraform pattern. `chomp` strips the trailing newline
# the endpoint returns.
data "http" "apply_caller_ip" {
  url = "https://api.ipify.org?format=text"
}

# purge_protection_enabled is required by Azure for a vault used as a CMK
# source for Storage/Postgres, and is permanent once set -- tearing down
# this environment leaves the vault soft-deleted for its retention window
# rather than immediately gone.
#
# public_network_access_enabled = true + a Deny-by-default network ACL that
# allows only the current apply caller's IP -- NOT the `false` a first
# instinct reaches for. Reason: azurerm_key_vault_key.storage/.postgres
# below are managed through the Key Vault DATA plane (creating/reading
# actual key material), which network ACLs govern regardless of the
# caller's RBAC role -- "AzureServices" bypass only covers first-party
# Azure services (Azure Backup, Azure ML, etc.), never GitHub Actions or a
# human's laptop, even authenticated via OIDC/az login. With
# public_network_access_enabled = false outright, ip_rules are ignored
# entirely and there is no way for Terraform itself (CI or a human running
# the first bootstrap apply) to ever create these two key resources -- every
# `terraform apply` that touches them would fail. Everything that reaches
# this vault over the VNet (the app's own managed identities, Azure ML) goes
# through the Private Endpoint below, which bypasses network_acls
# entirely and is unaffected either way; this only changes how EXTERNAL
# callers (whoever is running Terraform) are gated -- from "always blocked,
# no matter what" to "blocked unless it's the IP running this exact apply."
resource "azurerm_key_vault" "main" {
  name                = "kv-${local.name_prefix}-${local.suffix}"
  resource_group_name = azurerm_resource_group.main.name
  location            = azurerm_resource_group.main.location
  tenant_id           = data.azurerm_client_config.current.tenant_id
  sku_name            = "standard"

  rbac_authorization_enabled = true
  purge_protection_enabled   = true

  public_network_access_enabled = true

  network_acls {
    default_action = "Deny"
    bypass         = "AzureServices"
    ip_rules       = ["${chomp(data.http.apply_caller_ip.response_body)}/32"]
  }

  tags = local.tags
}

resource "azurerm_private_endpoint" "vault" {
  name                = "pe-${local.name_prefix}-vault"
  resource_group_name = azurerm_resource_group.main.name
  location            = azurerm_resource_group.main.location
  subnet_id           = azurerm_subnet.private_endpoints.id

  private_service_connection {
    name                           = "psc-vault"
    private_connection_resource_id = azurerm_key_vault.main.id
    subresource_names              = ["vault"]
    is_manual_connection           = false
  }

  private_dns_zone_group {
    name                 = "dns-vault"
    private_dns_zone_ids = [azurerm_private_dns_zone.vault.id]
  }

  tags = local.tags
}

# Built-in role scoped specifically for CMK usage (wrap/unwrap only) --
# deliberately narrower than a general Key Vault data-access role.
resource "azurerm_role_assignment" "cmk_key_access" {
  scope                = azurerm_key_vault.main.id
  role_definition_name = "Key Vault Crypto Service Encryption User"
  principal_id         = azurerm_user_assigned_identity.cmk.principal_id
}

resource "azurerm_key_vault_key" "storage" {
  name         = "cmk-storage"
  key_vault_id = azurerm_key_vault.main.id
  key_type     = "RSA"
  key_size     = 2048
  key_opts     = ["wrapKey", "unwrapKey"]

  depends_on = [azurerm_role_assignment.cmk_key_access]
}

resource "azurerm_key_vault_key" "postgres" {
  name         = "cmk-postgres"
  key_vault_id = azurerm_key_vault.main.id
  key_type     = "RSA"
  key_size     = 2048
  key_opts     = ["wrapKey", "unwrapKey"]

  depends_on = [azurerm_role_assignment.cmk_key_access]
}
