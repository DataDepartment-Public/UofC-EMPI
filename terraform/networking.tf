# VNet integration + private endpoints — the PHI-readiness network pass (see
# to-do.md A4). Before this file, both App Services and Postgres were public
# HTTPS/public-network-access-enabled with no VNet at all; the "standard
# baseline, not PHI-ready" tradeoff called out in the original review.
#
# Two distinct mechanisms are used together, matching each Azure service's
# supported private-connectivity model:
#   * Postgres Flexible Server uses VNet-injected (delegated subnet) private
#     access, not Private Endpoint (see postgres.tf) — its own dedicated
#     subnet below.
#   * Storage Account and the backend App Service (inbound) use standard
#     Private Endpoints into a shared subnet.
#   * Both App Services get *outbound* regional VNet Integration
#     (`virtual_network_subnet_id`, set directly on each azurerm_linux_web_app
#     in app_service.tf) into the same delegated subnet, so they can reach
#     the now-private Postgres/Storage/backend over the VNet instead of the
#     public internet.
#
# The dashboard's EMPI_API_URL keeps pointing at the backend's normal
# *.azurewebsites.net hostname (app_service.tf) — no code change needed.
# Once the dashboard has VNet integration and the private DNS zone below is
# linked to this VNet, that same hostname resolves to the backend's private
# IP from inside the VNet and stays public-IP/blocked from anywhere else
# (split-horizon DNS is what Private Endpoint + Private DNS Zone gives you).

resource "azurerm_virtual_network" "main" {
  name                = "vnet-${local.name_prefix}"
  resource_group_name = azurerm_resource_group.main.name
  location            = azurerm_resource_group.main.location
  address_space       = ["10.20.0.0/16"]

  tags = local.tags
}

# ---- Subnets ----------------------------------------------------------------

# Outbound regional VNet Integration for both App Services (backend +
# dashboard both use this same subnet — that's supported and normal; it's
# what lets either app reach anything privately, including each other).
resource "azurerm_subnet" "app" {
  name                 = "snet-app"
  resource_group_name  = azurerm_resource_group.main.name
  virtual_network_name = azurerm_virtual_network.main.name
  address_prefixes     = ["10.20.1.0/24"]

  delegation {
    name = "app-service-delegation"
    service_delegation {
      name    = "Microsoft.Web/serverFarms"
      actions = ["Microsoft.Network/virtualNetworks/subnets/action"]
    }
  }
}

# Postgres Flexible Server's VNet-injected private access (postgres.tf) --
# this subnet is *dedicated* to Postgres; Azure requires the delegated
# subnet not be shared with other resource types.
resource "azurerm_subnet" "postgres" {
  name                 = "snet-postgres"
  resource_group_name  = azurerm_resource_group.main.name
  virtual_network_name = azurerm_virtual_network.main.name
  address_prefixes     = ["10.20.2.0/24"]

  delegation {
    name = "postgres-delegation"
    service_delegation {
      name    = "Microsoft.DBforPostgreSQL/flexibleServers"
      actions = ["Microsoft.Network/virtualNetworks/subnets/join/action"]
    }
  }
}

# Private Endpoints land here: Storage (file share), the backend App Service
# (inbound), and Key Vault (see keyvault.tf).
resource "azurerm_subnet" "private_endpoints" {
  name                 = "snet-private-endpoints"
  resource_group_name  = azurerm_resource_group.main.name
  virtual_network_name = azurerm_virtual_network.main.name
  address_prefixes     = ["10.20.3.0/24"]

  private_endpoint_network_policies = "Disabled"
}

# ---- Private DNS zones + VNet links -----------------------------------------
#
# Each zone's records are populated automatically by its resource's
# `private_dns_zone_id` (Postgres) or `private_dns_zone_group` (Private
# Endpoints) below -- these resources just create the zone and link it to
# the VNet so records in it actually resolve for anything in that VNet.

resource "azurerm_private_dns_zone" "postgres" {
  name                = "privatelink.postgres.database.azure.com"
  resource_group_name = azurerm_resource_group.main.name

  tags = local.tags
}

resource "azurerm_private_dns_zone_virtual_network_link" "postgres" {
  name                  = "link-postgres"
  resource_group_name   = azurerm_resource_group.main.name
  private_dns_zone_name = azurerm_private_dns_zone.postgres.name
  virtual_network_id    = azurerm_virtual_network.main.id
  registration_enabled  = false
}

resource "azurerm_private_dns_zone" "storage_file" {
  name                = "privatelink.file.core.windows.net"
  resource_group_name = azurerm_resource_group.main.name

  tags = local.tags
}

resource "azurerm_private_dns_zone_virtual_network_link" "storage_file" {
  name                  = "link-storage-file"
  resource_group_name   = azurerm_resource_group.main.name
  private_dns_zone_name = azurerm_private_dns_zone.storage_file.name
  virtual_network_id    = azurerm_virtual_network.main.id
  registration_enabled  = false
}

resource "azurerm_private_dns_zone" "sites" {
  name                = "privatelink.azurewebsites.net"
  resource_group_name = azurerm_resource_group.main.name

  tags = local.tags
}

resource "azurerm_private_dns_zone_virtual_network_link" "sites" {
  name                  = "link-sites"
  resource_group_name   = azurerm_resource_group.main.name
  private_dns_zone_name = azurerm_private_dns_zone.sites.name
  virtual_network_id    = azurerm_virtual_network.main.id
  registration_enabled  = false
}

resource "azurerm_private_dns_zone" "vault" {
  name                = "privatelink.vaultcore.azure.net"
  resource_group_name = azurerm_resource_group.main.name

  tags = local.tags
}

resource "azurerm_private_dns_zone_virtual_network_link" "vault" {
  name                  = "link-vault"
  resource_group_name   = azurerm_resource_group.main.name
  private_dns_zone_name = azurerm_private_dns_zone.vault.name
  virtual_network_id    = azurerm_virtual_network.main.id
  registration_enabled  = false
}

# ---- Private Endpoints -------------------------------------------------------

resource "azurerm_private_endpoint" "storage_file" {
  name                = "pe-${local.name_prefix}-storage-file"
  resource_group_name = azurerm_resource_group.main.name
  location            = azurerm_resource_group.main.location
  subnet_id           = azurerm_subnet.private_endpoints.id

  private_service_connection {
    name                           = "psc-storage-file"
    private_connection_resource_id = azurerm_storage_account.main.id
    subresource_names              = ["file"]
    is_manual_connection           = false
  }

  private_dns_zone_group {
    name                 = "dns-storage-file"
    private_dns_zone_ids = [azurerm_private_dns_zone.storage_file.id]
  }

  tags = local.tags
}

# Inbound-only Private Endpoint for the backend -- this plus
# public_network_access_enabled = false on azurerm_linux_web_app.backend
# (app_service.tf) is what makes "no public ingress on the backend" real.
# The dashboard reaches it privately via its own outbound VNet Integration
# (also app_service.tf) plus this zone being linked to the same VNet.
resource "azurerm_private_endpoint" "backend" {
  name                = "pe-${local.name_prefix}-backend"
  resource_group_name = azurerm_resource_group.main.name
  location            = azurerm_resource_group.main.location
  subnet_id           = azurerm_subnet.private_endpoints.id

  private_service_connection {
    name                           = "psc-backend"
    private_connection_resource_id = azurerm_linux_web_app.backend.id
    subresource_names              = ["sites"]
    is_manual_connection           = false
  }

  private_dns_zone_group {
    name                 = "dns-backend"
    private_dns_zone_ids = [azurerm_private_dns_zone.sites.id]
  }

  tags = local.tags
}
