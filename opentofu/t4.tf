# resource "azurerm_virtual_network" "westus2_testing" {
#   name                = "westus2_testing"
#   address_space       = ["10.0.0.0/16"]
#   location            = azurerm_resource_group.westus2_testing.location
#   resource_group_name = azurerm_resource_group.westus2_testing.name
# }

# resource "azurerm_subnet" "westus2_testing" {
#   name                 = "westus2_testing"
#   resource_group_name  = azurerm_resource_group.westus2_testing.name
#   virtual_network_name = azurerm_virtual_network.westus2_testing.name
#   address_prefixes     = ["10.0.0.0/24"]
# }

# resource "azurerm_public_ip" "westus2_testing" {
#   name                = "westus2_testing"
#   resource_group_name = azurerm_resource_group.westus2_testing.name
#   location            = azurerm_resource_group.westus2_testing.location
#   ip_version          = "IPv4"  # IPv6 only is too hard. Cannot connect to most things
#   allocation_method   = "Static"
#   sku                 = "Standard"
#   sku_tier            = "Regional"
#   zones = toset(["3"])
# }

# resource "azurerm_network_interface" "westus2_testing" {
#   name                = "westus2_testing"
#   location            = azurerm_resource_group.westus2_testing.location
#   resource_group_name = azurerm_resource_group.westus2_testing.name
#   accelerated_networking_enabled = true

#   ip_configuration {
#     name                          = "v4"
#     primary                       = true
#     subnet_id                     = azurerm_subnet.westus2_testing.id
#     public_ip_address_id          = azurerm_public_ip.westus2_testing.id
#     private_ip_address_version    = "IPv4"
#     private_ip_address_allocation = "Dynamic"
#   }
# }

# resource "azurerm_network_security_group" "westus2_testing" {
#   name                = "westus2_testing"
#   location            = azurerm_resource_group.westus2_testing.location
#   resource_group_name = azurerm_resource_group.westus2_testing.name

#   security_rule {
#     name                       = "SSH"
#     priority                   = 100
#     direction                  = "Inbound"
#     access                     = "Allow"
#     protocol                   = "Tcp"
#     source_port_range          = "*"
#     destination_port_range     = "22"
#     source_address_prefix      = "*"
#     destination_address_prefix = "*"
#   }
# }

# resource "azurerm_network_interface_security_group_association" "westus2_testing" {
#   network_interface_id      = azurerm_network_interface.westus2_testing.id
#   network_security_group_id = azurerm_network_security_group.westus2_testing.id
# }

# resource "azurerm_linux_virtual_machine" "t4_instance" {
#   name                = "t4"
#   resource_group_name = azurerm_resource_group.westus2_testing.name
#   location            = azurerm_resource_group.westus2_testing.location
#   size                = "Standard_NC4as_T4_v3"
#   admin_username      = "ubuntu"
#   priority            = "Spot"
#   eviction_policy     = "Delete"
#   network_interface_ids = [
#     azurerm_network_interface.westus2_testing.id,
#   ]
#   zone = "3"

#   identity {
#     type = "SystemAssigned"
#   }

#   admin_ssh_key {
#     username   = "ubuntu"
#     public_key = file(var.rsa_publickey_location)  # ed25519 not supported
#   }

#   os_disk {
#     caching              = "ReadOnly"
#     storage_account_type = "Standard_LRS"  # Premium_LRS fails when using Ephemeral Disk
#     disk_size_gb         = 176

#     diff_disk_settings {
#       option = "Local"
#       placement = "ResourceDisk"
#     }
#   }

#   source_image_reference {
#     publisher = "canonical"
#     offer     = "ubuntu-24_04-lts"
#     sku       = "server"
#     version   = "latest"
#   }
# }

# resource "azurerm_role_assignment" "vm" {
#   scope                = azurerm_storage_account.westus2_testing.id
#   role_definition_name = "Storage Blob Data Contributor"
#   principal_id         = azurerm_linux_virtual_machine.t4_instance.identity[0].principal_id
# }
