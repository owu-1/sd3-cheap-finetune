resource "azurerm_storage_account" "westus2_testing" {
  name                     = "11${azurerm_resource_group.westus2_testing.location}"
  resource_group_name      = azurerm_resource_group.westus2_testing.name
  location                 = azurerm_resource_group.westus2_testing.location
  account_tier             = "Standard"
  account_replication_type = "LRS"
}

resource "azurerm_storage_container" "westus2_testing_general" {
  name                  = "general"
  storage_account_name  = azurerm_storage_account.westus2_testing.name
  container_access_type = "private"
}

resource "azurerm_storage_account" "westus3_testing" {
  name                     = "11${azurerm_resource_group.westus3_testing.location}"
  resource_group_name      = azurerm_resource_group.westus3_testing.name
  location                 = azurerm_resource_group.westus3_testing.location
  account_tier             = "Standard"
  account_replication_type = "LRS"
}

resource "azurerm_storage_container" "westus3_testing_general" {
  name                  = "general"
  storage_account_name  = azurerm_storage_account.westus3_testing.name
  container_access_type = "private"
}
