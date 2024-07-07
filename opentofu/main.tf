terraform {
  required_providers {
    azurerm = {
      source  = "hashicorp/azurerm"
      version = "~> 3.0"
    }
  }
}

provider "azurerm" {
  features {}
}

resource "azurerm_resource_group" "westus2_testing" {
  name     = "westus2_testing"
  location = "westus2"
}

resource "azurerm_resource_group" "westus3_testing" {
  name     = "westus3_testing"
  location = "westus3"
}
