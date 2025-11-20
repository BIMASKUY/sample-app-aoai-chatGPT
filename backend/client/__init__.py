"""
Client classes for external services
Provides reusable clients for Azure AI Search and CosmosDB
"""
from .azure_search import AzureAISearch
from .cosmos_db import (
    CosmosDBClient, 
    get_permit_container,
    get_conversation_container
)

__all__ = [
    'AzureAISearch',
    'CosmosDBClient',
    'get_permit_container',
    'get_conversation_container'
]