"""
CosmosDB client for permit metadata storage and queries
"""
import os
from typing import Optional, Dict, Any, List
from azure.cosmos import CosmosClient as AzureCosmosClient, PartitionKey
from azure.cosmos.container import ContainerProxy
from azure.cosmos.database import DatabaseProxy
import logging

logger = logging.getLogger(__name__)

class CosmosDBClient:
    """
    Client for CosmosDB operations
    Manages connections and provides database/container access
    """
    
    def __init__(
        self,
        uri: Optional[str] = None,
        key: Optional[str] = None,
        database_id: Optional[str] = None,
        container_id: Optional[str] = None
    ):
        """
        Initialize CosmosDB client
        
        Args:
            uri: CosmosDB endpoint URI
            key: CosmosDB key
            database_id: Database name
            container_id: Container name
        """
        self.uri = uri
        self.key = key
        self.database_id = database_id
        self.container_id = container_id
        
        if not all([self.uri, self.key]):
            raise ValueError(
                "CosmosDB credentials not configured. "
                "Provide uri and key parameters"
            )
        
        self.client = AzureCosmosClient(url=self.uri, credential=self.key)
        self._database = None
        self._container = None
        
        logger.info(
            f"Initialized CosmosDBClient for database: {self.database_id}, "
            f"container: {self.container_id}"
        )
    
    @property
    def database(self) -> DatabaseProxy:
        """Get or create database proxy"""
        if self._database is None:
            self._database = self.client.get_database_client(self.database_id)
        return self._database
    
    @property
    def container(self) -> ContainerProxy:
        """Get or create container proxy"""
        if self._container is None:
            self._container = self.database.get_container_client(self.container_id)
        return self._container
    
    def query_items(
        self,
        query: str,
        parameters: Optional[List[Dict[str, Any]]] = None,
        enable_cross_partition_query: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Execute query on container
        
        Args:
            query: SQL query string
            parameters: Query parameters
            enable_cross_partition_query: Enable cross-partition queries
            
        Returns:
            List of query results
        """
        try:
            items = list(self.container.query_items(
                query=query,
                parameters=parameters,
                enable_cross_partition_query=enable_cross_partition_query
            ))
            logger.debug(f"Query returned {len(items)} items")
            return items
        except Exception as e:
            logger.error(f"Error querying items: {str(e)}")
            raise


def get_permit_container() -> ContainerProxy:
    """
    Get permit metadata container from CosmosDB
    Uses environment variables from .env:
    - COSMOS_DB_URI
    - COSMOS_DB_KEY  
    - COSMOS_DB_DATABASE
    - COSMOS_DB_CONTAINER
    
    Returns:
        Container proxy for permit metadata
        
    Example:
        >>> container = get_permit_container()
        >>> results = container.query_items(
        ...     query="SELECT * FROM c WHERE c.permitType = 'PLO'",
        ...     enable_cross_partition_query=True
        ... )
    """
    uri = os.getenv("COSMOS_DB_URI")
    key = os.getenv("COSMOS_DB_KEY")
    database = os.getenv("COSMOS_DB_DATABASE")
    container = os.getenv("COSMOS_DB_CONTAINER")
    
    if not all([uri, key, database, container]):
        raise ValueError(
            "Permit Metadata CosmosDB not configured. "
            "Set COSMOS_DB_URI, COSMOS_DB_KEY, COSMOS_DB_DATABASE, "
            "and COSMOS_DB_CONTAINER in .env"
        )
    
    client = CosmosDBClient(
        uri=uri,
        key=key,
        database_id=database,
        container_id=container
    )
    return client.container


def get_conversation_container() -> ContainerProxy:
    """
    Get conversation history container from CosmosDB
    Uses existing environment variables from .env:
    - AZURE_COSMOSDB_ACCOUNT
    - AZURE_COSMOSDB_ACCOUNT_KEY
    - AZURE_COSMOSDB_DATABASE
    - AZURE_COSMOSDB_CONVERSATIONS_CONTAINER
    
    Returns:
        Container proxy for conversation history
        
    Example:
        >>> container = get_conversation_container()
        >>> # Query conversations
    """
    account = os.getenv("AZURE_COSMOSDB_ACCOUNT")
    key = os.getenv("AZURE_COSMOSDB_ACCOUNT_KEY")
    database = os.getenv("AZURE_COSMOSDB_DATABASE")
    container = os.getenv("AZURE_COSMOSDB_CONVERSATIONS_CONTAINER")
    
    if not all([account, key, database, container]):
        raise ValueError(
            "Conversation History CosmosDB not configured. "
            "Set AZURE_COSMOSDB_ACCOUNT, AZURE_COSMOSDB_ACCOUNT_KEY, "
            "AZURE_COSMOSDB_DATABASE, and AZURE_COSMOSDB_CONVERSATIONS_CONTAINER in .env"
        )
    
    uri = f"https://{account}.documents.azure.com:443/"
    
    client = CosmosDBClient(
        uri=uri,
        key=key,
        database_id=database,
        container_id=container
    )
    return client.container