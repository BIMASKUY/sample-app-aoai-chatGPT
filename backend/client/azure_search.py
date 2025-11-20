"""
Azure AI Search client for document retrieval
"""
import os
import httpx
from typing import List, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

class AzureAISearch:
    """
    Client for Azure AI Search operations
    Provides semantic search and vector search capabilities
    """
    
    def __init__(
        self, 
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        index_name: Optional[str] = None
    ):
        """
        Initialize Azure AI Search client
        
        Args:
            base_url: Azure AI Search endpoint (defaults to env var)
            api_key: API key (defaults to env var)
            index_name: Index name (defaults to env var)
        """
        self.base_url = base_url or os.getenv("AZURE_AI_SEARCH_ENDPOINT")
        self.api_key = api_key or os.getenv("AZURE_AI_SEARCH_API_KEY")
        self.index_name = index_name or os.getenv("AZURE_AI_SEARCH_INDEX_NAME")
        
        if not all([self.base_url, self.api_key, self.index_name]):
            raise ValueError(
                "Azure AI Search credentials not configured. "
                "Set AZURE_AI_SEARCH_ENDPOINT, AZURE_AI_SEARCH_API_KEY, and AZURE_AI_SEARCH_INDEX_NAME"
            )
        
        self.headers = {
            "Content-Type": "application/json",
            "api-key": self.api_key
        }
        
        logger.info(f"Initialized AzureAISearch client for index: {self.index_name}")
    
    async def semantic_ranking_search(
        self, 
        keyword: str, 
        k: int = 10,
        select_fields: Optional[List[str]] = None,
        vector_fields: Optional[List[str]] = None,
        filter_query: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Perform semantic ranking search on Azure AI Search
        
        Args:
            keyword: Search query
            k: Number of results to return (top-k)
            select_fields: Fields to include in response
            vector_fields: Vector fields for hybrid search
            filter_query: OData filter expression
            
        Returns:
            Search results with value list containing matching documents
            
        Example:
            >>> client = AzureAISearch()
            >>> results = await client.semantic_ranking_search(
            ...     keyword="submarine pipeline",
            ...     k=5,
            ...     select_fields=["title", "content"],
            ...     filter_query="permitType eq 'PLO'"
            ... )
        """
        url = f"{self.base_url}/indexes/{self.index_name}/docs/search?api-version=2023-11-01"
        
        payload = {
            "search": keyword,
            "queryType": "semantic",
            "semanticConfiguration": "default",
            "top": k,
        }
        
        if select_fields:
            payload["select"] = ",".join(select_fields)
        
        if vector_fields:
            payload["vectorQueries"] = [
                {
                    "kind": "text",
                    "text": keyword,
                    "fields": ",".join(vector_fields)
                }
            ]
        
        if filter_query:
            payload["filter"] = filter_query
        
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(url, json=payload, headers=self.headers)
                response.raise_for_status()
                return response.json()
        except httpx.HTTPError as e:
            logger.error(f"Azure AI Search error: {e}")
            raise
    
    async def vector_search(
        self,
        vector: List[float],
        k: int = 10,
        vector_fields: List[str] = None,
        select_fields: List[str] = None,
        filter_query: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Perform pure vector search
        
        Args:
            vector: Embedding vector
            k: Number of results
            vector_fields: Fields containing vectors
            select_fields: Fields to return
            filter_query: OData filter
            
        Returns:
            Search results
        """
        url = f"{self.base_url}/indexes/{self.index_name}/docs/search?api-version=2023-11-01"
        
        payload = {
            "vectorQueries": [
                {
                    "kind": "vector",
                    "vector": vector,
                    "fields": ",".join(vector_fields or ["contentVector"]),
                    "k": k
                }
            ]
        }
        
        if select_fields:
            payload["select"] = ",".join(select_fields)
        
        if filter_query:
            payload["filter"] = filter_query
        
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(url, json=payload, headers=self.headers)
                response.raise_for_status()
                return response.json()
        except httpx.HTTPError as e:
            logger.error(f"Vector search error: {e}")
            raise
    
    async def hybrid_search(
        self,
        keyword: str,
        vector: Optional[List[float]] = None,
        k: int = 10,
        select_fields: Optional[List[str]] = None,
        filter_query: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Perform hybrid search (semantic + vector)
        
        Args:
            keyword: Text query
            vector: Optional embedding vector
            k: Number of results
            select_fields: Fields to return
            filter_query: OData filter
            
        Returns:
            Search results combining semantic and vector search
        """
        url = f"{self.base_url}/indexes/{self.index_name}/docs/search?api-version=2023-11-01"
        
        payload = {
            "search": keyword,
            "queryType": "semantic",
            "semanticConfiguration": "default",
            "top": k,
        }
        
        if vector:
            payload["vectorQueries"] = [
                {
                    "kind": "vector",
                    "vector": vector,
                    "fields": "contentVector,titleVector",
                    "k": k
                }
            ]
        
        if select_fields:
            payload["select"] = ",".join(select_fields)
        
        if filter_query:
            payload["filter"] = filter_query
        
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(url, json=payload, headers=self.headers)
                response.raise_for_status()
                return response.json()
        except httpx.HTTPError as e:
            logger.error(f"Hybrid search error: {e}")
            raise