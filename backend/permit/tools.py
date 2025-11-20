"""
LangChain tools for permit agent
Provides tools for document search, temporal queries, and permit metadata retrieval
"""
import os
import logging
from datetime import datetime
from typing import Literal, Optional, List
from langchain_core.tools import tool, Tool
from openai import OpenAI

# Import dari backend.client
from backend.client import AzureAISearch, get_permit_container

from .queries import (
    query_documents_by_issue_year,
    query_documents_by_expiration_year,
    query_expired_documents,
    query_documents_expiring_soon,
    query_permit_by_number,
    query_permits_by_installation
)

logger = logging.getLogger(__name__)

# Initialize clients using classes from backend.client
retrieval_client = AzureAISearch(
    base_url=os.getenv("AZURE_AI_SEARCH_ENDPOINT"),
    api_key=os.getenv("AZURE_AI_SEARCH_API_KEY"),
    index_name=os.getenv("AZURE_AI_SEARCH_INDEX_NAME")
)

cosmos_container = get_permit_container()

# Initialize OpenAI client for embeddings (if needed)
openai_client = OpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    base_url=f"{os.getenv('AZURE_OPENAI_ENDPOINT')}/openai/deployments/{os.getenv('AZURE_OPENAI_EMBEDDING_NAME')}",
    default_headers={"api-key": os.getenv("AZURE_OPENAI_API_KEY")}
)


async def get_permit_document_content(keyword: str) -> str:
    """
    Get relevant permit documents content from Azure AI Search based on keyword.
    Use this when questions require looking up document content.
    
    This tool performs semantic ranking search to find the most relevant documents
    and returns their content with titles for citation purposes.
    
    Args:
        keyword: Search query or filename from previous search
        
    Returns:
        Concatenated string of relevant documents with titles
        
    Example:
        >>> content = await get_permit_document_content("submarine pipeline IT Semarang")
        >>> print(content)
        # Returns: "Document1.pdf: Content about submarine pipeline..."
    """
    try:
        logger.info(f"Searching documents with keyword: {keyword}")
        
        search_results = await retrieval_client.semantic_ranking_search(
            keyword=keyword,
            k=10,
            select_fields=["title", "content"]
        )
        
        if not search_results.get('value'):
            logger.warning(f"No documents found for keyword: {keyword}")
            return "No relevant documents found for the query."
        
        docs = [doc.get('content', '') for doc in search_results['value']]
        titles = [doc.get('title', 'Untitled') for doc in search_results['value']]
        
        result = "\n\n".join([f"[{t}]:\n{d}" for t, d in zip(titles, docs)])
        
        logger.info(f"Found {len(docs)} documents for keyword: {keyword}")
        return result
    
    except Exception as e:
        logger.error(f"Error in get_permit_document_content: {e}")
        return f"Error retrieving documents: {str(e)}"


@tool
def get_current_date() -> str:
    """
    Get current date in YYYY-MM-DD format.
    Use this tool when you need to know today's date for temporal calculations.
    
    Returns:
        Current date as string in YYYY-MM-DD format
        
    Example:
        >>> date = get_current_date()
        >>> print(date)  # "2025-11-20"
    """
    current = datetime.now().strftime("%Y-%m-%d")
    logger.info(f"Current date requested: {current}")
    return current


@tool
def get_time_difference(now_datetime: str, target_datetime: str) -> int:
    """
    Calculate time difference in days between two dates.
    Use this to determine how many days until permit expiration or since issuance.
    
    Args:
        now_datetime: Current date in YYYY-MM-DD format
        target_datetime: Target date (expiration/issue date) in YYYY-MM-DD format
        
    Returns:
        Number of days difference (positive if target is in future, negative if in past)
        
    Example:
        >>> days = get_time_difference("2025-11-20", "2026-01-05")
        >>> print(days)  # 46
    """
    try:
        now = datetime.strptime(now_datetime, "%Y-%m-%d")
        target = datetime.strptime(target_datetime, "%Y-%m-%d")
        delta = target - now
        days = delta.days
        
        logger.info(f"Time difference calculated: {days} days between {now_datetime} and {target_datetime}")
        return days
    
    except ValueError as e:
        logger.error(f"Invalid date format: {e}")
        return 0


@tool
def get_list_documents_by_issue_year(
    permit_type: Optional[Literal['PLO', 'KKPR/KKPRL', 'Ijin Lingkungan']] = None,
    year: Optional[int] = None,
    organization: Optional[str] = None,
    operator: Optional[Literal['equal', 'greater', 'less']] = None,
    order_by: Optional[Literal['latest', 'earliest']] = 'latest'
) -> str:
    """
    Get list of documents issued in a specific year or year range.
    Use this when user asks about permits issued in certain years.
    
    Args:
        permit_type: Type of permit to filter ('PLO', 'KKPR/KKPRL', 'Ijin Lingkungan')
        year: Target year for filtering
        organization: Organization to filter by (PPN, PGN, KPI, SHU)
        operator: Comparison operator:
            - 'equal': Permits issued exactly in the year
            - 'greater': Permits issued in or after the year
            - 'less': Permits issued in or before the year
        order_by: Sort order ('latest' for newest first, 'earliest' for oldest first)
        
    Returns:
        Formatted string of matching documents with details
        
    Example:
        >>> result = get_list_documents_by_issue_year(
        ...     permit_type='PLO',
        ...     year=2024,
        ...     operator='equal',
        ...     organization='PPN'
        ... )
    """
    try:
        logger.info(
            f"Querying documents by issue year: type={permit_type}, year={year}, "
            f"org={organization}, operator={operator}, order={order_by}"
        )
        
        items = query_documents_by_issue_year(
            container=cosmos_container,
            permit_type=permit_type,
            year=year,
            organization=organization,
            operator=operator,
            order_by=order_by
        )
        
        if not items:
            return f"No documents found issued in year {year} with the specified filters."
        
        result_list = [f"List of documents issued: {len(items)} items found\n"]
        
        for idx, item in enumerate(items, 1):
            result_list.append(
                f"{idx}. {item.get('documentTitle', 'N/A')} - {item.get('permitNumber', 'N/A')}\n"
                f"   Organization: {item.get('organization', 'N/A')}\n"
                f"   Issue Date: {item.get('issueDate', 'N/A')}\n"
                f"   Summary: {item.get('permitSummary', 'No summary available')}\n"
            )
        
        result = "\n".join(result_list)
        logger.info(f"Found {len(items)} documents by issue year")
        return result
    
    except Exception as e:
        logger.error(f"Error in get_list_documents_by_issue_year: {e}")
        return f"Error retrieving documents: {str(e)}"


@tool
def get_list_documents_by_expiration_year(
    permit_type: Optional[Literal['PLO']] = None,
    year: Optional[int] = None,
    organization: Optional[str] = None,
    operator: Optional[Literal['equal', 'greater', 'less']] = None,
    order_by: Optional[Literal['latest', 'earliest']] = 'latest'
) -> str:
    """
    Get list of documents expiring in a specific year or year range.
    Use this when user asks about permit expiration dates.
    Note: Only PLO permits have expiration dates.
    
    Args:
        permit_type: Type of permit (only 'PLO' has expiration dates)
        year: Target year for expiration
        organization: Organization to filter by (PPN, PGN, KPI, SHU)
        operator: Comparison operator ('equal', 'greater', 'less')
        order_by: Sort order ('latest' or 'earliest')
        
    Returns:
        Formatted string of matching documents with expiration details
        
    Example:
        >>> result = get_list_documents_by_expiration_year(
        ...     permit_type='PLO',
        ...     year=2025,
        ...     operator='less',
        ...     organization='PGN'
        ... )
    """
    try:
        logger.info(
            f"Querying documents by expiration year: type={permit_type}, year={year}, "
            f"org={organization}, operator={operator}, order={order_by}"
        )
        
        items = query_documents_by_expiration_year(
            container=cosmos_container,
            permit_type=permit_type,
            year=year,
            organization=organization,
            operator=operator,
            order_by=order_by
        )
        
        if not items:
            return f"No documents found expiring in year {year} with the specified filters."
        
        result_list = [f"List of documents expiring: {len(items)} items found\n"]
        
        for idx, item in enumerate(items, 1):
            result_list.append(
                f"{idx}. {item.get('documentTitle', 'N/A')} - {item.get('permitNumber', 'N/A')}\n"
                f"   Organization: {item.get('organization', 'N/A')}\n"
                f"   Expiration Date: {item.get('expirationDate', 'N/A')}\n"
                f"   Summary: {item.get('permitSummary', 'No summary available')}\n"
            )
        
        result = "\n".join(result_list)
        logger.info(f"Found {len(items)} documents by expiration year")
        return result
    
    except Exception as e:
        logger.error(f"Error in get_list_documents_by_expiration_year: {e}")
        return f"Error retrieving documents: {str(e)}"


@tool
def get_list_documents_already_expired(
    organization: Optional[str] = None,
    order_by: Optional[Literal['latest', 'earliest']] = 'latest'
) -> str:
    """
    Get list of documents that have already expired (expiration date < today).
    Use this when user asks about expired permits or permits that need renewal.
    
    Args:
        organization: Organization to filter by (optional)
        order_by: Sort order ('latest' or 'earliest')
        
    Returns:
        Formatted string of expired documents with details
        
    Example:
        >>> result = get_list_documents_already_expired(organization='KPI')
    """
    try:
        current_date = datetime.now().strftime("%Y-%m-%d")
        
        logger.info(
            f"Querying expired documents as of {current_date}: org={organization}, order={order_by}"
        )
        
        items = query_expired_documents(
            container=cosmos_container,
            organization=organization,
            order_by=order_by
        )
        
        if not items:
            return f"No expired documents found as of {current_date}."
        
        result_list = [
            f"Documents that have already expired as of {current_date}:\n"
            f"Total: {len(items)} expired permits\n"
        ]
        
        for idx, item in enumerate(items, 1):
            result_list.append(
                f"{idx}. {item.get('documentTitle', 'N/A')} - {item.get('permitNumber', 'N/A')}\n"
                f"   Organization: {item.get('organization', 'N/A')}\n"
                f"   Installation: {item.get('installation', 'N/A')}\n"
                f"   Expired On: {item.get('expirationDate', 'N/A')}\n"
                f"   Summary: {item.get('permitSummary', 'No summary available')}\n"
            )
        
        result = "\n".join(result_list)
        logger.info(f"Found {len(items)} expired documents")
        return result
    
    except Exception as e:
        logger.error(f"Error in get_list_documents_already_expired: {e}")
        return f"Error retrieving expired documents: {str(e)}"


@tool
def get_list_documents_expiring_soon(
    days: int = 30,
    organization: Optional[str] = None,
    order_by: Optional[Literal['latest', 'earliest']] = 'earliest'
) -> str:
    """
    Get list of documents that will expire within specified number of days.
    Use this when user asks about permits expiring soon or needing renewal.
    
    Args:
        days: Number of days to look ahead (default 30)
        organization: Organization to filter by (optional)
        order_by: Sort order ('latest' or 'earliest')
        
    Returns:
        Formatted string of documents expiring soon
        
    Example:
        >>> result = get_list_documents_expiring_soon(days=60, organization='SHU')
    """
    try:
        current_date = datetime.now().strftime("%Y-%m-%d")
        
        logger.info(
            f"Querying documents expiring within {days} days: org={organization}, order={order_by}"
        )
        
        items = query_documents_expiring_soon(
            container=cosmos_container,
            days=days,
            organization=organization,
            order_by=order_by
        )
        
        if not items:
            return f"No documents expiring within the next {days} days."
        
        result_list = [
            f"Documents expiring within the next {days} days (as of {current_date}):\n"
            f"Total: {len(items)} permits\n"
        ]
        
        for idx, item in enumerate(items, 1):
            expiry = item.get('expirationDate', 'N/A')
            if expiry != 'N/A':
                days_remaining = (datetime.strptime(expiry, "%Y-%m-%d") - datetime.now()).days
                days_text = f"({days_remaining} days remaining)"
            else:
                days_text = ""
            
            result_list.append(
                f"{idx}. {item.get('documentTitle', 'N/A')} - {item.get('permitNumber', 'N/A')}\n"
                f"   Organization: {item.get('organization', 'N/A')}\n"
                f"   Installation: {item.get('installation', 'N/A')}\n"
                f"   Expires On: {expiry} {days_text}\n"
                f"   Summary: {item.get('permitSummary', 'No summary available')}\n"
            )
        
        result = "\n".join(result_list)
        logger.info(f"Found {len(items)} documents expiring within {days} days")
        return result
    
    except Exception as e:
        logger.error(f"Error in get_list_documents_expiring_soon: {e}")
        return f"Error retrieving documents expiring soon: {str(e)}"


@tool
def get_permit_details(permit_number: str) -> str:
    """
    Get detailed information about a specific permit by permit number.
    Use this when user asks about a specific permit number.
    
    Args:
        permit_number: The permit number to search for (e.g., 'PLO-2024-001')
        
    Returns:
        Detailed information about the permit
        
    Example:
        >>> details = get_permit_details('PLO-2024-001')
    """
    try:
        logger.info(f"Querying permit details for: {permit_number}")
        
        permit = query_permit_by_number(
            container=cosmos_container,
            permit_number=permit_number
        )
        
        if not permit:
            return f"Permit '{permit_number}' not found in the database."
        
        current_date = datetime.now()
        expiry = permit.get('expirationDate')
        
        status = "Active"
        days_info = ""
        
        if expiry:
            expiry_date = datetime.strptime(expiry, "%Y-%m-%d")
            days_diff = (expiry_date - current_date).days
            
            if days_diff < 0:
                status = "Expired"
                days_info = f"Expired {abs(days_diff)} days ago"
            elif days_diff <= 30:
                status = "Expiring Soon"
                days_info = f"Expires in {days_diff} days"
            else:
                days_info = f"Expires in {days_diff} days"
        
        result = f"""
Permit Details for {permit_number}:

Document Title: {permit.get('documentTitle', 'N/A')}
Permit Type: {permit.get('permitType', 'N/A')}
Organization: {permit.get('organization', 'N/A')}
Installation: {permit.get('installation', 'N/A')}

Issue Date: {permit.get('issueDate', 'N/A')}
Expiration Date: {expiry or 'N/A'}
Status: {status} {days_info}

Summary:
{permit.get('permitSummary', 'No summary available')}

Document Path: {permit.get('filepath', 'N/A')}
"""
        
        logger.info(f"Retrieved details for permit: {permit_number}")
        return result.strip()
    
    except Exception as e:
        logger.error(f"Error in get_permit_details: {e}")
        return f"Error retrieving permit details: {str(e)}"


@tool
def get_permits_by_installation(
    installation: str,
    permit_type: Optional[Literal['PLO', 'KKPR/KKPRL', 'Ijin Lingkungan']] = None
) -> str:
    """
    Get list of permits for a specific installation location.
    Use this when user asks about permits for specific installations.
    
    Args:
        installation: Installation location name (e.g., 'IT Semarang', 'IT Jakarta')
        permit_type: Type of permit to filter (optional)
        
    Returns:
        Formatted string of permits for the installation
        
    Example:
        >>> result = get_permits_by_installation('IT Semarang', permit_type='PLO')
    """
    try:
        logger.info(f"Querying permits for installation: {installation}, type={permit_type}")
        
        items = query_permits_by_installation(
            container=cosmos_container,
            installation=installation,
            permit_type=permit_type
        )
        
        if not items:
            return f"No permits found for installation '{installation}'."
        
        result_list = [
            f"Permits for installation '{installation}':\n"
            f"Total: {len(items)} permits found\n"
        ]
        
        for idx, item in enumerate(items, 1):
            result_list.append(
                f"{idx}. {item.get('permitNumber', 'N/A')} - {item.get('permitType', 'N/A')}\n"
                f"   Document: {item.get('documentTitle', 'N/A')}\n"
                f"   Organization: {item.get('organization', 'N/A')}\n"
                f"   Issue Date: {item.get('issueDate', 'N/A')}\n"
                f"   Expiration: {item.get('expirationDate', 'N/A')}\n"
                f"   Summary: {item.get('permitSummary', 'No summary available')}\n"
            )
        
        result = "\n".join(result_list)
        logger.info(f"Found {len(items)} permits for installation: {installation}")
        return result
    
    except Exception as e:
        logger.error(f"Error in get_permits_by_installation: {e}")
        return f"Error retrieving permits for installation: {str(e)}"


def get_permit_tools() -> List[Tool]:
    """
    Return all permit agent tools for LangChain agent.
    
    Returns:
        List of LangChain Tool objects
        
    Tools included:
        1. get_permit_document_content - Search documents by keyword
        2. get_current_date - Get today's date
        3. get_time_difference - Calculate days between dates
        4. get_list_documents_by_issue_year - Query by issue year
        5. get_list_documents_by_expiration_year - Query by expiration year
        6. get_list_documents_already_expired - Get expired permits
        7. get_list_documents_expiring_soon - Get permits expiring soon
        8. get_permit_details - Get specific permit details
        9. get_permits_by_installation - Get permits by installation
    """
    logger.info("Initializing permit agent tools")
    
    return [
        Tool(
            name="get_permit_document_content",
            description="""CRITICAL: Use this tool FIRST when the user asks ANY question about permit content, 
            specific information in permits, or details that require looking into documents. 
            This performs semantic search to find the most relevant permit documents.
            
            Input: Search query or keywords from the user's question
            Returns: Top 10 relevant permit documents with titles and content
            
            Example use cases:
            - "What is the pipeline length in IT Semarang permit?"
            - "What are the requirements in KKPRL document?"
            - "What depth is mentioned in the PLO permit?"
            
            DO NOT try to answer content questions without calling this tool first.""",
            func=get_permit_document_content,
            coroutine=get_permit_document_content
        ),
        get_current_date,
        get_time_difference,
        get_list_documents_by_issue_year,
        get_list_documents_by_expiration_year,
        get_list_documents_already_expired,
        get_list_documents_expiring_soon,
        get_permit_details,
        get_permits_by_installation
    ]