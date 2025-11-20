"""
CosmosDB queries for permit metadata
Handles temporal data queries (issue dates, expiration dates, etc.)
"""
import logging
from datetime import datetime
from typing import Optional, List, Dict, Any, Literal
from azure.cosmos.container import ContainerProxy

# Import dari backend.client
from backend.client import get_permit_container

logger = logging.getLogger(__name__)


def query_documents_by_issue_year(
    container: Optional[ContainerProxy] = None,
    permit_type: Optional[Literal['PLO', 'KKPR/KKPRL', 'Ijin Lingkungan']] = None,
    year: Optional[int] = None,
    organization: Optional[str] = None,
    operator: Optional[Literal['equal', 'greater', 'less']] = None,
    order_by: str = 'latest'
) -> List[Dict[str, Any]]:
    """
    Query documents by issue year from CosmosDB
    
    Args:
        container: CosmosDB container (auto-initialized if None)
        permit_type: Type of permit to filter
        year: Target year
        organization: Organization to filter
        operator: Comparison operator for year
        order_by: Sort order ('latest' or 'earliest')
        
    Returns:
        List of matching documents
        
    Example:
        >>> results = query_documents_by_issue_year(
        ...     permit_type='PLO',
        ...     year=2024,
        ...     operator='equal',
        ...     organization='PPN'
        ... )
    """
    # Auto-initialize container if not provided
    if container is None:
        container = get_permit_container()
    
    query = """
        SELECT c.documentTitle, c.permitType, c.organization, 
               p.issueDate, p.permitSummary, p.permitNumber
        FROM c
        JOIN p IN c.permits
    """
    
    conditions = []
    parameters = []

    if permit_type:
        conditions.append("c.permitType = @permitType")
        parameters.append(dict(name="@permitType", value=permit_type))

    if operator and year:
        parameters.append(dict(name="@year", value=year))
        if operator == 'greater':
            conditions.append("YEAR(p.issueDate) >= @year")
        elif operator == 'less':
            conditions.append("YEAR(p.issueDate) <= @year")
        else:
            conditions.append("YEAR(p.issueDate) = @year")
        
    if organization:
        conditions.append("c.organization = @organization")
        parameters.append(dict(name="@organization", value=organization))

    if conditions:
        query += " WHERE " + " AND ".join(conditions)

    try:
        results = container.query_items(
            query=query,
            parameters=parameters,
            enable_cross_partition_query=True
        )

        items = [item for item in results]
        
        if order_by == 'latest':
            items.sort(key=lambda x: x.get('issueDate', ''), reverse=True)
        else:
            items.sort(key=lambda x: x.get('issueDate', ''))
        
        logger.info(f"Found {len(items)} documents by issue year")
        return items
    
    except Exception as e:
        logger.error(f"Error querying documents by issue year: {e}")
        return []


def query_documents_by_expiration_year(
    container: Optional[ContainerProxy] = None,
    permit_type: Optional[Literal['PLO']] = None,
    year: Optional[int] = None,
    organization: Optional[str] = None,
    operator: Optional[Literal['equal', 'greater', 'less']] = None,
    order_by: str = 'latest'
) -> List[Dict[str, Any]]:
    """
    Query documents by expiration year from CosmosDB
    
    Args:
        container: CosmosDB container (auto-initialized if None)
        permit_type: Type of permit to filter (only PLO has expiration dates)
        year: Target year
        organization: Organization to filter
        operator: Comparison operator for year
        order_by: Sort order ('latest' or 'earliest')
        
    Returns:
        List of matching documents
        
    Example:
        >>> results = query_documents_by_expiration_year(
        ...     permit_type='PLO',
        ...     year=2025,
        ...     operator='less',
        ...     organization='PGN'
        ... )
    """
    # Auto-initialize container if not provided
    if container is None:
        container = get_permit_container()
    
    query = """
        SELECT c.documentTitle, c.permitType, c.organization, 
               p.expirationDate, p.permitSummary, p.permitNumber
        FROM c
        JOIN p IN c.permits
    """
    
    conditions = []
    parameters = []

    if permit_type:
        conditions.append("c.permitType = @permitType")
        parameters.append(dict(name="@permitType", value=permit_type))

    if operator and year:
        parameters.append(dict(name="@year", value=year))
        if operator == 'greater':
            conditions.append("YEAR(p.expirationDate) >= @year")
        elif operator == 'less':
            conditions.append("YEAR(p.expirationDate) <= @year")
        else:
            conditions.append("YEAR(p.expirationDate) = @year")

    if organization:
        conditions.append("c.organization = @organization")
        parameters.append(dict(name="@organization", value=organization))

    if conditions:
        query += " WHERE " + " AND ".join(conditions)

    try:
        results = container.query_items(
            query=query,
            parameters=parameters,
            enable_cross_partition_query=True
        )

        items = [item for item in results]
        
        if order_by == 'latest':
            items.sort(key=lambda x: x.get('expirationDate', ''), reverse=True)
        else:
            items.sort(key=lambda x: x.get('expirationDate', ''))
        
        logger.info(f"Found {len(items)} documents by expiration year")
        return items
    
    except Exception as e:
        logger.error(f"Error querying documents by expiration year: {e}")
        return []


def query_expired_documents(
    container: Optional[ContainerProxy] = None,
    organization: Optional[str] = None,
    order_by: str = 'latest'
) -> List[Dict[str, Any]]:
    """
    Query documents that have already expired
    
    Args:
        container: CosmosDB container (auto-initialized if None)
        organization: Organization to filter (optional)
        order_by: Sort order ('latest' or 'earliest')
        
    Returns:
        List of expired documents
        
    Example:
        >>> expired = query_expired_documents(organization='KPI')
        >>> for doc in expired:
        ...     print(f"{doc['permitNumber']} expired on {doc['expirationDate']}")
    """
    # Auto-initialize container if not provided
    if container is None:
        container = get_permit_container()
    
    current_date = datetime.now().strftime("%Y-%m-%d")
    parameters = [dict(name="@currentDate", value=current_date)]
    
    query = """
        SELECT c.documentTitle, c.permitType, c.organization, c.filepath,
               p.expirationDate, p.permitSummary, p.permitNumber, p.installation
        FROM c
        JOIN p in c.permits
        WHERE p.expirationDate < @currentDate AND c.permitType = 'PLO'
    """
    
    if organization:
        query += " AND c.organization = @organization"
        parameters.append(dict(name="@organization", value=organization))
    
    try:
        results = container.query_items(
            query=query,
            parameters=parameters,
            enable_cross_partition_query=True
        )
        
        items = [item for item in results]
        
        if order_by == 'latest':
            items.sort(key=lambda x: x.get('expirationDate', ''), reverse=True)
        else:
            items.sort(key=lambda x: x.get('expirationDate', ''))
        
        logger.info(f"Found {len(items)} expired documents")
        return items
    
    except Exception as e:
        logger.error(f"Error querying expired documents: {e}")
        return []


def query_documents_expiring_soon(
    container: Optional[ContainerProxy] = None,
    days: int = 30,
    organization: Optional[str] = None,
    order_by: str = 'earliest'
) -> List[Dict[str, Any]]:
    """
    Query documents that will expire within specified number of days
    
    Args:
        container: CosmosDB container (auto-initialized if None)
        days: Number of days to look ahead (default 30)
        organization: Organization to filter (optional)
        order_by: Sort order ('latest' or 'earliest')
        
    Returns:
        List of documents expiring soon
        
    Example:
        >>> soon_expire = query_documents_expiring_soon(days=60, organization='SHU')
        >>> for doc in soon_expire:
        ...     print(f"{doc['permitNumber']} expires in {days} days")
    """
    # Auto-initialize container if not provided
    if container is None:
        container = get_permit_container()
    
    current_date = datetime.now()
    future_date = datetime.now()
    from datetime import timedelta
    future_date = (current_date + timedelta(days=days)).strftime("%Y-%m-%d")
    current_date_str = current_date.strftime("%Y-%m-%d")
    
    parameters = [
        dict(name="@currentDate", value=current_date_str),
        dict(name="@futureDate", value=future_date)
    ]
    
    query = """
        SELECT c.documentTitle, c.permitType, c.organization, c.filepath,
               p.expirationDate, p.permitSummary, p.permitNumber, p.installation
        FROM c
        JOIN p in c.permits
        WHERE p.expirationDate >= @currentDate 
        AND p.expirationDate <= @futureDate 
        AND c.permitType = 'PLO'
    """
    
    if organization:
        query += " AND c.organization = @organization"
        parameters.append(dict(name="@organization", value=organization))
    
    try:
        results = container.query_items(
            query=query,
            parameters=parameters,
            enable_cross_partition_query=True
        )
        
        items = [item for item in results]
        
        if order_by == 'latest':
            items.sort(key=lambda x: x.get('expirationDate', ''), reverse=True)
        else:
            items.sort(key=lambda x: x.get('expirationDate', ''))
        
        logger.info(f"Found {len(items)} documents expiring within {days} days")
        return items
    
    except Exception as e:
        logger.error(f"Error querying documents expiring soon: {e}")
        return []


def query_permit_by_number(
    container: Optional[ContainerProxy] = None,
    permit_number: str = None
) -> Optional[Dict[str, Any]]:
    """
    Query specific permit by permit number
    
    Args:
        container: CosmosDB container (auto-initialized if None)
        permit_number: Permit number to search for
        
    Returns:
        Permit document if found, None otherwise
        
    Example:
        >>> permit = query_permit_by_number(permit_number='PLO-2024-001')
        >>> if permit:
        ...     print(f"Found: {permit['documentTitle']}")
    """
    # Auto-initialize container if not provided
    if container is None:
        container = get_permit_container()
    
    if not permit_number:
        logger.warning("Permit number is required")
        return None
    
    query = """
        SELECT c.documentTitle, c.permitType, c.organization, c.filepath,
               p.issueDate, p.expirationDate, p.permitSummary, p.permitNumber, p.installation
        FROM c
        JOIN p in c.permits
        WHERE p.permitNumber = @permitNumber
    """
    
    parameters = [dict(name="@permitNumber", value=permit_number)]
    
    try:
        results = container.query_items(
            query=query,
            parameters=parameters,
            enable_cross_partition_query=True
        )
        
        items = [item for item in results]
        
        if items:
            logger.info(f"Found permit: {permit_number}")
            return items[0]
        else:
            logger.warning(f"Permit not found: {permit_number}")
            return None
    
    except Exception as e:
        logger.error(f"Error querying permit by number: {e}")
        return None


def query_permits_by_installation(
    container: Optional[ContainerProxy] = None,
    installation: str = None,
    permit_type: Optional[Literal['PLO', 'KKPR/KKPRL', 'Ijin Lingkungan']] = None
) -> List[Dict[str, Any]]:
    """
    Query permits by installation location
    
    Args:
        container: CosmosDB container (auto-initialized if None)
        installation: Installation location to search for
        permit_type: Type of permit to filter (optional)
        
    Returns:
        List of permits for the installation
        
    Example:
        >>> permits = query_permits_by_installation(
        ...     installation='IT Semarang',
        ...     permit_type='PLO'
        ... )
    """
    # Auto-initialize container if not provided
    if container is None:
        container = get_permit_container()
    
    if not installation:
        logger.warning("Installation name is required")
        return []
    
    query = """
        SELECT c.documentTitle, c.permitType, c.organization, c.filepath,
               p.issueDate, p.expirationDate, p.permitSummary, p.permitNumber, p.installation
        FROM c
        JOIN p in c.permits
        WHERE CONTAINS(LOWER(p.installation), LOWER(@installation))
    """
    
    parameters = [dict(name="@installation", value=installation)]
    
    if permit_type:
        query += " AND c.permitType = @permitType"
        parameters.append(dict(name="@permitType", value=permit_type))
    
    try:
        results = container.query_items(
            query=query,
            parameters=parameters,
            enable_cross_partition_query=True
        )
        
        items = [item for item in results]
        
        logger.info(f"Found {len(items)} permits for installation: {installation}")
        return items
    
    except Exception as e:
        logger.error(f"Error querying permits by installation: {e}")
        return []


def get_all_organizations(
    container: Optional[ContainerProxy] = None
) -> List[str]:
    """
    Get list of all unique organizations in the database
    
    Args:
        container: CosmosDB container (auto-initialized if None)
        
    Returns:
        List of organization names
        
    Example:
        >>> orgs = get_all_organizations()
        >>> print(orgs)  # ['PPN', 'PGN', 'KPI', 'SHU']
    """
    # Auto-initialize container if not provided
    if container is None:
        container = get_permit_container()
    
    query = """
        SELECT DISTINCT VALUE c.organization
        FROM c
        WHERE IS_DEFINED(c.organization)
    """
    
    try:
        results = container.query_items(
            query=query,
            enable_cross_partition_query=True
        )
        
        organizations = [item for item in results]
        
        logger.info(f"Found {len(organizations)} organizations")
        return sorted(organizations)
    
    except Exception as e:
        logger.error(f"Error getting organizations: {e}")
        return []


def get_permit_statistics(
    container: Optional[ContainerProxy] = None
) -> Dict[str, Any]:
    """
    Get overall statistics of permits in the database
    
    Args:
        container: CosmosDB container (auto-initialized if None)
        
    Returns:
        Dictionary containing statistics
        
    Example:
        >>> stats = get_permit_statistics()
        >>> print(f"Total permits: {stats['total_permits']}")
        >>> print(f"Expired: {stats['expired_count']}")
    """
    # Auto-initialize container if not provided
    if container is None:
        container = get_permit_container()
    
    current_date = datetime.now().strftime("%Y-%m-%d")
    
    # Count total permits
    total_query = """
        SELECT VALUE COUNT(1)
        FROM c
        JOIN p IN c.permits
    """
    
    # Count expired permits
    expired_query = """
        SELECT VALUE COUNT(1)
        FROM c
        JOIN p IN c.permits
        WHERE p.expirationDate < @currentDate AND c.permitType = 'PLO'
    """
    
    # Count by type
    type_query = """
        SELECT c.permitType, COUNT(1) as count
        FROM c
        GROUP BY c.permitType
    """
    
    # Count by organization
    org_query = """
        SELECT c.organization, COUNT(1) as count
        FROM c
        GROUP BY c.organization
    """
    
    try:
        # Get total count
        total_results = container.query_items(
            query=total_query,
            enable_cross_partition_query=True
        )
        total_count = next(total_results, 0)
        
        # Get expired count
        expired_results = container.query_items(
            query=expired_query,
            parameters=[dict(name="@currentDate", value=current_date)],
            enable_cross_partition_query=True
        )
        expired_count = next(expired_results, 0)
        
        # Get counts by type
        type_results = container.query_items(
            query=type_query,
            enable_cross_partition_query=True
        )
        by_type = {item['permitType']: item['count'] for item in type_results}
        
        # Get counts by organization
        org_results = container.query_items(
            query=org_query,
            enable_cross_partition_query=True
        )
        by_organization = {item['organization']: item['count'] for item in org_results}
        
        statistics = {
            'total_permits': total_count,
            'expired_count': expired_count,
            'active_count': total_count - expired_count,
            'by_type': by_type,
            'by_organization': by_organization,
            'generated_at': datetime.now().isoformat()
        }
        
        logger.info(f"Generated permit statistics: {total_count} total permits")
        return statistics
    
    except Exception as e:
        logger.error(f"Error getting permit statistics: {e}")
        return {
            'total_permits': 0,
            'expired_count': 0,
            'active_count': 0,
            'by_type': {},
            'by_organization': {},
            'error': str(e)
        }