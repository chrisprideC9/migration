# src/utils/url_utils.py

from urllib.parse import urlparse
from typing import Optional

def extract_path(url: str) -> str:
    """
    Extract the path component from a URL, stripping trailing slash.
    
    Args:
        url: The URL to extract the path from
        
    Returns:
        The path component of the URL, or empty string if invalid
        
    Examples:
        >>> extract_path("https://example.com/page/")
        "/page"
        >>> extract_path("https://example.com/")
        ""
        >>> extract_path("invalid-url")
        ""
    """
    try:
        parsed = urlparse(url)
        path = parsed.path.rstrip('/')
        return path if path != '/' else ''
    except Exception:
        return ''

def normalize_url(url: str) -> str:
    """
    Normalize a URL by converting to lowercase and removing trailing slash.
    
    Args:
        url: The URL to normalize
        
    Returns:
        Normalized URL
        
    Examples:
        >>> normalize_url("HTTPS://Example.COM/Page/")
        "https://example.com/page"
    """
    try:
        return url.lower().rstrip('/')
    except Exception:
        return url

def is_valid_url(url: str) -> bool:
    """
    Check if a URL is valid and has the required components.
    
    Args:
        url: The URL to validate
        
    Returns:
        True if the URL is valid, False otherwise
        
    Examples:
        >>> is_valid_url("https://example.com/page")
        True
        >>> is_valid_url("not-a-url")
        False
    """
    try:
        parsed = urlparse(url)
        return bool(parsed.scheme and parsed.netloc)
    except Exception:
        return False

def get_domain(url: str) -> Optional[str]:
    """
    Extract the domain from a URL.
    
    Args:
        url: The URL to extract the domain from
        
    Returns:
        The domain, or None if the URL is invalid
        
    Examples:
        >>> get_domain("https://example.com/page")
        "example.com"
        >>> get_domain("invalid-url")
        None
    """
    try:
        parsed = urlparse(url)
        return parsed.netloc.lower() if parsed.netloc else None
    except Exception:
        return None

def urls_same_domain(url1: str, url2: str) -> bool:
    """
    Check if two URLs are from the same domain.
    
    Args:
        url1: First URL
        url2: Second URL
        
    Returns:
        True if both URLs are from the same domain, False otherwise
    """
    domain1 = get_domain(url1)
    domain2 = get_domain(url2)
    return domain1 is not None and domain1 == domain2

def clean_url_for_comparison(url: str) -> str:
    """
    Clean a URL for comparison purposes by removing common variations.
    
    Args:
        url: The URL to clean
        
    Returns:
        Cleaned URL suitable for comparison
    """
    cleaned = normalize_url(url)
    
    # Remove common query parameters that don't affect content
    try:
        parsed = urlparse(cleaned)
        # Reconstruct without query parameters for cleaner comparison
        cleaned = f"{parsed.scheme}://{parsed.netloc}{parsed.path}"
    except Exception:
        pass
    
    return cleaned