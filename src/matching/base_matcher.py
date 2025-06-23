# src/matching/base_matcher.py

from abc import ABC, abstractmethod
from typing import List, Set
from src.core.models import URLData, URLMatch, MatchType

class BaseMatcher(ABC):
    """Abstract base class for URL matchers."""
    
    def __init__(self, config=None):
        self.config = config
        self._used_new_urls: Set[str] = set()
    
    @abstractmethod
    def find_matches(self, old_urls: List[URLData], new_urls: List[URLData]) -> List[URLMatch]:
        """
        Find matches between old and new URLs.
        
        Args:
            old_urls: List of old URLs to match
            new_urls: List of new URLs to match against
            
        Returns:
            List of URLMatch objects
        """
        pass
    
    @abstractmethod
    def get_match_type(self) -> MatchType:
        """Return the type of matching this matcher performs."""
        pass
    
    def is_new_url_available(self, new_url: URLData) -> bool:
        """Check if a new URL is available for matching (not already used)."""
        return new_url.address not in self._used_new_urls
    
    def mark_new_url_as_used(self, new_url: URLData):
        """Mark a new URL as used to prevent duplicate matches."""
        self._used_new_urls.add(new_url.address)
    
    def reset_used_urls(self):
        """Reset the used URLs tracker."""
        self._used_new_urls.clear()
    
    def create_match(self, 
                    old_url: URLData, 
                    new_url: URLData, 
                    confidence_score: int,
                    similarity_score: float = None) -> URLMatch:
        """
        Helper method to create a URLMatch object.
        
        Args:
            old_url: The old URL
            new_url: The new URL  
            confidence_score: Confidence score (0-100)
            similarity_score: Optional similarity score
            
        Returns:
            URLMatch object
        """
        return URLMatch(
            old_url=old_url,
            new_url=new_url,
            match_type=self.get_match_type(),
            confidence_score=confidence_score,
            similarity_score=similarity_score
        )
    
    def filter_valid_urls(self, urls: List[URLData], require_field: str = None) -> List[URLData]:
        """
        Filter URLs to only include valid ones.
        
        Args:
            urls: List of URLs to filter
            require_field: Optional field that must be non-empty
            
        Returns:
            Filtered list of URLs
        """
        valid_urls = []
        for url in urls:
            if require_field:
                field_value = getattr(url, require_field, None)
                if not field_value or (isinstance(field_value, str) and not field_value.strip()):
                    continue
            valid_urls.append(url)
        return valid_urls