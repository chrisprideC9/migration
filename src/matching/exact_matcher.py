# src/matching/exact_matcher.py

from typing import List, Dict
from src.matching.base_matcher import BaseMatcher
from src.core.models import URLData, URLMatch, MatchType

class ExactMatcher(BaseMatcher):
    """Matcher for exact path matches between old and new URLs."""
    
    def get_match_type(self) -> MatchType:
        """Return the match type for exact matching."""
        return MatchType.EXACT_PATH
    
    def find_matches(self, old_urls: List[URLData], new_urls: List[URLData]) -> List[URLMatch]:
        """
        Find exact path matches between old and new URLs.
        
        Args:
            old_urls: List of old URLs to match
            new_urls: List of new URLs to match against
            
        Returns:
            List of URLMatch objects for exact matches
        """
        matches = []
        
        # Create a lookup dictionary for new URLs by path
        new_urls_by_path: Dict[str, URLData] = {}
        for new_url in new_urls:
            if new_url.path and self.is_new_url_available(new_url):
                new_urls_by_path[new_url.path] = new_url
        
        # Find exact matches
        for old_url in old_urls:
            if not old_url.path:
                continue
                
            if old_url.path in new_urls_by_path:
                new_url = new_urls_by_path[old_url.path]
                
                # Create the match with high confidence
                match = self.create_match(
                    old_url=old_url,
                    new_url=new_url,
                    confidence_score=self.config.confidence.EXACT_MATCH_SCORE if self.config else 100,
                    similarity_score=1.0
                )
                
                matches.append(match)
                self.mark_new_url_as_used(new_url)
        
        return matches