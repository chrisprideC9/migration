# src/matching/fuzzy_matcher.py

from typing import List
from rapidfuzz import process, fuzz
from src.matching.base_matcher import BaseMatcher
from src.core.models import URLData, URLMatch, MatchType

class FuzzyMatcher(BaseMatcher):
    """Matcher for fuzzy path similarity between old and new URLs."""
    
    def __init__(self, config=None, threshold: int = 90):
        super().__init__(config)
        self.threshold = threshold
        self.scorer = getattr(fuzz, config.matching.FUZZY_SCORER if config else 'ratio')
    
    def get_match_type(self) -> MatchType:
        """Return the match type for fuzzy matching."""
        return MatchType.SIMILAR_PATH
    
    def find_matches(self, old_urls: List[URLData], new_urls: List[URLData]) -> List[URLMatch]:
        """
        Find fuzzy path matches between old and new URLs.
        
        Args:
            old_urls: List of old URLs to match
            new_urls: List of new URLs to match against
            
        Returns:
            List of URLMatch objects for fuzzy matches
        """
        matches = []
        
        # Filter URLs with valid paths
        old_urls_with_paths = self.filter_valid_urls(old_urls, require_field='path')
        new_urls_with_paths = [url for url in new_urls if url.path and self.is_new_url_available(url)]
        
        if not old_urls_with_paths or not new_urls_with_paths:
            return matches
        
        # Create list of new URL paths for fuzzy matching
        new_paths = [url.path for url in new_urls_with_paths]
        
        # Find fuzzy matches for each old URL
        for old_url in old_urls_with_paths:
            if not old_url.path:
                continue
            
            # Find best match using rapidfuzz
            match_result = process.extractOne(
                old_url.path, 
                new_paths, 
                scorer=self.scorer
            )
            
            if match_result and match_result[1] >= self.threshold:
                best_match_path, similarity_score, _ = match_result
                
                # Find the corresponding new URL
                new_url = next(
                    (url for url in new_urls_with_paths if url.path == best_match_path),
                    None
                )
                
                if new_url and self.is_new_url_available(new_url):
                    # Create the match
                    confidence_score = self._calculate_confidence_score(similarity_score)
                    
                    match = self.create_match(
                        old_url=old_url,
                        new_url=new_url,
                        confidence_score=confidence_score,
                        similarity_score=similarity_score / 100.0  # Convert to 0-1 scale
                    )
                    
                    matches.append(match)
                    self.mark_new_url_as_used(new_url)
        
        return matches
    
    def _calculate_confidence_score(self, similarity_score: float) -> int:
        """
        Calculate confidence score based on similarity score.
        
        Args:
            similarity_score: Similarity score from rapidfuzz (0-100)
            
        Returns:
            Confidence score (0-100)
        """
        if self.config:
            base_score = self.config.confidence.FUZZY_MATCH_SCORE
        else:
            base_score = 50
        
        # Adjust confidence based on how much above threshold the score is
        threshold_bonus = max(0, (similarity_score - self.threshold) / (100 - self.threshold) * 20)
        
        return min(100, int(base_score + threshold_bonus))