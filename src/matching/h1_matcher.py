# src/matching/h1_matcher.py

from typing import List, Dict
from rapidfuzz import process, fuzz
from src.matching.base_matcher import BaseMatcher
from src.core.models import URLData, URLMatch, MatchType

class H1Matcher(BaseMatcher):
    """Matcher for H1 header similarity between old and new URLs."""
    
    def __init__(self, config=None, threshold: int = 90):
        super().__init__(config)
        self.threshold = threshold
        self.scorer = fuzz.token_sort_ratio  # Good for H1 matching
    
    def get_match_type(self) -> MatchType:
        """Return the match type for H1 matching."""
        return MatchType.EXACT_H1  # Will be updated based on exact vs similar
    
    def find_matches(self, old_urls: List[URLData], new_urls: List[URLData]) -> List[URLMatch]:
        """
        Find H1 header matches between old and new URLs.
        
        Args:
            old_urls: List of old URLs to match
            new_urls: List of new URLs to match against
            
        Returns:
            List of URLMatch objects for H1 matches
        """
        matches = []
        
        # Filter URLs with valid H1 headers
        old_urls_with_h1 = self.filter_valid_urls(old_urls, require_field='h1')
        new_urls_with_h1 = [
            url for url in new_urls 
            if url.h1 and url.h1.strip() and self.is_new_url_available(url)
        ]
        
        if not old_urls_with_h1 or not new_urls_with_h1:
            return matches
        
        # Create lookup dictionary for exact matches first
        new_h1_lookup: Dict[str, URLData] = {}
        for new_url in new_urls_with_h1:
            h1_normalized = new_url.h1.strip().lower()
            if h1_normalized not in new_h1_lookup:
                new_h1_lookup[h1_normalized] = new_url
        
        # Find matches for each old URL
        for old_url in old_urls_with_h1:
            old_h1_normalized = old_url.h1.strip().lower()
            
            # Try exact match first
            if old_h1_normalized in new_h1_lookup:
                new_url = new_h1_lookup[old_h1_normalized]
                if self.is_new_url_available(new_url):
                    match = self._create_exact_h1_match(old_url, new_url)
                    matches.append(match)
                    self.mark_new_url_as_used(new_url)
                    continue
            
            # Try fuzzy match if no exact match found
            fuzzy_match = self._find_fuzzy_h1_match(old_url, new_urls_with_h1)
            if fuzzy_match:
                matches.append(fuzzy_match)
        
        return matches
    
    def _create_exact_h1_match(self, old_url: URLData, new_url: URLData) -> URLMatch:
        """Create a match for exact H1 header match."""
        confidence_score = self.config.confidence.EXACT_H1_SCORE if self.config else 80
        
        match = URLMatch(
            old_url=old_url,
            new_url=new_url,
            match_type=MatchType.EXACT_H1,
            confidence_score=confidence_score,
            similarity_score=1.0
        )
        return match
    
    def _find_fuzzy_h1_match(self, old_url: URLData, available_new_urls: List[URLData]) -> URLMatch:
        """Find the best fuzzy H1 match for an old URL."""
        # Get H1 headers from available new URLs
        available_h1s = [url.h1 for url in available_new_urls if self.is_new_url_available(url)]
        
        if not available_h1s:
            return None
        
        # Find best fuzzy match
        match_result = process.extractOne(
            old_url.h1,
            available_h1s,
            scorer=self.scorer
        )
        
        if not match_result or match_result[1] < self.threshold:
            return None
        
        best_h1, similarity_score, _ = match_result
        
        # Find the corresponding new URL
        new_url = next(
            (url for url in available_new_urls 
             if url.h1 == best_h1 and self.is_new_url_available(url)),
            None
        )
        
        if not new_url:
            return None
        
        # Calculate confidence score
        confidence_score = self._calculate_fuzzy_confidence_score(similarity_score)
        
        match = URLMatch(
            old_url=old_url,
            new_url=new_url,
            match_type=MatchType.SIMILAR_H1,
            confidence_score=confidence_score,
            similarity_score=similarity_score / 100.0
        )
        
        self.mark_new_url_as_used(new_url)
        return match
    
    def _calculate_fuzzy_confidence_score(self, similarity_score: float) -> int:
        """
        Calculate confidence score for fuzzy H1 matches.
        
        Args:
            similarity_score: Similarity score from rapidfuzz (0-100)
            
        Returns:
            Confidence score (0-100)
        """
        if self.config:
            base_score = self.config.confidence.SIMILAR_H1_SCORE
        else:
            base_score = 60
        
        # Adjust confidence based on how much above threshold the score is
        threshold_bonus = max(0, (similarity_score - self.threshold) / (100 - self.threshold) * 15)
        
        return min(100, int(base_score + threshold_bonus))
    
    def _clean_h1_text(self, h1_text: str) -> str:
        """Clean H1 text for better matching."""
        if not h1_text:
            return ""
        
        # Basic cleaning
        cleaned = h1_text.strip()
        
        # Remove common prefixes/suffixes that might interfere with matching
        prefixes_to_remove = ["Home | ", "Home - ", "Welcome to ", "Welcome - "]
        suffixes_to_remove = [" | Home", " - Home", " | Company Name", " - Company Name"]
        
        for prefix in prefixes_to_remove:
            if cleaned.startswith(prefix):
                cleaned = cleaned[len(prefix):]
                break
        
        for suffix in suffixes_to_remove:
            if cleaned.endswith(suffix):
                cleaned = cleaned[:-len(suffix)]
                break
        
        return cleaned.strip()