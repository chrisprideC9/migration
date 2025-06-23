# src/matching/vector_matcher.py

import numpy as np
from typing import List
from sklearn.metrics.pairwise import cosine_similarity
from src.matching.base_matcher import BaseMatcher
from src.core.models import URLData, URLMatch, MatchType

class VectorMatcher(BaseMatcher):
    """Matcher for vector similarity between old and new URLs using embeddings."""
    
    def __init__(self, config=None):
        super().__init__(config)
        self.similarity_method = config.matching.VECTOR_SIMILARITY_METHOD if config else "cosine"
        self.top_n = config.matching.VECTOR_TOP_N_MATCHES if config else 1
    
    def get_match_type(self) -> MatchType:
        """Return the match type for vector matching."""
        return MatchType.VECTOR_SIMILARITY
    
    def find_matches(self, old_urls: List[URLData], new_urls: List[URLData]) -> List[URLMatch]:
        """
        Find vector similarity matches between old and new URLs.
        
        Args:
            old_urls: List of old URLs to match
            new_urls: List of new URLs to match against
            
        Returns:
            List of URLMatch objects for vector matches
        """
        matches = []
        
        # Filter URLs with valid embeddings
        old_urls_with_embeddings = self.filter_valid_urls(old_urls, require_field='embedding')
        new_urls_with_embeddings = [
            url for url in new_urls 
            if url.embedding is not None and len(url.embedding) > 0 and self.is_new_url_available(url)
        ]
        
        if not old_urls_with_embeddings or not new_urls_with_embeddings:
            return matches
        
        # Stack embeddings into matrices
        try:
            old_embeddings = np.stack([url.embedding for url in old_urls_with_embeddings])
            new_embeddings = np.stack([url.embedding for url in new_urls_with_embeddings])
        except ValueError as e:
            # Handle embedding dimension mismatch
            return matches
        
        # Calculate similarity matrix
        if self.similarity_method == "cosine":
            similarity_matrix = cosine_similarity(old_embeddings, new_embeddings)
        else:
            # Add other similarity methods as needed
            similarity_matrix = cosine_similarity(old_embeddings, new_embeddings)
        
        # Find best matches for each old URL
        for i, old_url in enumerate(old_urls_with_embeddings):
            # Get similarity scores for this old URL
            similarities = similarity_matrix[i]
            
            # Find top N matches
            top_indices = np.argsort(-similarities)[:self.top_n]
            
            for j in top_indices:
                new_url = new_urls_with_embeddings[j]
                similarity_score = similarities[j]
                
                # Check if this new URL is still available
                if not self.is_new_url_available(new_url):
                    continue
                
                # Calculate confidence score based on similarity
                confidence_score = self._calculate_confidence_score(similarity_score)
                
                # Only create match if confidence is reasonable
                if confidence_score > 0:
                    match = self.create_match(
                        old_url=old_url,
                        new_url=new_url,
                        confidence_score=confidence_score,
                        similarity_score=float(similarity_score)
                    )
                    
                    matches.append(match)
                    self.mark_new_url_as_used(new_url)
                    break  # Only take the best match for each old URL
        
        return matches
    
    def _calculate_confidence_score(self, similarity_score: float) -> int:
        """
        Calculate confidence score based on vector similarity.
        
        Args:
            similarity_score: Cosine similarity score (0-1)
            
        Returns:
            Confidence score (0-100)
        """
        if self.config:
            high_threshold = self.config.confidence.VECTOR_HIGH_CONFIDENCE_THRESHOLD
            high_score = self.config.confidence.VECTOR_HIGH_CONFIDENCE_SCORE
            medium_threshold = self.config.confidence.VECTOR_MEDIUM_CONFIDENCE_THRESHOLD
            medium_score = self.config.confidence.VECTOR_MEDIUM_CONFIDENCE_SCORE
            low_score = self.config.confidence.VECTOR_LOW_CONFIDENCE_SCORE
        else:
            high_threshold = 0.9
            high_score = 90
            medium_threshold = 0.75
            medium_score = 60
            low_score = 30
        
        if similarity_score >= high_threshold:
            return high_score
        elif similarity_score >= medium_threshold:
            return medium_score
        else:
            return low_score
    
    def _normalize_embeddings(self, embeddings: np.ndarray) -> np.ndarray:
        """Normalize embeddings for better similarity calculation."""
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1  # Avoid division by zero
        return embeddings / norms