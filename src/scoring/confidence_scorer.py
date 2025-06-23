# src/scoring/confidence_scorer.py

from src.core.models import URLMatch, MatchType, URLData
from src.utils.url_utils import urls_same_domain

class ConfidenceScorer:
    """Calculates and adjusts confidence scores for URL matches."""
    
    def __init__(self, config=None):
        self.config = config
        
        if self.config:
            self.same_domain_bonus = self.config.confidence.SAME_DOMAIN_BONUS
            self.different_domain_penalty = self.config.confidence.DIFFERENT_DOMAIN_PENALTY
            self.empty_content_penalty = self.config.confidence.EMPTY_CONTENT_PENALTY
        else:
            self.same_domain_bonus = 10
            self.different_domain_penalty = -5
            self.empty_content_penalty = -10
    
    def calculate_confidence_score(self, match: URLMatch) -> int:
        """
        Calculate the final confidence score for a URL match.
        
        Args:
            match: URLMatch object
            
        Returns:
            Adjusted confidence score (0-100)
        """
        base_score = match.confidence_score
        adjusted_score = base_score
        
        # Apply domain-based adjustments
        adjusted_score = self._apply_domain_adjustments(match, adjusted_score)
        
        # Apply content-based adjustments
        adjusted_score = self._apply_content_adjustments(match, adjusted_score)
        
        # Apply match-type specific adjustments
        adjusted_score = self._apply_match_type_adjustments(match, adjusted_score)
        
        # Ensure score stays within bounds
        return max(0, min(100, int(adjusted_score)))
    
    def _apply_domain_adjustments(self, match: URLMatch, score: int) -> int:
        """Apply adjustments based on domain similarity."""
        old_url = match.old_url.address
        new_url = match.new_url.address
        
        if urls_same_domain(old_url, new_url):
            # Same domain - likely a good match
            return score + self.same_domain_bonus
        else:
            # Different domain - slightly less confident
            return score + self.different_domain_penalty
    
    def _apply_content_adjustments(self, match: URLMatch, score: int) -> int:
        """Apply adjustments based on content availability."""
        adjustments = 0
        
        # Penalty for missing content in old URL
        if not match.old_url.title or not match.old_url.title.strip():
            adjustments += self.empty_content_penalty // 2
        
        if not match.old_url.h1 or not match.old_url.h1.strip():
            adjustments += self.empty_content_penalty // 2
        
        # Bonus for rich content in old URL
        if (match.old_url.title and match.old_url.h1 and 
            match.old_url.meta_description):
            adjustments += 5
        
        return score + adjustments
    
    def _apply_match_type_adjustments(self, match: URLMatch, score: int) -> int:
        """Apply adjustments based on the type of match."""
        # Exact matches get a small bonus for being perfect
        if match.match_type == MatchType.EXACT_PATH:
            return score + 5
        
        # H1 matches get bonus if they're also path-similar
        if match.match_type in [MatchType.EXACT_H1, MatchType.SIMILAR_H1]:
            if self._paths_similar(match.old_url, match.new_url):
                return score + 10
        
        # Vector matches get bonus if similarity is very high
        if (match.match_type == MatchType.VECTOR_SIMILARITY and 
            match.similarity_score and match.similarity_score > 0.95):
            return score + 5
        
        return score
    
    def _paths_similar(self, old_url: URLData, new_url: URLData) -> bool:
        """Check if URL paths are similar."""
        if not old_url.path or not new_url.path:
            return False
        
        # Simple similarity check
        old_path_parts = old_url.path.strip('/').split('/')
        new_path_parts = new_url.path.strip('/').split('/')
        
        # Check for common path segments
        common_segments = set(old_path_parts) & set(new_path_parts)
        total_segments = set(old_path_parts) | set(new_path_parts)
        
        if total_segments:
            similarity = len(common_segments) / len(total_segments)
            return similarity > 0.5
        
        return False
    
    def get_confidence_category(self, score: int) -> str:
        """
        Categorize confidence score into human-readable categories.
        
        Args:
            score: Confidence score (0-100)
            
        Returns:
            Confidence category string
        """
        if score >= 90:
            return "Very High"
        elif score >= 75:
            return "High"
        elif score >= 60:
            return "Medium"
        elif score >= 40:
            return "Low"
        else:
            return "Very Low"
    
    def calculate_batch_confidence_scores(self, matches: list[URLMatch]) -> list[int]:
        """
        Calculate confidence scores for a batch of matches.
        
        Args:
            matches: List of URLMatch objects
            
        Returns:
            List of adjusted confidence scores
        """
        return [self.calculate_confidence_score(match) for match in matches]
    
    def filter_by_confidence(self, matches: list[URLMatch], min_confidence: int = 50) -> list[URLMatch]:
        """
        Filter matches by minimum confidence score.
        
        Args:
            matches: List of URLMatch objects
            min_confidence: Minimum confidence score to keep
            
        Returns:
            Filtered list of matches
        """
        filtered_matches = []
        
        for match in matches:
            adjusted_score = self.calculate_confidence_score(match)
            if adjusted_score >= min_confidence:
                # Update the match with the adjusted score
                match.confidence_score = adjusted_score
                filtered_matches.append(match)
        
        return filtered_matches
    
    def get_confidence_distribution(self, matches: list[URLMatch]) -> dict:
        """
        Get distribution of confidence scores across matches.
        
        Args:
            matches: List of URLMatch objects
            
        Returns:
            Dictionary with confidence distribution
        """
        if not matches:
            return {}
        
        scores = [self.calculate_confidence_score(match) for match in matches]
        
        categories = {
            "Very High (90-100)": sum(1 for s in scores if s >= 90),
            "High (75-89)": sum(1 for s in scores if 75 <= s < 90),
            "Medium (60-74)": sum(1 for s in scores if 60 <= s < 75),
            "Low (40-59)": sum(1 for s in scores if 40 <= s < 60),
            "Very Low (0-39)": sum(1 for s in scores if s < 40)
        }
        
        return {
            "categories": categories,
            "average_score": sum(scores) / len(scores),
            "min_score": min(scores),
            "max_score": max(scores),
            "total_matches": len(matches)
        }