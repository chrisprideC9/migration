# src/scoring/priority_scorer.py

from typing import Optional
from src.core.models import URLData

class PriorityScorer:
    """Calculates priority scores for URLs based on SEO metrics."""
    
    def __init__(self, config=None, has_gsc_data: bool = False):
        self.config = config
        self.has_gsc_data = has_gsc_data
        
        # Set weights based on whether GSC data is available
        if self.config:
            if has_gsc_data:
                self.weights = self.config.scoring.PRIORITY_WEIGHTS_WITH_GSC
            else:
                self.weights = self.config.scoring.PRIORITY_WEIGHTS_WITHOUT_GSC
            
            # Normalization factors
            self.traffic_norm = self.config.scoring.TRAFFIC_NORMALIZATION_FACTOR
            self.traffic_value_norm = self.config.scoring.TRAFFIC_VALUE_NORMALIZATION_FACTOR
            self.keywords_norm = self.config.scoring.KEYWORDS_NORMALIZATION_FACTOR
            self.impressions_norm = self.config.scoring.IMPRESSIONS_NORMALIZATION_FACTOR
        else:
            # Default weights
            if has_gsc_data:
                self.weights = {
                    'traffic': 0.4,
                    'traffic_value': 0.3,
                    'keywords': 0.2,
                    'impressions': 0.1
                }
            else:
                self.weights = {
                    'traffic': 0.5,
                    'traffic_value': 0.3,
                    'keywords': 0.2
                }
            
            # Default normalization factors
            self.traffic_norm = 1000.0
            self.traffic_value_norm = 100.0
            self.keywords_norm = 10.0
            self.impressions_norm = 10000.0
    
    def calculate_score(self, url_data: URLData) -> float:
        """
        Calculate priority score for a URL.
        
        Args:
            url_data: URLData object containing SEO metrics
            
        Returns:
            Priority score (higher = more important)
        """
        score = 0.0
        
        # Normalize and weight traffic
        if 'traffic' in self.weights:
            normalized_traffic = min(1.0, url_data.traffic / self.traffic_norm)
            score += self.weights['traffic'] * normalized_traffic
        
        # Normalize and weight traffic value
        if 'traffic_value' in self.weights:
            normalized_value = min(1.0, url_data.traffic_value / self.traffic_value_norm)
            score += self.weights['traffic_value'] * normalized_value
        
        # Normalize and weight keywords
        if 'keywords' in self.weights:
            normalized_keywords = min(1.0, url_data.keywords / self.keywords_norm)
            score += self.weights['keywords'] * normalized_keywords
        
        # Normalize and weight impressions (if GSC data available)
        if 'impressions' in self.weights and self.has_gsc_data:
            normalized_impressions = min(1.0, url_data.impressions / self.impressions_norm)
            score += self.weights['impressions'] * normalized_impressions
        
        # Apply bonuses/penalties
        score = self._apply_bonuses_and_penalties(url_data, score)
        
        return round(score, 4)
    
    def _apply_bonuses_and_penalties(self, url_data: URLData, base_score: float) -> float:
        """Apply additional bonuses and penalties to the base score."""
        score = base_score
        
        # Bonus for high-performing URLs
        if url_data.traffic > 1000:
            score *= 1.1  # 10% bonus for high traffic
        
        if url_data.keywords > 50:
            score *= 1.05  # 5% bonus for many keywords
        
        # Penalty for URLs with no SEO data
        if (url_data.traffic == 0 and 
            url_data.traffic_value == 0 and 
            url_data.keywords == 0):
            score *= 0.5  # 50% penalty for no SEO data
        
        # Bonus for URLs with GSC data
        if self.has_gsc_data and url_data.impressions > 0:
            score *= 1.02  # Small bonus for having GSC data
        
        return score
    
    def get_priority_category(self, score: float) -> str:
        """
        Categorize URLs based on priority score.
        
        Args:
            score: Priority score
            
        Returns:
            Priority category string
        """
        if score >= 0.8:
            return "Critical"
        elif score >= 0.6:
            return "High"
        elif score >= 0.4:
            return "Medium"
        elif score >= 0.2:
            return "Low"
        else:
            return "Minimal"
    
    def calculate_batch_scores(self, url_list: list[URLData]) -> list[float]:
        """
        Calculate priority scores for a batch of URLs.
        
        Args:
            url_list: List of URLData objects
            
        Returns:
            List of priority scores
        """
        return [self.calculate_score(url) for url in url_list]
    
    def get_top_priority_urls(self, url_list: list[URLData], top_n: int = 10) -> list[tuple[URLData, float]]:
        """
        Get the top N priority URLs.
        
        Args:
            url_list: List of URLData objects
            top_n: Number of top URLs to return
            
        Returns:
            List of tuples (URLData, priority_score) sorted by priority
        """
        url_scores = [(url, self.calculate_score(url)) for url in url_list]
        return sorted(url_scores, key=lambda x: x[1], reverse=True)[:top_n]
    
    def update_weights(self, new_weights: dict):
        """
        Update the scoring weights.
        
        Args:
            new_weights: Dictionary of new weights
        """
        # Validate weights sum to 1.0
        total_weight = sum(new_weights.values())
        if abs(total_weight - 1.0) > 0.01:
            raise ValueError(f"Weights must sum to 1.0, got {total_weight}")
        
        self.weights.update(new_weights)
    
    def get_score_breakdown(self, url_data: URLData) -> dict:
        """
        Get a detailed breakdown of how the priority score was calculated.
        
        Args:
            url_data: URLData object
            
        Returns:
            Dictionary with score breakdown
        """
        breakdown = {}
        
        if 'traffic' in self.weights:
            normalized_traffic = min(1.0, url_data.traffic / self.traffic_norm)
            breakdown['traffic'] = {
                'raw_value': url_data.traffic,
                'normalized': normalized_traffic,
                'weight': self.weights['traffic'],
                'contribution': self.weights['traffic'] * normalized_traffic
            }
        
        if 'traffic_value' in self.weights:
            normalized_value = min(1.0, url_data.traffic_value / self.traffic_value_norm)
            breakdown['traffic_value'] = {
                'raw_value': url_data.traffic_value,
                'normalized': normalized_value,
                'weight': self.weights['traffic_value'],
                'contribution': self.weights['traffic_value'] * normalized_value
            }
        
        if 'keywords' in self.weights:
            normalized_keywords = min(1.0, url_data.keywords / self.keywords_norm)
            breakdown['keywords'] = {
                'raw_value': url_data.keywords,
                'normalized': normalized_keywords,
                'weight': self.weights['keywords'],
                'contribution': self.weights['keywords'] * normalized_keywords
            }
        
        if 'impressions' in self.weights and self.has_gsc_data:
            normalized_impressions = min(1.0, url_data.impressions / self.impressions_norm)
            breakdown['impressions'] = {
                'raw_value': url_data.impressions,
                'normalized': normalized_impressions,
                'weight': self.weights['impressions'],
                'contribution': self.weights['impressions'] * normalized_impressions
            }
        
        breakdown['total_score'] = self.calculate_score(url_data)
        breakdown['priority_category'] = self.get_priority_category(breakdown['total_score'])
        
        return breakdown