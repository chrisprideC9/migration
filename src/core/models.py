# src/core/models.py

from dataclasses import dataclass
from typing import List, Optional, Dict, Any
import numpy as np
import pandas as pd
from enum import Enum

class MatchType(Enum):
    """Types of URL matches."""
    EXACT_PATH = "Exact Path"
    SIMILAR_PATH = "Similar Path"  
    VECTOR_SIMILARITY = "Vector Similarity"
    EXACT_H1 = "Exact H1"
    SIMILAR_H1 = "Similar H1"
    AI_MATCH = "AI Match"
    NO_MATCH = "No Match"

@dataclass
class URLData:
    """Represents a URL with its associated data."""
    address: str
    title: Optional[str] = None
    meta_description: Optional[str] = None
    h1: Optional[str] = None
    path: Optional[str] = None
    embedding: Optional[np.ndarray] = None
    status_code: Optional[int] = None
    
    # SEO data
    traffic: float = 0.0
    traffic_value: float = 0.0
    keywords: int = 0
    clicks: int = 0
    impressions: int = 0
    
    def __post_init__(self):
        """Extract path from address if not provided."""
        if self.path is None and self.address:
            from src.utils.url_utils import extract_path
            self.path = extract_path(self.address)

@dataclass
class URLMatch:
    """Represents a match between old and new URLs."""
    old_url: URLData
    new_url: URLData
    match_type: MatchType
    confidence_score: int
    similarity_score: Optional[float] = None
    priority_score: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert match to dictionary for DataFrame creation."""
        return {
            'Address_old': self.old_url.address,
            'Address_new': self.new_url.address,
            'Match_Type': self.match_type.value,
            'Confidence_Score': self.confidence_score,
            'Priority_Score': self.priority_score,
            'Title_old': self.old_url.title,
            'Meta Description_old': self.old_url.meta_description,
            'H1_old': self.old_url.h1,
            'Traffic': self.old_url.traffic,
            'Traffic value': self.old_url.traffic_value,
            'Keywords': self.old_url.keywords,
            'Clicks': self.old_url.clicks,
            'Impressions': self.old_url.impressions
        }

@dataclass
class MatchingResults:
    """Container for all matching results."""
    matches: List[URLMatch]
    unmatched_urls: List[URLData]
    has_gsc_data: bool = False
    
    def get_matches_df(self) -> pd.DataFrame:
        """Get matches as a pandas DataFrame."""
        if not self.matches:
            return pd.DataFrame()
        return pd.DataFrame([match.to_dict() for match in self.matches])
    
    def get_unmatched_df(self) -> pd.DataFrame:
        """Get unmatched URLs as a pandas DataFrame."""
        if not self.unmatched_urls:
            return pd.DataFrame()
        
        unmatched_data = []
        for url in self.unmatched_urls:
            unmatched_data.append({
                'Address_old': url.address,
                'Address_new': 'NOT FOUND',
                'Match_Type': MatchType.NO_MATCH.value,
                'Confidence_Score': 0,
                'Priority_Score': self._calculate_priority_score(url),
                'Title_old': url.title,
                'Meta Description_old': url.meta_description,
                'H1_old': url.h1,
                'Traffic': url.traffic,
                'Traffic value': url.traffic_value,
                'Keywords': url.keywords,
                'Clicks': url.clicks,
                'Impressions': url.impressions
            })
        return pd.DataFrame(unmatched_data)
    
    def _calculate_priority_score(self, url: URLData) -> float:
        """Calculate priority score for a URL."""
        from src.scoring.priority_scorer import PriorityScorer
        scorer = PriorityScorer(has_gsc_data=self.has_gsc_data)
        return scorer.calculate_score(url)
    
    def get_most_confident_matches(self) -> pd.DataFrame:
        """Get only the most confident match for each old URL."""
        matches_df = self.get_matches_df()
        if matches_df.empty:
            return matches_df
        
        return matches_df.loc[
            matches_df.groupby('Address_old')['Confidence_Score'].idxmax()
        ].reset_index(drop=True)

@dataclass
class ProcessingStats:
    """Statistics about the processing pipeline."""
    total_old_urls: int = 0
    total_new_urls: int = 0
    exact_matches: int = 0
    fuzzy_matches: int = 0
    vector_matches: int = 0
    h1_matches: int = 0
    ai_matches: int = 0
    unmatched: int = 0
    processing_time: float = 0.0
    
    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of the statistics."""
        total_matches = (self.exact_matches + self.fuzzy_matches + 
                        self.vector_matches + self.h1_matches + self.ai_matches)
        
        return {
            'Total Old URLs': self.total_old_urls,
            'Total New URLs': self.total_new_urls,
            'Total Matches': total_matches,
            'Match Rate': f"{(total_matches / self.total_old_urls * 100):.1f}%" if self.total_old_urls > 0 else "0%",
            'Exact Matches': self.exact_matches,
            'Fuzzy Matches': self.fuzzy_matches,
            'Vector Matches': self.vector_matches,
            'H1 Matches': self.h1_matches,
            'AI Matches': self.ai_matches,
            'Unmatched': self.unmatched,
            'Processing Time': f"{self.processing_time:.2f}s"
        }