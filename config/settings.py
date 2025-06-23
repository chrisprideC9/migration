# config/settings.py

import os
from dataclasses import dataclass
from typing import List, Dict, Any

@dataclass
class FileConfig:
    """Configuration for file handling."""
    REQUIRED_FILES: List[str] = None
    OPTIONAL_FILES: List[str] = None
    
    def __post_init__(self):
        if self.REQUIRED_FILES is None:
            self.REQUIRED_FILES = ['top_pages.csv', 'old_vectors.csv', 'new_vectors.csv']
        if self.OPTIONAL_FILES is None:
            self.OPTIONAL_FILES = ['impressions.csv']

@dataclass 
class ColumnConfig:
    """Configuration for expected CSV columns."""
    TOP_PAGES_REQUIRED: List[str] = None
    OLD_VECTORS_REQUIRED: List[str] = None
    NEW_VECTORS_REQUIRED: List[str] = None
    
    def __post_init__(self):
        if self.TOP_PAGES_REQUIRED is None:
            self.TOP_PAGES_REQUIRED = [
                'URL', 'Current traffic', 'Current traffic value', 'Current # of keywords',
                'Current top keyword', 'Current top keyword: Country',
                'Current top keyword: Volume', 'Current top keyword: Position'
            ]
        if self.OLD_VECTORS_REQUIRED is None:
            self.OLD_VECTORS_REQUIRED = [
                'Address', 'Status Code', 'Title 1', 'Meta Description 1', 'H1-1', 'embeds 1'
            ]
        if self.NEW_VECTORS_REQUIRED is None:
            self.NEW_VECTORS_REQUIRED = ['Address', 'H1-1', 'embeds 1']

@dataclass
class MatchingConfig:
    """Configuration for matching algorithms."""
    DEFAULT_FUZZY_THRESHOLD: int = 90
    DEFAULT_H1_THRESHOLD: int = 90
    DEFAULT_VECTOR_THRESHOLD: float = 0.75
    MAX_NEW_URLS_FOR_AI: int = 50
    OPENAI_MODEL: str = "gpt-3.5-turbo"
    SENTENCE_TRANSFORMER_MODEL: str = "all-MiniLM-L6-v2"

@dataclass
class ScoringConfig:
    """Configuration for priority scoring."""
    # Weights for priority scoring with GSC data
    PRIORITY_WEIGHTS_WITH_GSC: Dict[str, float] = None
    # Weights for priority scoring without GSC data  
    PRIORITY_WEIGHTS_WITHOUT_GSC: Dict[str, float] = None
    
    def __post_init__(self):
        if self.PRIORITY_WEIGHTS_WITH_GSC is None:
            self.PRIORITY_WEIGHTS_WITH_GSC = {
                'traffic': 0.4,
                'traffic_value': 0.3,
                'keywords': 0.2,
                'impressions': 0.1
            }
        if self.PRIORITY_WEIGHTS_WITHOUT_GSC is None:
            self.PRIORITY_WEIGHTS_WITHOUT_GSC = {
                'traffic': 0.5,
                'traffic_value': 0.3,
                'keywords': 0.2
            }

@dataclass
class ConfidenceConfig:
    """Configuration for confidence scoring."""
    EXACT_MATCH_SCORE: int = 100
    FUZZY_MATCH_SCORE: int = 50
    EXACT_H1_SCORE: int = 80
    SIMILAR_H1_SCORE: int = 60
    AI_MATCH_SCORE: int = 70
    AI_UNVERIFIED_SCORE: int = 50
    
    # Vector similarity confidence thresholds
    VECTOR_HIGH_CONFIDENCE: float = 0.9
    VECTOR_HIGH_SCORE: int = 90
    VECTOR_MEDIUM_CONFIDENCE: float = 0.75
    VECTOR_MEDIUM_SCORE: int = 60
    VECTOR_LOW_SCORE: int = 30

class AppConfig:
    """Main application configuration."""
    
    def __init__(self):
        self.files = FileConfig()
        self.columns = ColumnConfig()
        self.matching = MatchingConfig()
        self.scoring = ScoringConfig()
        self.confidence = ConfidenceConfig()
        
        # API Keys
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        
    def validate(self) -> bool:
        """Validate configuration."""
        if not self.openai_api_key:
            return False
        return True