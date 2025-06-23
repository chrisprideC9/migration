# src/matching/matcher_factory.py

from typing import Dict, Type, Optional
from src.matching.base_matcher import BaseMatcher
from src.matching.exact_matcher import ExactMatcher
from src.matching.fuzzy_matcher import FuzzyMatcher
from src.matching.vector_matcher import VectorMatcher
from src.matching.h1_matcher import H1Matcher
from src.matching.ai_matcher import AIMatcher

class MatcherFactory:
    """Factory class for creating URL matcher instances."""
    
    def __init__(self, config=None):
        self.config = config
        self._matchers: Dict[str, Type[BaseMatcher]] = {
            'exact': ExactMatcher,
            'fuzzy': FuzzyMatcher,
            'vector': VectorMatcher,
            'h1': H1Matcher,
            'ai': AIMatcher
        }
    
    def create_exact_matcher(self) -> ExactMatcher:
        """Create an exact path matcher."""
        return ExactMatcher(self.config)
    
    def create_fuzzy_matcher(self, threshold: Optional[int] = None) -> FuzzyMatcher:
        """
        Create a fuzzy path matcher.
        
        Args:
            threshold: Similarity threshold (0-100), uses config default if None
        """
        if threshold is None:
            threshold = self.config.matching.DEFAULT_FUZZY_THRESHOLD if self.config else 90
        
        return FuzzyMatcher(self.config, threshold=threshold)
    
    def create_vector_matcher(self) -> VectorMatcher:
        """Create a vector similarity matcher."""
        return VectorMatcher(self.config)
    
    def create_h1_matcher(self, threshold: Optional[int] = None) -> H1Matcher:
        """
        Create an H1 header matcher.
        
        Args:
            threshold: Similarity threshold (0-100), uses config default if None
        """
        if threshold is None:
            threshold = self.config.matching.DEFAULT_H1_THRESHOLD if self.config else 90
        
        return H1Matcher(self.config, threshold=threshold)
    
    def create_ai_matcher(self) -> AIMatcher:
        """Create an AI-based matcher."""
        return AIMatcher(self.config)
    
    def create_matcher(self, matcher_type: str, **kwargs) -> BaseMatcher:
        """
        Create a matcher of the specified type.
        
        Args:
            matcher_type: Type of matcher ('exact', 'fuzzy', 'vector', 'h1', 'ai')
            **kwargs: Additional arguments for the matcher
            
        Returns:
            Matcher instance
            
        Raises:
            ValueError: If matcher type is not supported
        """
        if matcher_type not in self._matchers:
            raise ValueError(f"Unsupported matcher type: {matcher_type}. "
                           f"Supported types: {list(self._matchers.keys())}")
        
        matcher_class = self._matchers[matcher_type]
        
        # Handle special cases with additional parameters
        if matcher_type == 'fuzzy':
            threshold = kwargs.get('threshold')
            return self.create_fuzzy_matcher(threshold)
        elif matcher_type == 'h1':
            threshold = kwargs.get('threshold')
            return self.create_h1_matcher(threshold)
        else:
            return matcher_class(self.config)
    
    def get_available_matchers(self) -> list[str]:
        """Get list of available matcher types."""
        return list(self._matchers.keys())
    
    def register_matcher(self, name: str, matcher_class: Type[BaseMatcher]):
        """
        Register a custom matcher type.
        
        Args:
            name: Name for the matcher type
            matcher_class: Matcher class that extends BaseMatcher
        """
        if not issubclass(matcher_class, BaseMatcher):
            raise ValueError("Matcher class must extend BaseMatcher")
        
        self._matchers[name] = matcher_class
    
    def create_all_matchers(self, fuzzy_threshold: int = None, h1_threshold: int = None) -> Dict[str, BaseMatcher]:
        """
        Create instances of all available matchers.
        
        Args:
            fuzzy_threshold: Threshold for fuzzy matcher
            h1_threshold: Threshold for H1 matcher
            
        Returns:
            Dictionary mapping matcher type to instance
        """
        matchers = {}
        
        matchers['exact'] = self.create_exact_matcher()
        matchers['fuzzy'] = self.create_fuzzy_matcher(fuzzy_threshold)
        matchers['vector'] = self.create_vector_matcher()
        matchers['h1'] = self.create_h1_matcher(h1_threshold)
        
        # Only create AI matcher if API key is available
        if self.config and self.config.openai_api_key:
            matchers['ai'] = self.create_ai_matcher()
        
        return matchers