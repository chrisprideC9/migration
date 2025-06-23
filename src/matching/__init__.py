# src/matching/__init__.py

"""URL matching strategies and algorithms."""

from src.matching.base_matcher import BaseMatcher
from src.matching.exact_matcher import ExactMatcher
from src.matching.fuzzy_matcher import FuzzyMatcher
from src.matching.vector_matcher import VectorMatcher
from src.matching.h1_matcher import H1Matcher
from src.matching.ai_matcher import AIMatcher
from src.matching.matcher_factory import MatcherFactory

__all__ = [
    "BaseMatcher",
    "ExactMatcher",
    "FuzzyMatcher", 
    "VectorMatcher",
    "H1Matcher",
    "AIMatcher",
    "MatcherFactory"
]