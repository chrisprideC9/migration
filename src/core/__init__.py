# src/core/__init__.py

"""Core components for URL matching pipeline."""

from src.core.models import (
    URLData,
    URLMatch, 
    MatchingResults,
    ProcessingStats,
    MatchType
)
from src.core.pipeline import URLMatchingPipeline

__all__ = [
    "URLData",
    "URLMatch",
    "MatchingResults", 
    "ProcessingStats",
    "MatchType",
    "URLMatchingPipeline"
]