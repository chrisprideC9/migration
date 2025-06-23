# src/core/pipeline.py

import time
from typing import Dict, List, Optional, Tuple
import zipfile
import streamlit as st

from src.core.models import URLData, URLMatch, MatchingResults, ProcessingStats, MatchType
from src.data.loader import DataLoader
from src.data.processor import DataProcessor
from src.data.validator import DataValidator
from src.matching.matcher_factory import MatcherFactory
from src.scoring.priority_scorer import PriorityScorer
from config.settings import AppConfig

class URLMatchingPipeline:
    """Main pipeline for URL matching process."""
    
    def __init__(self, config: Optional[AppConfig] = None):
        self.config = config or AppConfig()
        self.loader = DataLoader(self.config)
        self.processor = DataProcessor(self.config)
        self.validator = DataValidator(self.config)
        self.matcher_factory = MatcherFactory(self.config)
        self.priority_scorer = PriorityScorer()
        
        self.stats = ProcessingStats()
        self._old_urls: List[URLData] = []
        self._new_urls: List[URLData] = []
        
    def process_zip_file(self, uploaded_zip: zipfile.ZipFile) -> Tuple[bool, str]:
        """
        Process uploaded ZIP file and load data.
        
        Returns:
            Tuple of (success, message)
        """
        start_time = time.time()
        
        try:
            # Load raw data from ZIP
            raw_data = self.loader.load_from_zip(uploaded_zip)
            
            # Validate the data
            is_valid, validation_message = self.validator.validate_raw_data(raw_data)
            if not is_valid:
                return False, validation_message
            
            # Process the data
            self._old_urls, self._new_urls, has_gsc_data = self.processor.process_raw_data(raw_data)
            
            # Update stats
            self.stats.total_old_urls = len(self._old_urls)
            self.stats.total_new_urls = len(self._new_urls)
            self.stats.processing_time = time.time() - start_time
            
            return True, f"Successfully processed {len(self._old_urls)} old URLs and {len(self._new_urls)} new URLs"
            
        except Exception as e:
            return False, f"Error processing ZIP file: {str(e)}"
    
    def run_matching(self, 
                    fuzzy_threshold: int = 90,
                    h1_threshold: int = 90,
                    use_ai_matching: bool = False) -> MatchingResults:
        """
        Run the complete matching pipeline.
        
        Args:
            fuzzy_threshold: Threshold for fuzzy matching (0-100)
            h1_threshold: Threshold for H1 matching (0-100)
            use_ai_matching: Whether to use AI matching for unmatched URLs
            
        Returns:
            MatchingResults object containing all matches and unmatched URLs
        """
        if not self._old_urls or not self._new_urls:
            raise ValueError("No data loaded. Please process a ZIP file first.")
        
        start_time = time.time()
        all_matches: List[URLMatch] = []
        unmatched_urls = self._old_urls.copy()
        
        # Step 1: Exact Path Matching
        exact_matcher = self.matcher_factory.create_exact_matcher()
        exact_matches = exact_matcher.find_matches(unmatched_urls, self._new_urls)
        all_matches.extend(exact_matches)
        self.stats.exact_matches = len(exact_matches)
        
        # Remove matched URLs from unmatched list
        matched_old_addresses = {match.old_url.address for match in exact_matches}
        unmatched_urls = [url for url in unmatched_urls if url.address not in matched_old_addresses]
        
        # Step 2: Fuzzy Path Matching  
        if unmatched_urls:
            fuzzy_matcher = self.matcher_factory.create_fuzzy_matcher(threshold=fuzzy_threshold)
            fuzzy_matches = fuzzy_matcher.find_matches(unmatched_urls, self._new_urls)
            all_matches.extend(fuzzy_matches)
            self.stats.fuzzy_matches = len(fuzzy_matches)
            
            matched_old_addresses = {match.old_url.address for match in fuzzy_matches}
            unmatched_urls = [url for url in unmatched_urls if url.address not in matched_old_addresses]
        
        # Step 3: Vector Similarity Matching
        if unmatched_urls:
            vector_matcher = self.matcher_factory.create_vector_matcher()
            vector_matches = vector_matcher.find_matches(unmatched_urls, self._new_urls)
            all_matches.extend(vector_matches)
            self.stats.vector_matches = len(vector_matches)
            
            matched_old_addresses = {match.old_url.address for match in vector_matches}
            unmatched_urls = [url for url in unmatched_urls if url.address not in matched_old_addresses]
        
        # Step 4: H1 Header Matching
        if unmatched_urls:
            h1_matcher = self.matcher_factory.create_h1_matcher(threshold=h1_threshold)
            h1_matches = h1_matcher.find_matches(unmatched_urls, self._new_urls)
            all_matches.extend(h1_matches)
            self.stats.h1_matches = len(h1_matches)
            
            matched_old_addresses = {match.old_url.address for match in h1_matches}
            unmatched_urls = [url for url in unmatched_urls if url.address not in matched_old_addresses]
        
        # Step 5: AI Matching (optional)
        if unmatched_urls and use_ai_matching:
            try:
                ai_matcher = self.matcher_factory.create_ai_matcher()
                ai_matches = ai_matcher.find_matches(unmatched_urls, self._new_urls)
                all_matches.extend(ai_matches)
                self.stats.ai_matches = len(ai_matches)
                
                matched_old_addresses = {match.old_url.address for match in ai_matches}
                unmatched_urls = [url for url in unmatched_urls if url.address not in matched_old_addresses]
            except Exception as e:
                st.warning(f"AI matching failed: {str(e)}")
        
        # Calculate priority scores for all matches
        has_gsc_data = any(url.impressions > 0 for url in self._old_urls)
        self.priority_scorer.has_gsc_data = has_gsc_data
        
        for match in all_matches:
            match.priority_score = self.priority_scorer.calculate_score(match.old_url)
        
        # Update final stats
        self.stats.unmatched = len(unmatched_urls)
        self.stats.processing_time += time.time() - start_time
        
        return MatchingResults(
            matches=all_matches,
            unmatched_urls=unmatched_urls,
            has_gsc_data=has_gsc_data
        )
    
    def get_processing_stats(self) -> ProcessingStats:
        """Get current processing statistics."""
        return self.stats
    
    def reset(self):
        """Reset the pipeline state."""
        self.stats = ProcessingStats()
        self._old_urls = []
        self._new_urls = []