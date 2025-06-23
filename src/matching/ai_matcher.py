# src/matching/ai_matcher.py

import openai
from typing import List
from src.matching.base_matcher import BaseMatcher
from src.core.models import URLData, URLMatch, MatchType

class AIMatcher(BaseMatcher):
    """Matcher using OpenAI's GPT models for intelligent URL matching."""
    
    def __init__(self, config=None):
        super().__init__(config)
        self.model = config.matching.OPENAI_MODEL if config else "gpt-3.5-turbo"
        self.temperature = config.matching.OPENAI_TEMPERATURE if config else 0.0
        self.max_retries = config.matching.OPENAI_MAX_RETRIES if config else 3
        self.max_new_urls = config.matching.MAX_NEW_URLS_FOR_AI if config else 50
        
        # Set OpenAI API key
        if config and config.openai_api_key:
            openai.api_key = config.openai_api_key
    
    def get_match_type(self) -> MatchType:
        """Return the match type for AI matching."""
        return MatchType.AI_MATCH
    
    def find_matches(self, old_urls: List[URLData], new_urls: List[URLData]) -> List[URLMatch]:
        """
        Find AI-based matches between old and new URLs.
        
        Args:
            old_urls: List of old URLs to match
            new_urls: List of new URLs to match against
            
        Returns:
            List of URLMatch objects for AI matches
        """
        matches = []
        
        # Filter available new URLs and limit for performance
        available_new_urls = [
            url for url in new_urls 
            if self.is_new_url_available(url)
        ][:self.max_new_urls]
        
        if not available_new_urls:
            return matches
        
        # Process each old URL
        for old_url in old_urls:
            match = self._find_ai_match_for_url(old_url, available_new_urls)
            if match:
                matches.append(match)
        
        return matches
    
    def _find_ai_match_for_url(self, old_url: URLData, new_urls: List[URLData]) -> URLMatch:
        """Find the best AI match for a single old URL."""
        prompt = self._build_prompt(old_url, new_urls)
        
        for attempt in range(self.max_retries):
            try:
                response = openai.chat.completions.create(
                    model=self.model,
                    messages=[
                        {
                            "role": "system", 
                            "content": "You are an expert SEO consultant helping with website migrations. Your job is to match old URLs to new URLs based on semantic relevance and content similarity."
                        },
                        {
                            "role": "user", 
                            "content": prompt
                        }
                    ],
                    temperature=self.temperature,
                    max_tokens=100
                )
                
                suggested_url = response.choices[0].message.content.strip()
                return self._process_ai_response(old_url, suggested_url, new_urls)
                
            except Exception as e:
                if attempt == self.max_retries - 1:
                    # Last attempt failed, return no match
                    return self._create_error_match(old_url, str(e))
                continue
        
        return None
    
    def _build_prompt(self, old_url: URLData, new_urls: List[URLData]) -> str:
        """Build the prompt for OpenAI API."""
        prompt = f"""I need to find the best redirect for an old URL during a website migration.

OLD URL DETAILS:
- URL: {old_url.address}
- Title: {old_url.title or 'N/A'}
- Meta Description: {old_url.meta_description or 'N/A'}
- H1 Header: {old_url.h1 or 'N/A'}

AVAILABLE NEW URLs TO CHOOSE FROM:"""
        
        for i, new_url in enumerate(new_urls, 1):
            prompt += f"\n{i}. {new_url.address}"
            if new_url.h1:
                prompt += f" | H1: {new_url.h1}"
        
        prompt += """

Please select the BEST matching new URL from the list above based on:
1. Content relevance and semantic similarity
2. URL structure and path similarity  
3. Topic and purpose alignment

Respond with ONLY the complete URL of your choice, or "NONE" if no good match exists.
Do not provide explanations or additional text."""
        
        return prompt
    
    def _process_ai_response(self, old_url: URLData, suggested_url: str, new_urls: List[URLData]) -> URLMatch:
        """Process the AI response and create a match."""
        if suggested_url.upper() == "NONE":
            return self._create_no_match(old_url)
        
        # Find the suggested URL in the new URLs list
        matched_new_url = None
        for new_url in new_urls:
            if suggested_url.lower().strip() == new_url.address.lower().strip():
                if self.is_new_url_available(new_url):
                    matched_new_url = new_url
                    break
        
        if matched_new_url:
            # Verified match
            confidence_score = self.config.confidence.AI_MATCH_SCORE if self.config else 70
            match = self.create_match(
                old_url=old_url,
                new_url=matched_new_url,
                confidence_score=confidence_score
            )
            self.mark_new_url_as_used(matched_new_url)
            return match
        else:
            # AI suggested a URL that doesn't exist or is unavailable
            return self._create_unverified_match(old_url, suggested_url)
    
    def _create_no_match(self, old_url: URLData) -> URLMatch:
        """Create a no-match result for AI matching."""
        no_match_url = URLData(address="NOT FOUND (AI)")
        
        return URLMatch(
            old_url=old_url,
            new_url=no_match_url,
            match_type=MatchType.AI_MATCH,
            confidence_score=0
        )
    
    def _create_unverified_match(self, old_url: URLData, suggested_url: str) -> URLMatch:
        """Create an unverified match when AI suggests a URL not in our list."""
        unverified_url = URLData(address=f"{suggested_url} (AI Not Verified)")
        
        confidence_score = self.config.confidence.AI_UNVERIFIED_SCORE if self.config else 50
        
        return URLMatch(
            old_url=old_url,
            new_url=unverified_url,
            match_type=MatchType.AI_MATCH,
            confidence_score=confidence_score
        )
    
    def _create_error_match(self, old_url: URLData, error_message: str) -> URLMatch:
        """Create an error match when AI matching fails."""
        error_url = URLData(address="NOT FOUND (API Error)")
        
        return URLMatch(
            old_url=old_url,
            new_url=error_url,
            match_type=MatchType.AI_MATCH,
            confidence_score=0
        )