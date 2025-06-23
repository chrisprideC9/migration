# src/data/processor.py

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from src.core.models import URLData
from src.utils.url_utils import extract_path
from src.utils.embedding_utils import string_to_embedding

class DataProcessor:
    """Processes raw data into URLData objects."""
    
    def __init__(self, config=None):
        self.config = config
        
    def process_raw_data(self, raw_data: Dict[str, pd.DataFrame]) -> Tuple[List[URLData], List[URLData], bool]:
        """
        Process raw DataFrames into URLData objects.
        
        Args:
            raw_data: Dictionary of raw DataFrames
            
        Returns:
            Tuple of (old_urls, new_urls, has_gsc_data)
        """
        old_urls = []
        new_urls = []
        has_gsc_data = False
        
        # Process top pages (old URLs with SEO data)
        if 'top_pages.csv' in raw_data:
            old_urls = self._process_top_pages(raw_data['top_pages.csv'])
            
        # Process old vectors (additional data for old URLs)
        if 'old_vectors.csv' in raw_data:
            old_urls = self._merge_old_vectors(old_urls, raw_data['old_vectors.csv'])
            
        # Process new vectors (new URLs)
        if 'new_vectors.csv' in raw_data:
            new_urls = self._process_new_vectors(raw_data['new_vectors.csv'])
            
        # Process impressions data if available
        if 'impressions.csv' in raw_data:
            old_urls = self._merge_impressions_data(old_urls, raw_data['impressions.csv'])
            has_gsc_data = True
            
        return old_urls, new_urls, has_gsc_data
    
    def _process_top_pages(self, df: pd.DataFrame) -> List[URLData]:
        """Process top pages CSV into URLData objects."""
        urls = []
        
        # Map column names (handle variations)
        url_col = self._find_column(df, ['URL', 'url', 'Address', 'address'])
        traffic_col = self._find_column(df, ['Current traffic', 'Traffic', 'traffic'])
        traffic_value_col = self._find_column(df, ['Current traffic value', 'Traffic value', 'traffic_value'])
        keywords_col = self._find_column(df, ['Current # of keywords', 'Keywords', 'keywords'])
        
        for _, row in df.iterrows():
            try:
                url_data = URLData(
                    address=str(row.get(url_col, '')).strip(),
                    traffic=float(row.get(traffic_col, 0) or 0),
                    traffic_value=float(row.get(traffic_value_col, 0) or 0),
                    keywords=int(row.get(keywords_col, 0) or 0),
                    path=extract_path(str(row.get(url_col, '')))
                )
                
                if url_data.address:  # Only add if URL exists
                    urls.append(url_data)
                    
            except Exception as e:
                continue  # Skip invalid rows
                
        return urls
    
    def _process_new_vectors(self, df: pd.DataFrame) -> List[URLData]:
        """Process new vectors CSV into URLData objects."""
        urls = []
        
        # Map column names
        url_col = self._find_column(df, ['Address', 'URL', 'url', 'address'])
        h1_col = self._find_column(df, ['H1-1', 'H1', 'h1'])
        embedding_col = self._find_column(df, ['embeds 1', 'embedding', 'embeddings'])
        
        for _, row in df.iterrows():
            try:
                # Process embedding
                embedding = None
                if embedding_col and pd.notna(row.get(embedding_col)):
                    embedding = string_to_embedding(str(row[embedding_col]))
                
                url_data = URLData(
                    address=str(row.get(url_col, '')).strip(),
                    h1=str(row.get(h1_col, '')).strip() if pd.notna(row.get(h1_col)) else None,
                    embedding=embedding,
                    path=extract_path(str(row.get(url_col, '')))
                )
                
                if url_data.address:  # Only add if URL exists
                    urls.append(url_data)
                    
            except Exception as e:
                continue  # Skip invalid rows
                
        return urls
    
    def _merge_old_vectors(self, old_urls: List[URLData], df: pd.DataFrame) -> List[URLData]:
        """Merge old vectors data with existing old URLs."""
        # Create lookup by URL
        url_lookup = {url.address: url for url in old_urls}
        
        # Map column names
        url_col = self._find_column(df, ['Address', 'URL', 'url', 'address'])
        title_col = self._find_column(df, ['Title 1', 'Title', 'title'])
        meta_col = self._find_column(df, ['Meta Description 1', 'Meta Description', 'meta_description'])
        h1_col = self._find_column(df, ['H1-1', 'H1', 'h1'])
        embedding_col = self._find_column(df, ['embeds 1', 'embedding', 'embeddings'])
        status_col = self._find_column(df, ['Status Code', 'status_code', 'status'])
        
        for _, row in df.iterrows():
            try:
                url = str(row.get(url_col, '')).strip()
                
                if url in url_lookup:
                    # Update existing URLData
                    url_data = url_lookup[url]
                    
                    if title_col and pd.notna(row.get(title_col)):
                        url_data.title = str(row[title_col]).strip()
                    
                    if meta_col and pd.notna(row.get(meta_col)):
                        url_data.meta_description = str(row[meta_col]).strip()
                    
                    if h1_col and pd.notna(row.get(h1_col)):
                        url_data.h1 = str(row[h1_col]).strip()
                    
                    if embedding_col and pd.notna(row.get(embedding_col)):
                        url_data.embedding = string_to_embedding(str(row[embedding_col]))
                    
                    if status_col and pd.notna(row.get(status_col)):
                        url_data.status_code = int(row[status_col])
                        
                else:
                    # Create new URLData if URL not found in top_pages
                    embedding = None
                    if embedding_col and pd.notna(row.get(embedding_col)):
                        embedding = string_to_embedding(str(row[embedding_col]))
                    
                    url_data = URLData(
                        address=url,
                        title=str(row.get(title_col, '')).strip() if pd.notna(row.get(title_col)) else None,
                        meta_description=str(row.get(meta_col, '')).strip() if pd.notna(row.get(meta_col)) else None,
                        h1=str(row.get(h1_col, '')).strip() if pd.notna(row.get(h1_col)) else None,
                        embedding=embedding,
                        status_code=int(row[status_col]) if pd.notna(row.get(status_col)) else None,
                        path=extract_path(url)
                    )
                    
                    if url_data.address:
                        old_urls.append(url_data)
                        url_lookup[url] = url_data
                        
            except Exception as e:
                continue  # Skip invalid rows
                
        return old_urls
    
    def _merge_impressions_data(self, old_urls: List[URLData], df: pd.DataFrame) -> List[URLData]:
        """Merge Google Search Console impressions data."""
        # Create lookup by URL
        url_lookup = {url.address: url for url in old_urls}
        
        # Map column names
        url_col = self._find_column(df, ['URL', 'url', 'Address', 'address', 'Page'])
        clicks_col = self._find_column(df, ['Clicks', 'clicks'])
        impressions_col = self._find_column(df, ['Impressions', 'impressions'])
        
        for _, row in df.iterrows():
            try:
                url = str(row.get(url_col, '')).strip()
                
                if url in url_lookup:
                    url_data = url_lookup[url]
                    
                    if clicks_col and pd.notna(row.get(clicks_col)):
                        url_data.clicks = int(row[clicks_col])
                    
                    if impressions_col and pd.notna(row.get(impressions_col)):
                        url_data.impressions = int(row[impressions_col])
                        
            except Exception as e:
                continue  # Skip invalid rows
                
        return old_urls
    
    def _find_column(self, df: pd.DataFrame, possible_names: List[str]) -> str:
        """Find a column by checking multiple possible names."""
        for name in possible_names:
            if name in df.columns:
                return name
        return None