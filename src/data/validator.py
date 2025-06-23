# src/data/validator.py

import pandas as pd
from typing import Dict, Tuple, List

class DataValidator:
    """Validates loaded data for completeness and correctness."""
    
    def __init__(self, config=None):
        self.config = config
        
    def validate_raw_data(self, raw_data: Dict[str, pd.DataFrame]) -> Tuple[bool, str]:
        """
        Validate raw data dictionary.
        
        Args:
            raw_data: Dictionary of loaded DataFrames
            
        Returns:
            Tuple of (is_valid, message)
        """
        # Check required files
        required_files = ['top_pages.csv', 'old_vectors.csv', 'new_vectors.csv']
        missing_files = []
        
        for required_file in required_files:
            if required_file not in raw_data:
                missing_files.append(required_file)
        
        if missing_files:
            return False, f"Missing required files: {', '.join(missing_files)}"
        
        # Validate each file
        validation_errors = []
        
        # Validate top_pages.csv
        error = self._validate_top_pages(raw_data.get('top_pages.csv'))
        if error:
            validation_errors.append(f"top_pages.csv: {error}")
            
        # Validate old_vectors.csv
        error = self._validate_old_vectors(raw_data.get('old_vectors.csv'))
        if error:
            validation_errors.append(f"old_vectors.csv: {error}")
            
        # Validate new_vectors.csv
        error = self._validate_new_vectors(raw_data.get('new_vectors.csv'))
        if error:
            validation_errors.append(f"new_vectors.csv: {error}")
            
        # Validate impressions.csv if present
        if 'impressions.csv' in raw_data:
            error = self._validate_impressions(raw_data.get('impressions.csv'))
            if error:
                validation_errors.append(f"impressions.csv: {error}")
        
        if validation_errors:
            return False, "; ".join(validation_errors)
        
        return True, "All data files are valid"
    
    def _validate_top_pages(self, df: pd.DataFrame) -> str:
        """Validate top_pages.csv structure."""
        if df is None or df.empty:
            return "File is empty"
        
        # Check for URL column
        url_columns = ['URL', 'url', 'Address', 'address']
        if not any(col in df.columns for col in url_columns):
            return f"Missing URL column. Expected one of: {', '.join(url_columns)}"
        
        # Check for at least some SEO data columns
        seo_columns = ['Current traffic', 'Traffic', 'traffic', 'Current traffic value', 'Traffic value']
        if not any(col in df.columns for col in seo_columns):
            return "Missing SEO data columns (traffic, traffic value, etc.)"
        
        return None
    
    def _validate_old_vectors(self, df: pd.DataFrame) -> str:
        """Validate old_vectors.csv structure."""
        if df is None or df.empty:
            return "File is empty"
        
        # Check for required columns
        required_columns = [
            (['Address', 'URL', 'url', 'address'], "URL column"),
            (['H1-1', 'H1', 'h1'], "H1 column"),
            (['embeds 1', 'embedding', 'embeddings'], "embedding column")
        ]
        
        for column_options, column_name in required_columns:
            if not any(col in df.columns for col in column_options):
                return f"Missing {column_name}. Expected one of: {', '.join(column_options)}"
        
        return None
    
    def _validate_new_vectors(self, df: pd.DataFrame) -> str:
        """Validate new_vectors.csv structure."""
        if df is None or df.empty:
            return "File is empty"
        
        # Check for required columns
        required_columns = [
            (['Address', 'URL', 'url', 'address'], "URL column"),
            (['H1-1', 'H1', 'h1'], "H1 column"),
            (['embeds 1', 'embedding', 'embeddings'], "embedding column")
        ]
        
        for column_options, column_name in required_columns:
            if not any(col in df.columns for col in column_options):
                return f"Missing {column_name}. Expected one of: {', '.join(column_options)}"
        
        return None
    
    def _validate_impressions(self, df: pd.DataFrame) -> str:
        """Validate impressions.csv structure."""
        if df is None or df.empty:
            return "File is empty"
        
        # Check for required columns
        required_columns = [
            (['URL', 'url', 'Address', 'address', 'Page'], "URL column"),
            (['Impressions', 'impressions'], "impressions column")
        ]
        
        for column_options, column_name in required_columns:
            if not any(col in df.columns for col in column_options):
                return f"Missing {column_name}. Expected one of: {', '.join(column_options)}"
        
        return None
    
    def get_data_summary(self, raw_data: Dict[str, pd.DataFrame]) -> Dict[str, Dict]:
        """Get summary information about the loaded data."""
        summary = {}
        
        for filename, df in raw_data.items():
            if df is not None:
                summary[filename] = {
                    'rows': len(df),
                    'columns': len(df.columns),
                    'column_names': list(df.columns),
                    'memory_usage': df.memory_usage(deep=True).sum(),
                    'has_null_values': df.isnull().any().any(),
                    'duplicate_rows': df.duplicated().sum()
                }
            else:
                summary[filename] = {
                    'error': 'Failed to load file'
                }
        
        return summary