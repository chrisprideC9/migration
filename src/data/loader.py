# src/data/loader.py

import pandas as pd
import zipfile
import io
from typing import Dict, Any, Optional
import streamlit as st

class DataLoader:
    """Handles loading data from various sources."""
    
    def __init__(self, config=None):
        self.config = config
        
    def load_from_zip(self, zip_file: zipfile.ZipFile) -> Dict[str, pd.DataFrame]:
        """
        Load CSV files from a ZIP archive.
        
        Args:
            zip_file: ZipFile object
            
        Returns:
            Dictionary mapping filename to DataFrame
        """
        data = {}
        
        try:
            # Get list of files in the ZIP
            file_list = zip_file.namelist()
            
            # Expected files
            expected_files = [
                'top_pages.csv',
                'old_vectors.csv', 
                'new_vectors.csv',
                'impressions.csv'  # Optional
            ]
            
            for filename in file_list:
                if filename.endswith('.csv'):
                    try:
                        # Read the CSV file
                        with zip_file.open(filename) as csv_file:
                            df = pd.read_csv(csv_file)
                            data[filename] = df
                            
                    except Exception as e:
                        st.warning(f"Could not read {filename}: {str(e)}")
                        
            return data
            
        except Exception as e:
            raise Exception(f"Error loading data from ZIP: {str(e)}")
    
    def validate_required_files(self, data: Dict[str, pd.DataFrame]) -> tuple[bool, str]:
        """
        Validate that required files are present.
        
        Args:
            data: Dictionary of loaded DataFrames
            
        Returns:
            Tuple of (is_valid, message)
        """
        required_files = ['top_pages.csv', 'old_vectors.csv', 'new_vectors.csv']
        missing_files = []
        
        for required_file in required_files:
            if required_file not in data:
                missing_files.append(required_file)
        
        if missing_files:
            return False, f"Missing required files: {', '.join(missing_files)}"
        
        return True, "All required files present"