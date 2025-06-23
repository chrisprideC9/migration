# main.py - Clean Streamlit Entry Point

import streamlit as st
from src.ui.pages import URLMatchingApp
from src.core.pipeline import URLMatchingPipeline
from config.settings import AppConfig

def main():
    """Main Streamlit application entry point."""
    st.set_page_config(
        page_title="URL Matching Tool",
        page_icon="📄",
        layout="wide"
    )
    
    # Initialize the pipeline
    pipeline = URLMatchingPipeline()
    
    # Initialize and run the UI
    app = URLMatchingApp(pipeline)
    app.run()

if __name__ == "__main__":
    main()