# main.py - Clean Streamlit Entry Point

import streamlit as st
from src.ui.pages import URLMatchingApp
from src.core.pipeline import URLMatchingPipeline
from config.settings import AppConfig

# Work around Streamlit's module watcher error when inspecting torch.classes
try:
    import torch
    from types import SimpleNamespace

    if hasattr(torch, "classes") and not hasattr(torch.classes, "__path__"):
        torch.classes.__path__ = SimpleNamespace(_path=[])
except Exception:
    # torch may not be installed or not needed; ignore any issues
    pass

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