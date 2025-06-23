# src/ui/pages.py

import streamlit as st
import zipfile
from typing import Optional
from src.core.pipeline import URLMatchingPipeline
from src.core.models import MatchingResults
from src.ui.components import UIComponents

class URLMatchingApp:
    """Main Streamlit application for URL matching."""
    
    def __init__(self, pipeline: URLMatchingPipeline):
        self.pipeline = pipeline
        self.ui = UIComponents()
        
        # Initialize session state
        if 'data_loaded' not in st.session_state:
            st.session_state.data_loaded = False
        if 'matching_results' not in st.session_state:
            st.session_state.matching_results = None
    
    def run(self):
        """Run the main application."""
        self._render_header()
        
        if not st.session_state.data_loaded:
            self._render_upload_section()
        else:
            self._render_matching_section()
            if st.session_state.matching_results:
                self._render_results_section()
    
    def _render_header(self):
        """Render the application header."""
        st.title("📄 URL Matching Tool for Website Migration")
        
        st.markdown("""
        This tool assists in matching URLs for website migration by comparing old and new URLs using:
        - Path-based matching
        - Fuzzy matching  
        - Vector similarity
        - H1 header matching
        - **AI-based GPT matching** (final fallback)
        
        It also cross-references data from Ahrefs and optionally Google Search Console to prioritize important URLs for redirects.
        """)
    
    def _render_upload_section(self):
        """Render the file upload section."""
        st.header("1. Upload Your Data")
        
        uploaded_zip = st.file_uploader(
            "Upload a ZIP file containing top_pages.csv, old_vectors.csv, new_vectors.csv (impressions.csv is optional)",
            type=["zip"]
        )
        
        if uploaded_zip:
            self._process_uploaded_file(uploaded_zip)
    
    def _process_uploaded_file(self, uploaded_zip):
        """Process the uploaded ZIP file."""
        with st.spinner("🔄 Processing uploaded file..."):
            try:
                with zipfile.ZipFile(uploaded_zip) as z:
                    success, message = self.pipeline.process_zip_file(z)
                    
                    if success:
                        st.success(f"✅ {message}")
                        st.session_state.data_loaded = True
                        
                        # Show processing stats
                        stats = self.pipeline.get_processing_stats()
                        self.ui.display_processing_stats(stats)
                        
                        st.rerun()
                    else:
                        st.error(f"❌ {message}")
                        
            except zipfile.BadZipFile:
                st.error("❌ The uploaded file is not a valid ZIP archive.")
            except Exception as e:
                st.error(f"❌ An error occurred: {str(e)}")
    
    def _render_matching_section(self):
        """Render the URL matching configuration section."""
        st.header("2. Configure URL Matching")
        
        col1, col2 = st.columns(2)
        
        with col1:
            fuzzy_threshold = st.slider(
                "Fuzzy Matching Threshold (%)",
                min_value=70,
                max_value=100, 
                value=90,
                help="Minimum similarity score for fuzzy path matching"
            )
        
        with col2:
            h1_threshold = st.slider(
                "H1 Matching Threshold (%)",
                min_value=70,
                max_value=100,
                value=90,
                help="Minimum similarity score for H1 header matching"
            )
        
        use_ai_matching = st.checkbox(
            "🤖 Use AI-based matching for unmatched URLs",
            help="Uses OpenAI's GPT model to suggest matches for URLs that couldn't be matched by other methods"
        )
        
        if st.button("🚀 Start URL Matching", type="primary"):
            self._run_matching(fuzzy_threshold, h1_threshold, use_ai_matching)
    
    def _run_matching(self, fuzzy_threshold: int, h1_threshold: int, use_ai_matching: bool):
        """Run the URL matching process."""
        with st.spinner("🔄 Running URL matching..."):
            try:
                results = self.pipeline.run_matching(
                    fuzzy_threshold=fuzzy_threshold,
                    h1_threshold=h1_threshold,
                    use_ai_matching=use_ai_matching
                )
                
                st.session_state.matching_results = results
                
                # Show final stats
                stats = self.pipeline.get_processing_stats()
                st.success("✅ URL matching completed!")
                self.ui.display_matching_summary(stats)
                
                st.rerun()
                
            except Exception as e:
                st.error(f"❌ Error during matching: {str(e)}")
    
    def _render_results_section(self):
        """Render the results section."""
        st.header("3. Matching Results")
        
        results: MatchingResults = st.session_state.matching_results
        
        # Display results overview
        self.ui.display_results_overview(results)
        
        # Display matched URLs
        self._render_matched_urls(results)
        
        # Display unmatched URLs
        self._render_unmatched_urls(results)
        
        # Download buttons
        self._render_download_section(results)
    
    def _render_matched_urls(self, results: MatchingResults):
        """Render the matched URLs section."""
        matches_df = results.get_matches_df()
        
        if not matches_df.empty:
            with st.expander("🔍 View Matched URLs", expanded=True):
                show_most_confident = st.checkbox(
                    "Show only most confident match per URL",
                    key="show_confident"
                )
                
                if show_most_confident:
                    display_df = results.get_most_confident_matches()
                    st.write("### 📈 Most confident match for each old URL")
                else:
                    display_df = matches_df
                    st.write("### 📊 All matches for each old URL")
                
                self.ui.display_matches_table(display_df)
        else:
            st.info("ℹ️ No matched URLs found.")
    
    def _render_unmatched_urls(self, results: MatchingResults):
        """Render the unmatched URLs section."""
        unmatched_df = results.get_unmatched_df()
        
        if not unmatched_df.empty:
            with st.expander("🔍 View Unmatched URLs"):
                st.write("### 🛑 URLs that could not be matched")
                self.ui.display_matches_table(unmatched_df)
        else:
            st.success("🎉 All URLs were successfully matched!")
    
    def _render_download_section(self, results: MatchingResults):
        """Render the download section."""
        st.header("4. Download Results")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            most_confident_df = results.get_most_confident_matches()
            if not most_confident_df.empty:
                csv_data = self.ui.convert_df_to_csv(most_confident_df)
                st.download_button(
                    label="📥 Download Most Confident Matches",
                    data=csv_data,
                    file_name="most_confident_matches.csv",
                    mime="text/csv"
                )
            else:
                st.info("No confident matches available")
        
        with col2:
            all_matches_df = results.get_matches_df()
            if not all_matches_df.empty:
                csv_data = self.ui.convert_df_to_csv(all_matches_df)
                st.download_button(
                    label="📥 Download All Matches",
                    data=csv_data,
                    file_name="all_matches.csv",
                    mime="text/csv"
                )
            else:
                st.info("No matches available")
        
        with col3:
            unmatched_df = results.get_unmatched_df()
            if not unmatched_df.empty:
                csv_data = self.ui.convert_df_to_csv(unmatched_df)
                st.download_button(
                    label="📥 Download Unmatched URLs",
                    data=csv_data,
                    file_name="unmatched_urls.csv",
                    mime="text/csv"
                )
            else:
                st.info("No unmatched URLs")
        
        # Reset button
        if st.button("🔄 Process New File", type="secondary"):
            self._reset_application()
    
    def _reset_application(self):
        """Reset the application state."""
        st.session_state.data_loaded = False
        st.session_state.matching_results = None
        self.pipeline.reset()
        st.rerun()