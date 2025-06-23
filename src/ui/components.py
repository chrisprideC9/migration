# src/ui/components.py

import streamlit as st
import pandas as pd
from typing import Dict, Any
from src.core.models import ProcessingStats, MatchingResults

class UIComponents:
    """Reusable UI components for the Streamlit app."""
    
    @staticmethod
    def display_processing_stats(stats: ProcessingStats):
        """Display processing statistics in a nice format."""
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Old URLs", stats.total_old_urls)
        with col2:
            st.metric("Total New URLs", stats.total_new_urls)  
        with col3:
            st.metric("Processing Time", f"{stats.processing_time:.2f}s")
    
    @staticmethod
    def display_matching_summary(stats: ProcessingStats):
        """Display a summary of matching results."""
        summary = stats.get_summary()
        
        st.subheader("📊 Matching Summary")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Total Matches", summary['Total Matches'])
            st.metric("Match Rate", summary['Match Rate'])
            st.metric("Unmatched URLs", summary['Unmatched'])
        
        with col2:
            st.write("**Matches by Type:**")
            st.write(f"- Exact Matches: {summary['Exact Matches']}")
            st.write(f"- Fuzzy Matches: {summary['Fuzzy Matches']}")
            st.write(f"- Vector Matches: {summary['Vector Matches']}")
            st.write(f"- H1 Matches: {summary['H1 Matches']}")
            st.write(f"- AI Matches: {summary['AI Matches']}")
    
    @staticmethod
    def display_results_overview(results: MatchingResults):
        """Display an overview of the matching results."""
        matches_df = results.get_matches_df()
        unmatched_df = results.get_unmatched_df()
        
        total_old_urls = len(matches_df) + len(unmatched_df)
        match_rate = (len(matches_df) / total_old_urls * 100) if total_old_urls > 0 else 0
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total URLs", total_old_urls)
        with col2:
            st.metric("Matched", len(matches_df))
        with col3:
            st.metric("Unmatched", len(unmatched_df))
        with col4:
            st.metric("Match Rate", f"{match_rate:.1f}%")
    
    @staticmethod
    def display_matches_table(df: pd.DataFrame):
        """Display a formatted table of matches."""
        if df.empty:
            st.info("No data to display")
            return
        
        # Format the DataFrame for display
        display_df = df.copy()
        
        # Format numeric columns
        numeric_columns = ['Confidence_Score', 'Priority_Score', 'Traffic', 'Traffic value', 'Keywords']
        for col in numeric_columns:
            if col in display_df.columns:
                if col == 'Confidence_Score':
                    display_df[col] = display_df[col].apply(lambda x: f"{x:.0f}%" if pd.notnull(x) else "N/A")
                elif col == 'Priority_Score':
                    display_df[col] = display_df[col].apply(lambda x: f"{x:.2f}" if pd.notnull(x) else "N/A")
                else:
                    display_df[col] = display_df[col].apply(lambda x: f"{x:,.0f}" if pd.notnull(x) else "N/A")
        
        # Display the table
        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True
        )
        
        # Show additional statistics
        st.write(f"**Total rows:** {len(df)}")
        
        if 'Priority_Score' in df.columns:
            avg_priority = df['Priority_Score'].mean()
            st.write(f"**Average Priority Score:** {avg_priority:.2f}")
        
        if 'Confidence_Score' in df.columns:
            avg_confidence = df['Confidence_Score'].mean()
            st.write(f"**Average Confidence Score:** {avg_confidence:.1f}%")
    
    @staticmethod
    def convert_df_to_csv(df: pd.DataFrame) -> bytes:
        """Convert DataFrame to CSV bytes for download."""
        return df.to_csv(index=False).encode('utf-8')
    
    @staticmethod
    def display_error_message(title: str, message: str, details: str = None):
        """Display a formatted error message."""
        st.error(f"**{title}**")
        st.write(message)
        if details:
            with st.expander("Error Details"):
                st.code(details)
    
    @staticmethod
    def display_success_message(title: str, message: str):
        """Display a formatted success message."""
        st.success(f"**{title}**")
        st.write(message)
    
    @staticmethod
    def display_info_box(title: str, content: str, icon: str = "ℹ️"):
        """Display an informational box."""
        st.info(f"{icon} **{title}**")
        st.write(content)
    
    @staticmethod
    def create_progress_bar(current: int, total: int, label: str = "Progress"):
        """Create a progress bar."""
        progress = current / total if total > 0 else 0
        st.progress(progress, text=f"{label}: {current}/{total} ({progress*100:.1f}%)")
    
    @staticmethod
    def display_configuration_summary(config: Dict[str, Any]):
        """Display a summary of the current configuration."""
        st.subheader("⚙️ Configuration")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Matching Settings:**")
            for key, value in config.items():
                if 'threshold' in key.lower():
                    st.write(f"- {key}: {value}%")
                else:
                    st.write(f"- {key}: {value}")
        
        with col2:
            st.write("**File Settings:**")
            st.write("- Required files: top_pages.csv, old_vectors.csv, new_vectors.csv")
            st.write("- Optional files: impressions.csv")