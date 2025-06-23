# src/ui/display.py

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, List, Any
from src.core.models import MatchingResults, ProcessingStats

class DisplayHelpers:
    """Helper functions for displaying data and visualizations."""
    
    @staticmethod
    def display_data_summary(summary: Dict[str, Dict[str, Any]]):
        """Display a summary of loaded data files."""
        st.subheader("📊 Data Summary")
        
        for filename, info in summary.items():
            with st.expander(f"📄 {filename}"):
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Rows", f"{info['rows']:,}")
                with col2:
                    st.metric("Columns", info['columns'])
                with col3:
                    st.metric("Memory", f"{info['memory_usage'] / 1024:.1f} KB")
                
                if info['column_names']:
                    st.write("**Columns:**")
                    st.write(", ".join(info['column_names']))
                
                if info['has_null_values']:
                    st.warning("⚠️ Contains null values")
                if info['duplicate_rows'] > 0:
                    st.warning(f"⚠️ Contains {info['duplicate_rows']} duplicate rows")
    
    @staticmethod
    def display_matching_progress(current_step: str, progress: float = None):
        """Display matching progress with current step."""
        st.write(f"**Current Step:** {current_step}")
        if progress is not None:
            st.progress(progress)
    
    @staticmethod
    def create_match_type_chart(results: MatchingResults):
        """Create a pie chart showing distribution of match types."""
        matches_df = results.get_matches_df()
        
        if matches_df.empty:
            return None
        
        # Count matches by type
        match_counts = matches_df['Match_Type'].value_counts()
        
        # Create pie chart
        fig = px.pie(
            values=match_counts.values,
            names=match_counts.index,
            title="Distribution of Match Types"
        )
        
        fig.update_traces(textposition='inside', textinfo='percent+label')
        
        return fig
    
    @staticmethod
    def create_confidence_distribution_chart(results: MatchingResults):
        """Create a histogram showing confidence score distribution."""
        matches_df = results.get_matches_df()
        
        if matches_df.empty:
            return None
        
        # Create histogram
        fig = px.histogram(
            matches_df,
            x='Confidence_Score',
            bins=10,
            title="Distribution of Confidence Scores",
            labels={'Confidence_Score': 'Confidence Score (%)', 'count': 'Number of Matches'}
        )
        
        return fig
    
    @staticmethod
    def create_priority_vs_confidence_scatter(results: MatchingResults):
        """Create a scatter plot of priority vs confidence scores."""
        matches_df = results.get_matches_df()
        
        if matches_df.empty or 'Priority_Score' not in matches_df.columns:
            return None
        
        # Create scatter plot
        fig = px.scatter(
            matches_df,
            x='Priority_Score',
            y='Confidence_Score',
            color='Match_Type',
            title="Priority Score vs Confidence Score",
            labels={
                'Priority_Score': 'Priority Score',
                'Confidence_Score': 'Confidence Score (%)',
                'Match_Type': 'Match Type'
            },
            hover_data=['Address_old']
        )
        
        return fig
    
    @staticmethod
    def display_top_priority_urls(results: MatchingResults, n: int = 10):
        """Display top N priority URLs."""
        matches_df = results.get_matches_df()
        
        if matches_df.empty or 'Priority_Score' not in matches_df.columns:
            st.info("No priority data available")
            return
        
        # Get top N by priority
        top_urls = matches_df.nlargest(n, 'Priority_Score')
        
        st.subheader(f"🔝 Top {n} Priority URLs")
        
        display_cols = ['Address_old', 'Address_new', 'Match_Type', 'Confidence_Score', 'Priority_Score']
        if 'Traffic' in top_urls.columns:
            display_cols.append('Traffic')
        
        st.dataframe(
            top_urls[display_cols],
            use_container_width=True,
            hide_index=True
        )
    
    @staticmethod
    def display_match_quality_metrics(results: MatchingResults):
        """Display overall match quality metrics."""
        matches_df = results.get_matches_df()
        unmatched_df = results.get_unmatched_df()
        
        if matches_df.empty:
            st.info("No matches to analyze")
            return
        
        st.subheader("📈 Match Quality Metrics")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            avg_confidence = matches_df['Confidence_Score'].mean()
            st.metric("Avg Confidence", f"{avg_confidence:.1f}%")
        
        with col2:
            high_confidence_count = (matches_df['Confidence_Score'] >= 80).sum()
            high_confidence_pct = (high_confidence_count / len(matches_df)) * 100
            st.metric("High Confidence", f"{high_confidence_pct:.1f}%")
        
        with col3:
            if 'Priority_Score' in matches_df.columns:
                avg_priority = matches_df['Priority_Score'].mean()
                st.metric("Avg Priority", f"{avg_priority:.2f}")
            else:
                st.metric("Avg Priority", "N/A")
        
        with col4:
            total_urls = len(matches_df) + len(unmatched_df)
            match_rate = (len(matches_df) / total_urls) * 100 if total_urls > 0 else 0
            st.metric("Match Rate", f"{match_rate:.1f}%")
    
    @staticmethod
    def display_processing_timeline(stats: ProcessingStats):
        """Display processing time breakdown."""
        st.subheader("⏱️ Processing Timeline")
        
        # Create a simple timeline chart
        timeline_data = {
            'Step': ['Exact Matching', 'Fuzzy Matching', 'Vector Matching', 'H1 Matching', 'AI Matching'],
            'Count': [stats.exact_matches, stats.fuzzy_matches, stats.vector_matches, stats.h1_matches, stats.ai_matches]
        }
        
        fig = px.bar(
            timeline_data,
            x='Step',
            y='Count',
            title="Matches Found by Method",
            labels={'Count': 'Number of Matches', 'Step': 'Matching Method'}
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        st.write(f"**Total Processing Time:** {stats.processing_time:.2f} seconds")
    
    @staticmethod
    def display_error_summary(errors: List[str]):
        """Display a summary of errors encountered during processing."""
        if not errors:
            return
        
        st.subheader("⚠️ Processing Warnings")
        
        for i, error in enumerate(errors, 1):
            st.warning(f"{i}. {error}")
    
    @staticmethod
    def format_url_for_display(url: str, max_length: int = 50) -> str:
        """Format URL for display in tables."""
        if len(url) <= max_length:
            return url
        
        # Truncate in the middle to preserve domain and end
        if "://" in url:
            protocol, rest = url.split("://", 1)
            if "/" in rest:
                domain, path = rest.split("/", 1)
                if len(domain) + len(protocol) + 3 < max_length - 10:
                    remaining = max_length - len(protocol) - len(domain) - 6
                    if remaining > 10:
                        truncated_path = path[:remaining//2] + "..." + path[-(remaining//2):]
                        return f"{protocol}://{domain}/{truncated_path}"
        
        # Simple truncation if above doesn't work
        return url[:max_length-3] + "..."
    
    @staticmethod
    def create_downloadable_report(results: MatchingResults, stats: ProcessingStats) -> str:
        """Create a text report summarizing the matching results."""
        report = []
        
        report.append("# URL Matching Report")
        report.append("=" * 50)
        report.append("")
        
        # Summary
        matches_df = results.get_matches_df()
        unmatched_df = results.get_unmatched_df()
        total_urls = len(matches_df) + len(unmatched_df)
        
        report.append("## Summary")
        report.append(f"Total URLs processed: {total_urls}")
        report.append(f"Successfully matched: {len(matches_df)}")
        report.append(f"Unmatched: {len(unmatched_df)}")
        report.append(f"Match rate: {(len(matches_df) / total_urls * 100):.1f}%" if total_urls > 0 else "Match rate: 0%")
        report.append("")
        
        # Match breakdown
        report.append("## Matches by Type")
        if not matches_df.empty:
            match_counts = matches_df['Match_Type'].value_counts()
            for match_type, count in match_counts.items():
                report.append(f"- {match_type}: {count}")
        report.append("")
        
        # Processing stats
        report.append("## Processing Statistics")
        report.append(f"Processing time: {stats.processing_time:.2f} seconds")
        report.append(f"Total old URLs: {stats.total_old_urls}")
        report.append(f"Total new URLs: {stats.total_new_urls}")
        report.append("")
        
        # Quality metrics
        if not matches_df.empty:
            report.append("## Quality Metrics")
            avg_confidence = matches_df['Confidence_Score'].mean()
            report.append(f"Average confidence score: {avg_confidence:.1f}%")
            
            high_conf_count = (matches_df['Confidence_Score'] >= 80).sum()
            high_conf_pct = (high_conf_count / len(matches_df)) * 100
            report.append(f"High confidence matches (≥80%): {high_conf_pct:.1f}%")
            
            if 'Priority_Score' in matches_df.columns:
                avg_priority = matches_df['Priority_Score'].mean()
                report.append(f"Average priority score: {avg_priority:.2f}")
        
        return "\n".join(report)