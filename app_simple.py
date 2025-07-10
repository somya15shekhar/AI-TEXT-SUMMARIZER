"""
Simple AI Text Summarizer
Clean, easy-to-deploy Streamlit app for text summarization
"""

import streamlit as st
import pandas as pd
import time
from text_summarizer import TextSummarizer
from file_handler import (
    read_csv_file, get_text_columns, process_batch_summaries, 
    create_download_data, get_sample_data
)

# Page setup
st.set_page_config(
    page_title="AI Text Summarizer",
    page_icon="📝",
    layout="wide"
)

# Initialize summarizer
@st.cache_resource
def load_summarizer():
    return TextSummarizer()

# Load summarizer
summarizer = load_summarizer()

# Header
st.title("📝 AI Text Summarizer")
st.markdown(f"**Model:** {summarizer.get_status()}")
st.markdown("---")

# Main tabs
tab1, tab2 = st.tabs(["📊 File Upload", "📰 Sample Articles"])

# Tab 1: File Upload
with tab1:
    st.header("Upload CSV File")
    
    uploaded_file = st.file_uploader(
        "Choose a CSV file containing articles",
        type="csv",
        help="Upload a CSV file with a column containing text to summarize"
    )
    
    if uploaded_file:
        df = read_csv_file(uploaded_file)
        
        if df is not None:
            st.success("File uploaded successfully!")
            
            # Show preview
            st.subheader("Data Preview")
            st.dataframe(df.head())
            
            # Get text columns
            text_columns = get_text_columns(df)
            
            # Settings
            col1, col2 = st.columns(2)
            
            with col1:
                text_column = st.selectbox(
                    "Select text column:",
                    text_columns
                )
            
            with col2:
                max_rows = st.slider(
                    "Number of rows to process:",
                    min_value=1,
                    max_value=min(20, len(df)),
                    value=min(5, len(df))
                )
            
            # Process files
            if st.button("📝 Generate Summaries", type="primary"):
                with st.spinner(f"Processing {max_rows} articles..."):
                    progress_bar = st.progress(0)
                    
                    # Process batch
                    results_df = process_batch_summaries(
                        df, text_column, summarizer, max_rows
                    )
                    
                    progress_bar.progress(100)
                
                st.success("Processing complete!")
                
                # Show results
                st.subheader("Results")
                st.dataframe(results_df[[text_column, 'summary']])
                
                # Download results
                csv_data = create_download_data(results_df, text_column)
                st.download_button(
                    "📥 Download Results",
                    csv_data,
                    file_name="summaries.csv",
                    mime="text/csv"
                )
        else:
            st.error("Error reading file. Please check your CSV format.")

# Tab 2: Sample Articles
with tab2:
    st.header("Try Sample Articles")
    
    samples = get_sample_data()
    
    for i, sample in enumerate(samples):
        with st.expander(f"📰 {sample['title']}", expanded=(i == 0)):
            st.text_area(
                "Article:",
                sample['content'],
                height=150,
                disabled=True,
                key=f"sample_{i}"
            )
            
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button(f"📝 Summarize", key=f"btn_{i}"):
                    with st.spinner("Summarizing..."):
                        summary = summarizer.summarize(sample['content'])
                    
                    st.markdown("**Summary:**")
                    st.success(summary)
            
            with col2:
                if st.button(f"📋 Copy to Main", key=f"copy_{i}"):
                    st.session_state.text_to_copy = sample['content']
                    st.success("Text copied! Go to Text Summarizer tab.")

# Handle text copying
if 'text_to_copy' in st.session_state:
    st.session_state.text_input = st.session_state.text_to_copy
    del st.session_state.text_to_copy

# Footer
st.markdown("---")
st.markdown("**Tips:** For best results, use articles with 50+ words. The AI will create concise summaries preserving key information.")