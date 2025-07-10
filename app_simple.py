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
tab1, tab2, tab3 = st.tabs(["✏️ Text Summarizer", "📊 File Upload", "📰 Sample Articles"])

# Tab 1: Text Summarizer
with tab1:
    st.header("Paste Your Text")
    
    # Text input
    text_input = st.text_area(
        "Enter your text here:",
        height=300,
        placeholder="Paste your article, news story, or any text you want to summarize..."
    )
    
    # Settings
    col1, col2 = st.columns([3, 1])
    
    with col1:
        if text_input:
            words = len(text_input.split())
            st.caption(f"📊 {words:,} words")
    
    with col2:
        summary_length = st.selectbox(
            "Summary length:",
            ["Short (100 words)", "Medium (150 words)", "Long (200 words)"],
            index=1
        )
        
        # Extract number from selection
        length_map = {
            "Short (100 words)": 100,
            "Medium (150 words)": 150,
            "Long (200 words)": 200
        }
        max_length = length_map[summary_length]
    
    # Generate summary
    if st.button("🚀 Generate Summary", type="primary", use_container_width=True):
        if not text_input:
            st.error("Please enter some text to summarize")
        elif len(text_input.split()) < 20:
            st.error("Text is too short. Please enter at least 20 words.")
        else:
            with st.spinner("Generating summary..."):
                start_time = time.time()
                summary = summarizer.summarize(text_input, max_length)
                end_time = time.time()
            
            st.success("Summary generated!")
            
            # Show metrics
            col1, col2, col3 = st.columns(3)
            original_words = len(text_input.split())
            summary_words = len(summary.split())
            compression = round((1 - summary_words/original_words) * 100, 1)
            
            with col1:
                st.metric("Original", f"{original_words} words")
            with col2:
                st.metric("Summary", f"{summary_words} words")
            with col3:
                st.metric("Compression", f"{compression}%")
            
            # Display summary
            st.markdown("### 📄 Summary")
            st.info(summary)
            
            # Download
            summary_data = f"Original Text:\n{text_input}\n\nSummary:\n{summary}\n\nStats:\n- Original: {original_words} words\n- Summary: {summary_words} words\n- Compression: {compression}%\n- Time: {end_time - start_time:.2f}s"
            
            st.download_button(
                "📥 Download Summary",
                summary_data,
                file_name="summary.txt",
                mime="text/plain"
            )

# Tab 2: File Upload
with tab2:
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

# Tab 3: Sample Articles
with tab3:
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