import streamlit as st
import pandas as pd
import io
import time
from modules.summarizer import SummarizerEngine
from modules.utils import validate_text, calculate_metrics
from modules.sample_data import get_sample_articles

# Page configuration
st.set_page_config(
    page_title="AI Text Summarizer",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for better UI
st.markdown("""
<style>
    .metric-container {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .summary-box {
        background-color: #e8f4fd;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'summarizer' not in st.session_state:
    st.session_state.summarizer = None
if 'model_loaded' not in st.session_state:
    st.session_state.model_loaded = False

# Title and description
st.title("🤖 AI Text Summarizer")
st.markdown("Transform long articles into concise summaries using advanced AI models")

# Initialize summarizer
@st.cache_resource
def load_summarizer():
    """Load and cache the summarization engine"""
    return SummarizerEngine()

# Load model with progress indicator
if not st.session_state.model_loaded:
    with st.spinner("🔄 Loading AI model... This may take a moment on first run."):
        st.session_state.summarizer = load_summarizer()
        st.session_state.model_loaded = True
        
    # Show model status
    if st.session_state.summarizer.model_type == "transformer":
        st.success("✅ Advanced transformer model loaded successfully!")
        st.info("Using DistilBART - optimized for fast, high-quality summaries")
    else:
        st.success("✅ Enhanced extractive summarization ready!")
        st.info("Using advanced rule-based summarization with multi-factor scoring")

# Create main tabs
tab1, tab2, tab3 = st.tabs(["📝 Text Summarizer", "📊 Batch Processing", "🎯 Sample Articles"])

# Tab 1: Main Text Summarization
with tab1:
    st.header("Paste Your Text for Instant AI Summary")
    
    # Create columns for layout
    col1, col2 = st.columns([3, 1])
    
    with col1:
        # Text input area
        article_text = st.text_area(
            "📋 Paste your article or text here:",
            height=350,
            placeholder="Paste your article, blog post, research paper, or any text you want to summarize...",
            help="Enter the text you want to summarize. The AI will analyze and create a concise summary."
        )
        
        # Real-time text statistics
        if article_text:
            words = len(article_text.split())
            chars = len(article_text)
            st.caption(f"📊 **{words:,} words** | **{chars:,} characters**")
            
            # Validation warnings
            if words < 50:
                st.warning("⚠️ Text might be too short for effective summarization. Consider adding more content.")
            elif words > 5000:
                st.warning("⚠️ Very long text detected. Processing may take longer.")
    
    with col2:
        st.markdown("### ⚙️ Summary Settings")
        
        # Length selection
        length_option = st.selectbox(
            "📏 Summary Length:",
            options=["short", "medium", "long"],
            index=1,
            help="Choose how detailed you want your summary to be"
        )
        
        # Length descriptions
        length_info = {
            "short": "🔸 Brief overview (2-3 sentences)",
            "medium": "🔸 Balanced summary (4-6 sentences)",
            "long": "🔸 Detailed summary (7-10 sentences)"
        }
        st.markdown(length_info[length_option])
        
        # Model info
        st.markdown("### 🧠 AI Model")
        if st.session_state.summarizer:
            model_name = "DistilBART" if st.session_state.summarizer.model_type == "transformer" else "Enhanced Extractive"
            st.info(f"**{model_name}** Model")
        
        # Quick actions
        st.markdown("### 🚀 Quick Actions")
        col_clear, col_sample = st.columns(2)
        
        with col_clear:
            if st.button("🗑️ Clear Text", help="Clear the text area"):
                st.rerun()
        
        with col_sample:
            if st.button("📰 Load Sample", help="Load a sample article"):
                sample_articles = get_sample_articles()
                st.session_state.sample_loaded = sample_articles[0]["content"]
                st.rerun()
    
    # Use sample text if available
    if 'sample_loaded' in st.session_state:
        article_text = st.session_state.sample_loaded
        del st.session_state.sample_loaded
    
    # Summarize button
    st.markdown("---")
    col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
    
    with col_btn2:
        summarize_clicked = st.button(
            "🚀 Generate AI Summary",
            type="primary",
            use_container_width=True,
            help="Click to generate an AI-powered summary of your text"
        )
    
    # Generate summary
    if summarize_clicked:
        if not article_text or not article_text.strip():
            st.error("❌ Please paste some text to summarize.")
        else:
            validation_result = validate_text(article_text)
            if not validation_result["valid"]:
                st.error(f"❌ {validation_result['message']}")
            else:
                # Generate summary with progress
                with st.spinner("🧠 AI is analyzing your text and generating summary..."):
                    start_time = time.time()
                    
                    try:
                        summary = st.session_state.summarizer.summarize(
                            article_text, 
                            length=length_option
                        )
                        end_time = time.time()
                        
                        # Calculate metrics
                        metrics = calculate_metrics(article_text, summary)
                        
                        # Success message
                        st.success("✅ Summary generated successfully!")
                        
                        # Display metrics
                        st.markdown("### 📊 Summary Statistics")
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            st.metric("Original Words", f"{metrics['original_words']:,}")
                        with col2:
                            st.metric("Summary Words", f"{metrics['summary_words']:,}")
                        with col3:
                            st.metric("Compression", f"{metrics['compression_ratio']:.1f}%")
                        with col4:
                            st.metric("Processing Time", f"{end_time - start_time:.2f}s")
                        
                        # Display summary
                        st.markdown("### 📄 AI Generated Summary")
                        st.markdown(f'<div class="summary-box">{summary}</div>', unsafe_allow_html=True)
                        
                        # Download option
                        summary_data = f"Original Text:\n{article_text}\n\nAI Summary:\n{summary}\n\nMetrics:\n- Original Words: {metrics['original_words']}\n- Summary Words: {metrics['summary_words']}\n- Compression: {metrics['compression_ratio']:.1f}%\n- Processing Time: {end_time - start_time:.2f}s"
                        
                        st.download_button(
                            label="📥 Download Summary",
                            data=summary_data,
                            file_name=f"ai_summary_{int(time.time())}.txt",
                            mime="text/plain"
                        )
                        
                    except Exception as e:
                        st.error(f"❌ Error generating summary: {str(e)}")
                        st.info("💡 Try using shorter text or check your internet connection.")

# Tab 2: Batch Processing
with tab2:
    st.header("📊 Batch Processing from CSV")
    st.markdown("Upload a CSV file to summarize multiple articles at once")
    
    # File upload
    uploaded_file = st.file_uploader(
        "📂 Choose CSV file",
        type=["csv"],
        help="Upload a CSV file containing articles to summarize"
    )
    
    if uploaded_file is not None:
        try:
            # Read CSV
            df = pd.read_csv(uploaded_file)
            st.success("✅ File uploaded successfully!")
            
            # Data preview
            st.markdown("### 📋 Data Preview")
            st.dataframe(df.head(10), use_container_width=True)
            
            # Configuration
            st.markdown("### ⚙️ Processing Configuration")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                text_column = st.selectbox(
                    "📝 Select text column:",
                    options=df.columns.tolist(),
                    help="Choose the column containing text to summarize"
                )
            
            with col2:
                max_rows = min(len(df), 50)  # Limit for performance
                num_rows = st.slider(
                    "📊 Number of rows to process:",
                    min_value=1,
                    max_value=max_rows,
                    value=min(5, len(df)),
                    help=f"Select how many rows to process (max {max_rows})"
                )
            
            with col3:
                batch_length = st.selectbox(
                    "📏 Summary length:",
                    options=["short", "medium", "long"],
                    index=1,
                    help="Choose summary length for all articles"
                )
            
            # Process button
            if st.button("⚡ Process Articles", type="primary", use_container_width=True):
                if text_column not in df.columns:
                    st.error("❌ Selected column not found in the dataset.")
                else:
                    # Process batch
                    df_subset = df.head(num_rows).copy()
                    
                    # Progress tracking
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    summaries = []
                    processing_times = []
                    start_time = time.time()
                    
                    # Process each article
                    for i, text in enumerate(df_subset[text_column]):
                        # Update progress
                        progress = (i + 1) / len(df_subset)
                        progress_bar.progress(progress)
                        status_text.text(f"Processing article {i+1} of {len(df_subset)}...")
                        
                        # Generate summary
                        if pd.isna(text) or not str(text).strip():
                            summaries.append("❌ No text to summarize")
                            processing_times.append(0)
                        else:
                            try:
                                article_start = time.time()
                                summary = st.session_state.summarizer.summarize(
                                    str(text), 
                                    length=batch_length
                                )
                                article_end = time.time()
                                summaries.append(summary)
                                processing_times.append(article_end - article_start)
                            except Exception as e:
                                summaries.append(f"❌ Error: {str(e)}")
                                processing_times.append(0)
                    
                    # Add results to dataframe
                    df_subset['ai_summary'] = summaries
                    df_subset['processing_time'] = processing_times
                    
                    # Clear progress indicators
                    progress_bar.empty()
                    status_text.empty()
                    
                    # Show results
                    total_time = time.time() - start_time
                    st.success("✅ Batch processing completed!")
                    
                    # Batch metrics
                    st.markdown("### 📊 Batch Processing Results")
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Articles Processed", len(df_subset))
                    with col2:
                        successful = sum(1 for s in summaries if not s.startswith("❌"))
                        st.metric("Successful", successful)
                    with col3:
                        avg_time = sum(processing_times) / max(len(processing_times), 1)
                        st.metric("Avg Time/Article", f"{avg_time:.2f}s")
                    with col4:
                        st.metric("Total Time", f"{total_time:.2f}s")
                    
                    # Results table
                    st.markdown("### 📋 Results Table")
                    display_columns = [text_column, 'ai_summary']
                    if 'processing_time' in df_subset.columns:
                        display_columns.append('processing_time')
                    
                    st.dataframe(
                        df_subset[display_columns], 
                        use_container_width=True,
                        height=400
                    )
                    
                    # Download results
                    csv_buffer = io.StringIO()
                    df_subset.to_csv(csv_buffer, index=False)
                    
                    st.download_button(
                        label="📥 Download Results as CSV",
                        data=csv_buffer.getvalue(),
                        file_name=f"ai_summaries_{int(time.time())}.csv",
                        mime="text/csv",
                        type="primary"
                    )
                    
        except Exception as e:
            st.error(f"❌ Error processing file: {str(e)}")
            st.info("💡 Make sure your CSV file is properly formatted and contains text data.")

# Tab 3: Sample Articles
with tab3:
    st.header("🎯 Try Sample Articles")
    st.markdown("Test the AI summarizer with these sample articles")
    
    sample_articles = get_sample_articles()
    
    # Display sample articles
    for i, article in enumerate(sample_articles):
        with st.expander(f"📰 {article['title']}", expanded=(i == 0)):
            st.markdown(f"**Category:** {article['category']}")
            st.markdown(f"**Source:** {article['source']}")
            st.markdown("**Content:**")
            st.text_area(
                f"Article {i+1} Content:",
                value=article['content'],
                height=200,
                key=f"sample_{i}",
                help="This is a sample article for testing"
            )
            
            col1, col2 = st.columns([1, 1])
            
            with col1:
                length_choice = st.selectbox(
                    "Summary Length:",
                    options=["short", "medium", "long"],
                    index=1,
                    key=f"length_{i}"
                )
            
            with col2:
                if st.button(f"🚀 Summarize Article {i+1}", key=f"summarize_{i}"):
                    with st.spinner("Generating summary..."):
                        try:
                            summary = st.session_state.summarizer.summarize(
                                article['content'], 
                                length=length_choice
                            )
                            st.success("✅ Summary generated!")
                            st.markdown("**AI Summary:**")
                            st.info(summary)
                            
                            # Metrics
                            metrics = calculate_metrics(article['content'], summary)
                            col_m1, col_m2, col_m3 = st.columns(3)
                            with col_m1:
                                st.metric("Original Words", metrics['original_words'])
                            with col_m2:
                                st.metric("Summary Words", metrics['summary_words'])
                            with col_m3:
                                st.metric("Compression", f"{metrics['compression_ratio']:.1f}%")
                                
                        except Exception as e:
                            st.error(f"❌ Error: {str(e)}")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 20px;">
    <p>🤖 AI Text Summarizer | Powered by DistilBART and Advanced NLP</p>
    <p>Built with Streamlit | Optimized for fast, accurate text summarization</p>
</div>
""", unsafe_allow_html=True)
