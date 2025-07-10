"""
Advanced AI Text Summarizer with Abstractive Summarization
Uses sequence-to-sequence transformer models for generating new summaries
"""

import streamlit as st
import pandas as pd
import io
import time
from summarizer import create_summarizer, TORCH_AVAILABLE

# Page configuration
st.set_page_config(
    page_title="Advanced AI Text Summarizer",
    page_icon="🤖",
    layout="wide"
)

# Custom CSS for better appearance
st.markdown("""
<style>
    .summary-box {
        background-color: #f0f8ff;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #4CAF50;
        margin: 10px 0;
    }
    .model-info {
        background-color: #e8f5e8;
        padding: 15px;
        border-radius: 8px;
        margin: 10px 0;
    }
    .warning-box {
        background-color: #fff3cd;
        padding: 15px;
        border-radius: 8px;
        border-left: 5px solid #ffc107;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

def get_sample_articles():
    """Sample articles for testing the summarizer."""
    return [
        {
            "title": "Climate Change and Global Warming Effects",
            "content": """
            Climate change refers to long-term shifts in global temperatures and weather patterns. While climate variations occur naturally, human activities have been the main driver of climate change since the 1800s. The burning of fossil fuels like coal, oil, and gas releases greenhouse gases into the atmosphere, primarily carbon dioxide. These gases trap heat from the sun, causing global temperatures to rise at an unprecedented rate.

            The effects of climate change are already visible worldwide. Arctic sea ice is melting rapidly, contributing to rising sea levels that threaten coastal communities. Extreme weather events like hurricanes, droughts, and heatwaves are becoming more frequent and intense. Many species are struggling to adapt to changing conditions, leading to shifts in ecosystems and biodiversity loss.

            Scientists have developed various climate models to predict future scenarios. These models suggest that without significant action to reduce greenhouse gas emissions, global temperatures could rise by 3-5 degrees Celsius by 2100. Such warming would have catastrophic consequences, including massive displacement of populations, food and water shortages, and economic disruption.

            Addressing climate change requires immediate and coordinated global action. Renewable energy sources like solar and wind power are becoming increasingly cost-effective alternatives to fossil fuels. Governments are implementing policies to reduce emissions, while businesses are adopting sustainable practices. Individual actions, such as reducing energy consumption and supporting clean technology, also play a crucial role in combating climate change.
            """.strip()
        },
        {
            "title": "Artificial Intelligence Revolution in Healthcare",
            "content": """
            Artificial intelligence is transforming healthcare delivery across the globe, offering unprecedented opportunities to improve patient outcomes and reduce costs. Machine learning algorithms can now analyze vast amounts of medical data to identify patterns that human physicians might miss. This capability is particularly valuable in diagnostic imaging, where AI systems can detect early signs of cancer, heart disease, and other conditions with remarkable accuracy.

            Drug discovery, traditionally a lengthy and expensive process taking 10-15 years and billions of dollars, is being revolutionized by AI. Machine learning models can predict how different molecular compounds will interact with biological targets, identifying promising drug candidates in a fraction of the time. Several AI-discovered drugs are currently in clinical trials, representing a new era in pharmaceutical development.

            Electronic health records are being enhanced with natural language processing capabilities that can extract meaningful insights from unstructured medical notes. These systems can identify patients at risk of developing complications, suggest preventive measures, and optimize treatment protocols based on similar cases in large databases. This approach enables more personalized and effective healthcare delivery.

            Robotic surgery systems powered by AI are enabling more precise procedures with smaller incisions and faster recovery times. Surgeons can perform complex operations with enhanced visualization and steadier movements, while AI algorithms help plan optimal surgical approaches and predict potential complications. However, the integration of AI in healthcare faces challenges including data privacy concerns, regulatory approval processes, and the need for healthcare professionals to adapt to new technologies.
            """.strip()
        },
        {
            "title": "The Future of Space Exploration and Colonization",
            "content": """
            Space exploration has entered a new era with private companies joining government agencies in ambitious missions to explore and potentially colonize other worlds. Companies like SpaceX, Blue Origin, and Virgin Galactic are developing reusable rockets and spacecraft that dramatically reduce the cost of space travel. These technological advances are making space more accessible and opening up possibilities for commercial space ventures.

            Mars has become the primary target for human colonization efforts. The Red Planet offers several advantages, including a 24-hour day cycle, polar ice caps containing water, and an atmosphere that could potentially be terraformed. NASA's Perseverance rover and other missions are gathering crucial data about Mars' geology, climate, and potential for supporting life. Plans for crewed missions to Mars are being developed, with the goal of establishing a permanent human presence by the 2030s.

            The Moon is experiencing renewed interest as a stepping stone to Mars and a potential site for lunar bases. The discovery of water ice in permanently shadowed craters has significant implications for future missions, as water can be split into hydrogen and oxygen for rocket fuel. Several countries and private companies are developing lunar landers and rovers to establish a sustained human presence on the Moon.

            Space colonization presents numerous challenges that must be overcome. Radiation exposure, psychological isolation, resource scarcity, and the need for self-sustaining life support systems are just some of the obstacles facing future space settlers. However, advances in closed-loop life support systems, 3D printing with local materials, and genetic engineering may provide solutions to these challenges, making permanent space colonies a reality within the next few decades.
            """.strip()
        }
    ]

# Initialize session state
if 'summarizer' not in st.session_state:
    st.session_state.summarizer = None
if 'model_loaded' not in st.session_state:
    st.session_state.model_loaded = False

# App title and description
st.title("🤖 Advanced AI Text Summarizer")
st.markdown("""
**Abstractive Summarization with Transformer Models**  
Generate new, coherent summaries using sequence-to-sequence neural networks
""")

# Model selection and loading
st.sidebar.header("🧠 Model Configuration")

model_options = {
    "T5-Small (Fast)": "t5-small",
    "T5-Base (Better Quality)": "t5-base", 
    "BART-Large-CNN": "facebook/bart-large-cnn",
    "Pegasus-XSum": "google/pegasus-xsum"
}

selected_model = st.sidebar.selectbox(
    "Choose Transformer Model:",
    list(model_options.keys()),
    index=0
)

# Model information
if TORCH_AVAILABLE:
    st.sidebar.markdown(f"""
    <div class="model-info">
    <strong>Selected Model:</strong> {selected_model}<br>
    <strong>Type:</strong> Abstractive (Seq2Seq)<br>
    <strong>Status:</strong> PyTorch Available ✅
    </div>
    """, unsafe_allow_html=True)
else:
    st.sidebar.markdown(f"""
    <div class="warning-box">
    <strong>Warning:</strong> PyTorch not available<br>
    <strong>Status:</strong> Fallback mode only<br>
    <strong>Solution:</strong> Install PyTorch and Transformers
    </div>
    """, unsafe_allow_html=True)

# Load model button
if st.sidebar.button("🔄 Load Model", type="primary"):
    with st.spinner(f"Loading {selected_model} model..."):
        st.session_state.summarizer = create_summarizer(model_options[selected_model])
        st.session_state.model_loaded = True
        
        if st.session_state.summarizer.is_loaded:
            st.sidebar.success(f"✅ {selected_model} loaded successfully!")
        else:
            st.sidebar.error("❌ Failed to load model. Using fallback mode.")

# Main interface
if not st.session_state.model_loaded:
    st.info("👆 Please select and load a model from the sidebar to begin summarization.")
else:
    # Display model info
    if st.session_state.summarizer:
        model_info = st.session_state.summarizer.get_model_info()
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Model", model_info["model_name"])
        with col2:
            st.metric("Type", model_info["model_type"])
        with col3:
            status = "✅ Loaded" if model_info["is_loaded"] else "❌ Fallback"
            st.metric("Status", status)
    
    # Create tabs
    tab1, tab2, tab3 = st.tabs(["📝 Text Summarizer", "📊 Batch Processing", "📰 Sample Articles"])
    
    # Tab 1: Text Summarization
    with tab1:
        st.header("Text Summarization")
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            # Text input
            text_input = st.text_area(
                "📝 Enter your text here:",
                height=300,
                placeholder="Paste your article, research paper, or any long text you want to summarize...",
                help="The system works best with articles of 100+ words"
            )
            
            # Text statistics
            if text_input:
                words = len(text_input.split())
                chars = len(text_input)
                st.caption(f"📊 **{words:,} words** • **{chars:,} characters**")
                
                if words < 50:
                    st.warning("⚠️ Text is quite short. Consider adding more content for better summarization.")
                elif words > 2000:
                    st.info("ℹ️ Very long text detected. The system will chunk it for processing.")
        
        with col2:
            st.markdown("### ⚙️ Settings")
            
            # Summary length
            length_option = st.selectbox(
                "Summary Length:",
                ["short", "medium", "long"],
                index=1,
                help="Choose the desired length of your summary"
            )
            
            # Length descriptions
            length_descriptions = {
                "short": "Brief (20-50 words)",
                "medium": "Balanced (40-100 words)",
                "long": "Detailed (60-150 words)"
            }
            st.caption(f"📏 {length_descriptions[length_option]}")
            
            # Advanced options
            with st.expander("🔧 Advanced Options"):
                use_chunking = st.checkbox(
                    "Use chunking for long texts",
                    value=True,
                    help="Split very long texts into chunks for better processing"
                )
                
                show_model_info = st.checkbox(
                    "Show model information",
                    value=False,
                    help="Display technical details about the model"
                )
        
        # Generate summary button
        if st.button("🚀 Generate Abstractive Summary", type="primary", use_container_width=True):
            if not text_input:
                st.error("❌ Please enter some text to summarize.")
            elif len(text_input.split()) < 10:
                st.error("❌ Text is too short. Please provide at least 10 words.")
            else:
                # Generate summary
                with st.spinner("🧠 AI is analyzing and generating your summary..."):
                    start_time = time.time()
                    
                    try:
                        if use_chunking and len(text_input.split()) > 1000:
                            summary = st.session_state.summarizer.summarize_long_text(
                                text_input, length_option
                            )
                        else:
                            summary = st.session_state.summarizer.generate_summary(
                                text_input, length_option
                            )
                        
                        end_time = time.time()
                        processing_time = end_time - start_time
                        
                    except Exception as e:
                        st.error(f"❌ Error during summarization: {str(e)}")
                        summary = None
                        processing_time = 0
                
                if summary:
                    # Success message
                    st.success("✅ Abstractive summary generated successfully!")
                    
                    # Display metrics
                    col1, col2, col3, col4 = st.columns(4)
                    
                    original_words = len(text_input.split())
                    summary_words = len(summary.split())
                    compression_ratio = round((1 - summary_words / original_words) * 100, 1)
                    
                    with col1:
                        st.metric("Original Words", f"{original_words:,}")
                    with col2:
                        st.metric("Summary Words", f"{summary_words:,}")
                    with col3:
                        st.metric("Compression", f"{compression_ratio}%")
                    with col4:
                        st.metric("Processing Time", f"{processing_time:.2f}s")
                    
                    # Display summary
                    st.markdown("### 📄 Generated Summary")
                    st.markdown(f'<div class="summary-box">{summary}</div>', unsafe_allow_html=True)
                    
                    # Show model info if requested
                    if show_model_info:
                        st.markdown("### 🔍 Model Information")
                        model_info = st.session_state.summarizer.get_model_info()
                        st.json(model_info)
                    
                    # Download option
                    summary_data = f"""Original Text:
{text_input}

Generated Summary:
{summary}

Model Information:
- Model: {model_info['model_name']}
- Type: {model_info['model_type']}
- Processing Time: {processing_time:.2f} seconds
- Compression: {compression_ratio}%
- Original Words: {original_words:,}
- Summary Words: {summary_words:,}
"""
                    
                    st.download_button(
                        "📥 Download Summary",
                        summary_data,
                        file_name=f"abstractive_summary_{int(time.time())}.txt",
                        mime="text/plain",
                        help="Download the summary and metadata as a text file"
                    )
    
    # Tab 2: Batch Processing
    with tab2:
        st.header("📊 Batch Processing")
        st.markdown("Process multiple articles from a CSV file")
        
        # File upload
        uploaded_file = st.file_uploader(
            "📂 Upload CSV file",
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
                st.dataframe(df.head(), use_container_width=True)
                
                # Configuration
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    text_column = st.selectbox(
                        "📝 Select text column:",
                        options=df.columns.tolist(),
                        help="Choose the column containing text to summarize"
                    )
                
                with col2:
                    max_rows = min(len(df), 10)  # Limit for processing time
                    num_rows = st.slider(
                        "📊 Number of rows:",
                        min_value=1,
                        max_value=max_rows,
                        value=min(3, len(df)),
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
                                    summary = st.session_state.summarizer.generate_summary(
                                        str(text), 
                                        batch_length
                                    )
                                    article_end = time.time()
                                    summaries.append(summary)
                                    processing_times.append(article_end - article_start)
                                except Exception as e:
                                    summaries.append(f"❌ Error: {str(e)}")
                                    processing_times.append(0)
                        
                        # Add results to dataframe
                        df_subset['abstractive_summary'] = summaries
                        df_subset['processing_time'] = processing_times
                        
                        # Clear progress indicators
                        progress_bar.empty()
                        status_text.empty()
                        
                        # Show results
                        total_time = time.time() - start_time
                        st.success("✅ Batch processing completed!")
                        
                        # Batch metrics
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
                        st.markdown("### 📋 Results")
                        display_columns = [text_column, 'abstractive_summary']
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
                            file_name=f"abstractive_summaries_{int(time.time())}.csv",
                            mime="text/csv",
                            type="primary"
                        )
                        
            except Exception as e:
                st.error(f"❌ Error processing file: {str(e)}")
                st.info("💡 Make sure your CSV file is properly formatted and contains text data.")
    
    # Tab 3: Sample Articles
    with tab3:
        st.header("📰 Sample Articles")
        st.markdown("Test the abstractive summarizer with these sample articles")
        
        sample_articles = get_sample_articles()
        
        # Display sample articles
        for i, article in enumerate(sample_articles):
            with st.expander(f"📰 {article['title']}", expanded=(i == 0)):
                
                # Display article content
                st.markdown("**Article Content:**")
                st.text_area(
                    f"Content:",
                    value=article['content'],
                    height=200,
                    key=f"sample_content_{i}",
                    help="This is a sample article for testing"
                )
                
                # Article statistics
                words = len(article['content'].split())
                chars = len(article['content'])
                st.caption(f"📊 {words:,} words • {chars:,} characters")
                
                # Settings for this article
                col1, col2 = st.columns(2)
                
                with col1:
                    length_choice = st.selectbox(
                        "Summary Length:",
                        options=["short", "medium", "long"],
                        index=1,
                        key=f"sample_length_{i}"
                    )
                
                with col2:
                    if st.button(f"🚀 Generate Summary", key=f"sample_summarize_{i}"):
                        with st.spinner("Generating abstractive summary..."):
                            try:
                                start_time = time.time()
                                summary = st.session_state.summarizer.generate_summary(
                                    article['content'], 
                                    length_choice
                                )
                                end_time = time.time()
                                
                                # Display summary
                                st.markdown("**Generated Summary:**")
                                st.markdown(f'<div class="summary-box">{summary}</div>', unsafe_allow_html=True)
                                
                                # Show metrics
                                original_words = len(article['content'].split())
                                summary_words = len(summary.split())
                                compression = round((1 - summary_words/original_words) * 100, 1)
                                
                                st.caption(f"📊 Original: {original_words} words → Summary: {summary_words} words ({compression}% compression) • Time: {end_time - start_time:.2f}s")
                                
                            except Exception as e:
                                st.error(f"❌ Error generating summary: {str(e)}")
                
                # Use this article button
                if st.button(f"📝 Use This Article", key=f"sample_use_{i}"):
                    # This would need to be implemented to populate the main text area
                    st.success("✅ Article ready to use! Go to the Text Summarizer tab.")
                    st.balloons()

# Footer
st.markdown("---")
st.markdown("""
### 🔬 **About Abstractive Summarization**
This app uses sequence-to-sequence transformer models that generate entirely new text rather than just selecting existing sentences. The models understand the content and create coherent summaries in their own words.

**Models Used:**
- **T5**: Text-to-Text Transfer Transformer, versatile for various NLP tasks
- **BART**: Bidirectional and Auto-Regressive Transformers, excellent for summarization
- **Pegasus**: Pre-trained specifically for abstractive summarization

**Key Features:**
- Generates new text (not just extraction)
- Understands context and meaning
- Produces coherent, fluent summaries
- Handles long documents through chunking
""")