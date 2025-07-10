import streamlit as st
import pandas as pd
import io
import time
import re

# Page setup
st.set_page_config(
    page_title="AI Text Summarizer",
    page_icon="📝",
    layout="wide"
)

# Simple summarizer class
class SimpleSummarizer:
    def summarize(self, text, length="medium"):
        if not text or len(text.strip()) < 50:
            return "Text is too short to summarize effectively."
        
        # Clean and split text into sentences
        sentences = re.split(r'(?<=[.!?])\s+', text.strip())
        sentences = [s.strip() for s in sentences if len(s.strip()) > 10]
        
        if len(sentences) <= 2:
            return text
        
        # Determine number of sentences for summary
        sentence_count = {
            "short": 2,
            "medium": 3,
            "long": min(5, len(sentences))
        }
        target_sentences = sentence_count.get(length, 3)
        
        # Simple scoring algorithm
        word_freq = {}
        for word in text.lower().split():
            if len(word) > 3:
                word_freq[word] = word_freq.get(word, 0) + 1
        
        # Score each sentence
        sentence_scores = []
        for i, sentence in enumerate(sentences):
            words = sentence.lower().split()
            score = sum(word_freq.get(word, 0) for word in words if len(word) > 3)
            
            # Boost first sentences
            if i == 0:
                score *= 1.5
            elif i < len(sentences) * 0.3:
                score *= 1.2
            
            # Boost sentences with numbers or key words
            if any(char.isdigit() for char in sentence):
                score *= 1.1
            
            key_words = ['important', 'key', 'main', 'significant', 'shows', 'found']
            if any(word in sentence.lower() for word in key_words):
                score *= 1.2
            
            sentence_scores.append((score / len(words), i))
        
        # Select top sentences
        sentence_scores.sort(reverse=True)
        selected_indices = sorted([idx for _, idx in sentence_scores[:target_sentences]])
        
        # Create summary
        summary = '. '.join([sentences[i] for i in selected_indices])
        return summary.strip() + ('.' if not summary.endswith('.') else '')

# Sample data
def get_sample_articles():
    return [
        {
            "title": "Climate Change and Clean Energy",
            "content": "Climate change is one of the most pressing challenges of our time. Global temperatures are rising at an unprecedented rate due to human activities. The burning of fossil fuels releases greenhouse gases that trap heat in the atmosphere. Scientists worldwide agree that immediate action is needed to reduce emissions. Renewable energy sources like solar and wind power have become much more affordable and efficient. Many countries have set ambitious goals to reach net-zero emissions by 2050. The transition to clean energy is creating millions of new jobs globally. However, significant challenges remain in energy storage and grid infrastructure. Public support for climate action continues to grow as people experience more extreme weather events."
        },
        {
            "title": "Artificial Intelligence in Healthcare",
            "content": "Artificial intelligence is transforming healthcare in remarkable ways. AI systems can now analyze medical images faster and more accurately than human doctors in many cases. Machine learning algorithms help identify patterns in patient data that might be missed by traditional methods. Drug discovery is being accelerated through AI-powered research that can test thousands of compounds virtually. Electronic health records are being enhanced with AI to provide better insights into patient care. Robotic surgery systems guided by AI are making operations more precise and less invasive. However, concerns about data privacy and algorithm bias need to be addressed. Healthcare professionals require training to work effectively with these new AI tools."
        }
    ]

# Initialize session state
if 'summarizer' not in st.session_state:
    st.session_state.summarizer = SimpleSummarizer()

# Header
st.title("📝 AI Text Summarizer")
st.markdown("Transform long articles into concise summaries")

# Main tabs
tab1, tab2, tab3 = st.tabs(["✏️ Text Summarizer", "📊 Batch Processing", "📰 Sample Articles"])

# Tab 1: Text Summarizer
with tab1:
    col1, col2 = st.columns([3, 1])
    
    with col1:
        # Text input with session state
        if 'text_input' not in st.session_state:
            st.session_state.text_input = ""
        
        text_input = st.text_area(
            "📝 Paste your text here:",
            value=st.session_state.text_input,
            height=300,
            placeholder="Paste your article, blog post, or any text you want to summarize...",
            key="text_area"
        )
        
        if text_input:
            words = len(text_input.split())
            chars = len(text_input)
            st.caption(f"📊 {words:,} words • {chars:,} characters")
            
            if words < 20:
                st.warning("⚠️ Text might be too short for good summarization")
    
    with col2:
        st.markdown("### Settings")
        
        length = st.selectbox(
            "Summary length:",
            ["short", "medium", "long"],
            index=1
        )
        
        length_descriptions = {
            "short": "2 sentences",
            "medium": "3 sentences",
            "long": "4-5 sentences"
        }
        st.caption(f"📏 {length_descriptions[length]}")
        
        st.markdown("### Quick Actions")
        
        if st.button("🗑️ Clear Text"):
            st.session_state.text_input = ""
            st.rerun()
        
        if st.button("📰 Load Sample"):
            sample = get_sample_articles()[0]
            st.session_state.text_input = sample["content"]
            st.rerun()
    
    # Generate summary
    if st.button("🚀 Generate Summary", type="primary", use_container_width=True):
        if not text_input:
            st.error("Please enter some text to summarize")
        else:
            with st.spinner("Creating summary..."):
                start_time = time.time()
                summary = st.session_state.summarizer.summarize(text_input, length)
                end_time = time.time()
            
            st.success("Summary generated successfully!")
            
            # Metrics
            col1, col2, col3 = st.columns(3)
            original_words = len(text_input.split())
            summary_words = len(summary.split())
            compression = round((1 - summary_words/original_words) * 100, 1)
            
            with col1:
                st.metric("Original Words", original_words)
            with col2:
                st.metric("Summary Words", summary_words)
            with col3:
                st.metric("Compression", f"{compression}%")
            
            # Display summary
            st.markdown("### 📄 Summary")
            st.info(summary)
            
            # Download option
            summary_data = f"Original Text:\n{text_input}\n\nSummary:\n{summary}\n\nStats:\n- Original: {original_words} words\n- Summary: {summary_words} words\n- Compression: {compression}%"
            
            st.download_button(
                "📥 Download Summary",
                summary_data,
                file_name="summary.txt",
                mime="text/plain"
            )

# Tab 2: Batch Processing
with tab2:
    st.markdown("### Upload CSV File for Batch Processing")
    
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file)
            st.success("File uploaded successfully!")
            
            st.markdown("### Data Preview")
            st.dataframe(df.head())
            
            col1, col2 = st.columns(2)
            
            with col1:
                text_column = st.selectbox(
                    "Select text column:",
                    df.columns.tolist()
                )
            
            with col2:
                batch_length = st.selectbox(
                    "Summary length:",
                    ["short", "medium", "long"],
                    index=1
                )
            
            num_rows = st.slider(
                "Number of rows to process:",
                1, min(20, len(df)), 5
            )
            
            if st.button("Process Articles", type="primary"):
                progress_bar = st.progress(0)
                summaries = []
                
                for i in range(num_rows):
                    progress_bar.progress((i + 1) / num_rows)
                    text = str(df.iloc[i][text_column])
                    summary = st.session_state.summarizer.summarize(text, batch_length)
                    summaries.append(summary)
                
                df_results = df.head(num_rows).copy()
                df_results['summary'] = summaries
                
                st.success("Processing completed!")
                st.dataframe(df_results[[text_column, 'summary']])
                
                # Download results
                csv_buffer = io.StringIO()
                df_results.to_csv(csv_buffer, index=False)
                
                st.download_button(
                    "📥 Download Results",
                    csv_buffer.getvalue(),
                    file_name="batch_summaries.csv",
                    mime="text/csv"
                )
        
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")

# Tab 3: Sample Articles
with tab3:
    st.markdown("### Try These Sample Articles")
    
    samples = get_sample_articles()
    
    for i, sample in enumerate(samples):
        with st.expander(f"📰 {sample['title']}", expanded=(i == 0)):
            st.text_area(
                "Article content:",
                sample['content'],
                height=200,
                disabled=True,
                key=f"sample_{i}"
            )
            
            col1, col2 = st.columns(2)
            
            with col1:
                sample_length = st.selectbox(
                    "Summary length:",
                    ["short", "medium", "long"],
                    index=1,
                    key=f"length_{i}"
                )
            
            with col2:
                if st.button(f"Summarize This Article", key=f"summarize_{i}"):
                    with st.spinner("Generating summary..."):
                        summary = st.session_state.summarizer.summarize(
                            sample['content'], 
                            sample_length
                        )
                    
                    st.markdown("**Summary:**")
                    st.success(summary)
            
            if st.button(f"Use This Text in Main Tab", key=f"use_{i}"):
                st.session_state.text_input = sample['content']
                st.success("Text loaded! Go to the Text Summarizer tab.")
                st.balloons()

# Footer
st.markdown("---")
st.markdown("💡 **Tips for best results:**")
st.markdown("- Use articles with at least 50 words")
st.markdown("- The summarizer finds the most important sentences")
st.markdown("- Try different length settings for different needs")