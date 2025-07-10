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

# Simple text summarizer class
class TextSummarizer:
    def __init__(self):
        self.name = "Enhanced Extractive Summarizer"
    
    def summarize(self, text, length="medium"):
        """Create summary by selecting the best sentences"""
        if not text or len(text.strip()) < 50:
            return "Text too short to summarize"
        
        # Split into sentences
        sentences = re.split(r'(?<=[.!?])\s+', text.strip())
        sentences = [s.strip() for s in sentences if len(s.strip()) > 10]
        
        if len(sentences) <= 2:
            return text
        
        # Set number of sentences based on length
        if length == "short":
            num_sentences = 2
        elif length == "long":
            num_sentences = min(5, len(sentences))
        else:  # medium
            num_sentences = min(3, len(sentences))
        
        # Score sentences
        scores = []
        word_freq = {}
        words = text.lower().split()
        
        # Count word frequency
        for word in words:
            if len(word) > 3:
                word_freq[word] = word_freq.get(word, 0) + 1
        
        # Score each sentence
        for i, sentence in enumerate(sentences):
            score = 0
            sent_words = sentence.lower().split()
            
            # Word frequency score
            for word in sent_words:
                if len(word) > 3:
                    score += word_freq.get(word, 0)
            
            # Position bonus (first sentences important)
            if i == 0:
                score *= 1.5
            elif i < len(sentences) * 0.3:
                score *= 1.2
            
            # Length bonus (not too short, not too long)
            if 8 <= len(sent_words) <= 25:
                score *= 1.1
            
            # Keyword bonus
            keywords = ['important', 'key', 'main', 'significant', 'research', 'study', 'shows']
            for keyword in keywords:
                if keyword in sentence.lower():
                    score *= 1.2
            
            scores.append((score / len(sent_words), i))
        
        # Get top sentences
        scores.sort(reverse=True)
        top_indices = sorted([idx for _, idx in scores[:num_sentences]])
        
        # Create summary
        summary = '. '.join([sentences[i] for i in top_indices])
        
        # Clean up
        summary = re.sub(r'\s+', ' ', summary).strip()
        if not summary.endswith('.'):
            summary += '.'
        
        return summary

# Sample articles
def get_samples():
    return [
        {
            "title": "Climate Change Solutions",
            "text": "Climate change is one of the biggest challenges we face today. Global temperatures are rising faster than ever before. Scientists agree that we need to act quickly to reduce greenhouse gas emissions. Renewable energy like solar and wind power has become much cheaper and more efficient. Many countries are setting goals to reach zero carbon emissions by 2050. The clean energy industry is creating millions of new jobs worldwide. However, we still face challenges like energy storage and updating power grids. Public support for clean energy is growing stronger every year. The next ten years will be crucial for avoiding the worst effects of climate change."
        },
        {
            "title": "AI in Healthcare",
            "text": "Artificial intelligence is changing healthcare in amazing ways. AI can now look at medical images and find diseases like cancer earlier than human doctors. Machine learning helps discover new medicines much faster than traditional methods. AI systems can read patient records and predict who might get sick. Robot surgeons help doctors perform operations more precisely. However, we need to protect patient privacy and make sure AI systems work fairly for everyone. Healthcare workers need training to use these new AI tools effectively. The future of medicine will combine human expertise with AI assistance."
        },
        {
            "title": "Remote Work Revolution",
            "text": "The pandemic changed how we work forever. Many companies discovered that employees can be just as productive working from home. Remote work gives people better work-life balance and eliminates commuting time. Companies can now hire talent from anywhere in the world. Digital tools like video calls and cloud computing make remote teamwork possible. Some people have become digital nomads, working while traveling to different countries. However, remote work can be lonely and it's harder to build company culture. Most companies are now using hybrid models that combine office and remote work."
        }
    ]

# Initialize summarizer
if 'summarizer' not in st.session_state:
    st.session_state.summarizer = TextSummarizer()

# Title
st.title("📝 AI Text Summarizer")
st.markdown("Turn long articles into short summaries instantly")

# Tabs
tab1, tab2, tab3 = st.tabs(["✏️ Summarize Text", "📊 Batch Process", "📰 Try Samples"])

# Tab 1: Main summarizer
with tab1:
    col1, col2 = st.columns([3, 1])
    
    with col1:
        text_input = st.text_area(
            "Paste your text here:",
            height=300,
            placeholder="Paste your article, blog post, or any text you want to summarize...",
            key="main_text"
        )
        
        if text_input:
            words = len(text_input.split())
            chars = len(text_input)
            st.caption(f"📊 {words:,} words • {chars:,} characters")
    
    with col2:
        st.markdown("**Settings**")
        length = st.selectbox(
            "Summary length:",
            ["short", "medium", "long"],
            index=1
        )
        
        length_info = {
            "short": "2 sentences",
            "medium": "3 sentences", 
            "long": "4-5 sentences"
        }
        st.caption(f"📏 {length_info[length]}")
        
        st.markdown("**Model**")
        st.info("Enhanced Extractive")
    
    # Generate button
    if st.button("🚀 Generate Summary", type="primary", use_container_width=True):
        if not text_input:
            st.error("Please paste some text first")
        else:
            with st.spinner("Creating summary..."):
                start_time = time.time()
                summary = st.session_state.summarizer.summarize(text_input, length)
                end_time = time.time()
            
            st.success("Summary created!")
            
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
                st.metric("Saved", f"{compression}%")
            
            # Show summary
            st.markdown("### 📄 Your Summary")
            st.info(summary)
            
            # Download
            summary_text = f"Original Text:\n{text_input}\n\nSummary:\n{summary}"
            st.download_button(
                "📥 Download Summary",
                summary_text,
                file_name="summary.txt",
                mime="text/plain"
            )

# Tab 2: Batch processing
with tab2:
    st.markdown("### Upload CSV File")
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file)
            st.success("File uploaded!")
            st.dataframe(df.head())
            
            col1, col2 = st.columns(2)
            with col1:
                text_column = st.selectbox("Text column:", df.columns)
            with col2:
                batch_length = st.selectbox("Summary length:", ["short", "medium", "long"], index=1)
            
            rows_to_process = st.slider("Number of rows:", 1, min(20, len(df)), 5)
            
            if st.button("Process Articles"):
                progress_bar = st.progress(0)
                summaries = []
                
                for i in range(rows_to_process):
                    progress_bar.progress((i + 1) / rows_to_process)
                    text = str(df.iloc[i][text_column])
                    summary = st.session_state.summarizer.summarize(text, batch_length)
                    summaries.append(summary)
                
                df_results = df.head(rows_to_process).copy()
                df_results['summary'] = summaries
                
                st.success("Processing complete!")
                st.dataframe(df_results[[text_column, 'summary']])
                
                # Download results
                csv_buffer = io.StringIO()
                df_results.to_csv(csv_buffer, index=False)
                st.download_button(
                    "📥 Download Results",
                    csv_buffer.getvalue(),
                    file_name="summaries.csv",
                    mime="text/csv"
                )
        
        except Exception as e:
            st.error(f"Error: {str(e)}")

# Tab 3: Sample articles
with tab3:
    st.markdown("### Try These Sample Articles")
    
    samples = get_samples()
    
    for i, sample in enumerate(samples):
        with st.expander(f"📰 {sample['title']}", expanded=(i == 0)):
            st.text_area(
                "Article text:",
                sample['text'],
                height=150,
                key=f"sample_text_{i}"
            )
            
            col1, col2 = st.columns(2)
            with col1:
                sample_length = st.selectbox(
                    "Length:",
                    ["short", "medium", "long"],
                    index=1,
                    key=f"sample_length_{i}"
                )
            
            with col2:
                if st.button(f"Summarize", key=f"sample_btn_{i}"):
                    with st.spinner("Summarizing..."):
                        summary = st.session_state.summarizer.summarize(
                            sample['text'], 
                            sample_length
                        )
                    
                    st.markdown("**Summary:**")
                    st.success(summary)
                    
                    # Use this sample
                    if st.button(f"Use This Text", key=f"use_sample_{i}"):
                        st.session_state.main_text = sample['text']
                        st.success("Text copied to main tab!")
                        st.balloons()

# Footer
st.markdown("---")
st.markdown("💡 **Tips:** For best results, use articles with at least 100 words. The summarizer works by finding the most important sentences in your text.")