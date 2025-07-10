"""
File handling utilities for the text summarizer
"""

import pandas as pd
import io
from typing import List, Dict, Optional

def read_csv_file(uploaded_file) -> Optional[pd.DataFrame]:
    """Read and validate CSV file"""
    try:
        df = pd.read_csv(uploaded_file)
        return df
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return None

def get_text_columns(df: pd.DataFrame) -> List[str]:
    """Get columns that likely contain text"""
    text_columns = []
    for col in df.columns:
        # Check if column contains text (not just numbers)
        sample_values = df[col].dropna().head(5)
        if sample_values.empty:
            continue
        
        # Check if most values are strings with reasonable length
        text_count = 0
        for val in sample_values:
            if isinstance(val, str) and len(val) > 20:
                text_count += 1
        
        if text_count >= len(sample_values) * 0.6:  # 60% are text
            text_columns.append(col)
    
    return text_columns if text_columns else list(df.columns)

def process_batch_summaries(df: pd.DataFrame, text_column: str, 
                          summarizer, max_rows: int = 10) -> pd.DataFrame:
    """Process multiple texts for summarization"""
    # Limit rows for performance
    df_subset = df.head(max_rows).copy()
    
    summaries = []
    for i, row in df_subset.iterrows():
        text = str(row[text_column])
        if pd.isna(text) or len(text.strip()) < 20:
            summaries.append("Text too short to summarize")
        else:
            summary = summarizer.summarize(text)
            summaries.append(summary)
    
    df_subset['summary'] = summaries
    return df_subset

def create_download_data(df: pd.DataFrame, text_column: str) -> str:
    """Create CSV data for download"""
    buffer = io.StringIO()
    df[[text_column, 'summary']].to_csv(buffer, index=False)
    return buffer.getvalue()

def get_sample_data() -> List[Dict[str, str]]:
    """Get sample articles for testing"""
    return [
        {
            "title": "Climate Change Impact",
            "content": "Climate change is causing significant environmental changes worldwide. Rising temperatures are melting polar ice caps and glaciers at unprecedented rates. Sea levels are rising, threatening coastal communities and ecosystems. Extreme weather events like hurricanes, droughts, and floods are becoming more frequent and severe. Scientists warn that without immediate action to reduce greenhouse gas emissions, these effects will worsen. Renewable energy sources like solar and wind power offer promising solutions. Governments and businesses must work together to implement sustainable practices and reduce carbon footprints."
        },
        {
            "title": "AI in Healthcare",
            "content": "Artificial intelligence is revolutionizing healthcare by improving diagnosis accuracy and treatment efficiency. Machine learning algorithms can analyze medical images to detect diseases earlier than traditional methods. AI-powered systems help doctors make better decisions by processing vast amounts of patient data. Drug discovery is accelerated through AI models that predict molecular interactions. Robotic surgery systems provide greater precision and reduce recovery times. However, challenges remain in ensuring data privacy and algorithm fairness. Healthcare professionals need training to work effectively with AI tools."
        },
        {
            "title": "Remote Work Trends",
            "content": "Remote work has become a permanent fixture in the modern workplace following the global pandemic. Companies have discovered that many employees are more productive when working from home. This shift has led to cost savings on office space and improved work-life balance for workers. Digital collaboration tools have evolved to support distributed teams effectively. However, challenges include maintaining company culture and ensuring effective communication. Hybrid work models combining remote and in-office work are emerging as a popular compromise. The future of work will likely involve greater flexibility and technology integration."
        }
    ]