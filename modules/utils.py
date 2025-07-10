import re
from typing import Dict, Any

def validate_text(text: str) -> Dict[str, Any]:
    """
    Validate input text for summarization
    
    Args:
        text: Input text to validate
        
    Returns:
        Dictionary with validation results
    """
    if not text or not text.strip():
        return {"valid": False, "message": "Text cannot be empty"}
    
    # Clean text for analysis
    clean_text = re.sub(r'\s+', ' ', text.strip())
    words = clean_text.split()
    
    if len(words) < 10:
        return {"valid": False, "message": "Text is too short. Please provide at least 10 words."}
    
    if len(words) > 10000:
        return {"valid": False, "message": "Text is too long. Please provide less than 10,000 words."}
    
    # Check for mostly non-alphabetic content
    alpha_ratio = sum(1 for c in clean_text if c.isalpha()) / len(clean_text)
    if alpha_ratio < 0.5:
        return {"valid": False, "message": "Text contains too many non-alphabetic characters."}
    
    return {"valid": True, "message": "Text is valid for summarization"}

def calculate_metrics(original_text: str, summary_text: str) -> Dict[str, Any]:
    """
    Calculate summarization metrics
    
    Args:
        original_text: Original input text
        summary_text: Generated summary
        
    Returns:
        Dictionary with calculated metrics
    """
    # Word counts
    original_words = len(original_text.split())
    summary_words = len(summary_text.split())
    
    # Character counts
    original_chars = len(original_text)
    summary_chars = len(summary_text)
    
    # Compression ratios
    word_compression = ((original_words - summary_words) / original_words) * 100 if original_words > 0 else 0
    char_compression = ((original_chars - summary_chars) / original_chars) * 100 if original_chars > 0 else 0
    
    # Readability metrics
    sentences_original = len(re.split(r'[.!?]+', original_text))
    sentences_summary = len(re.split(r'[.!?]+', summary_text))
    
    return {
        "original_words": original_words,
        "summary_words": summary_words,
        "original_chars": original_chars,
        "summary_chars": summary_chars,
        "compression_ratio": round(word_compression, 1),
        "char_compression_ratio": round(char_compression, 1),
        "sentences_original": sentences_original,
        "sentences_summary": sentences_summary,
        "avg_words_per_sentence_original": round(original_words / max(sentences_original, 1), 1),
        "avg_words_per_sentence_summary": round(summary_words / max(sentences_summary, 1), 1)
    }

def format_time(seconds: float) -> str:
    """
    Format processing time in a human-readable way
    
    Args:
        seconds: Time in seconds
        
    Returns:
        Formatted time string
    """
    if seconds < 1:
        return f"{seconds*1000:.0f}ms"
    elif seconds < 60:
        return f"{seconds:.1f}s"
    else:
        minutes = int(seconds // 60)
        secs = seconds % 60
        return f"{minutes}m {secs:.1f}s"

def clean_csv_text(text: str) -> str:
    """
    Clean text from CSV files for better processing
    
    Args:
        text: Raw text from CSV
        
    Returns:
        Cleaned text
    """
    if not isinstance(text, str):
        return str(text)
    
    # Remove HTML tags if present
    text = re.sub(r'<[^>]+>', '', text)
    
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Remove special characters that might cause issues
    text = re.sub(r'[^\w\s\.\!\?\,\;\:\-\(\)\"\']+', '', text)
    
    return text.strip()

def truncate_text(text: str, max_chars: int = 1000) -> str:
    """
    Truncate text to a maximum number of characters
    
    Args:
        text: Input text
        max_chars: Maximum characters to keep
        
    Returns:
        Truncated text
    """
    if len(text) <= max_chars:
        return text
    
    # Try to cut at a sentence boundary
    truncated = text[:max_chars]
    last_sentence_end = max(
        truncated.rfind('.'),
        truncated.rfind('!'),
        truncated.rfind('?')
    )
    
    if last_sentence_end > max_chars * 0.7:  # If we can keep at least 70% of content
        return truncated[:last_sentence_end + 1]
    else:
        return truncated + "..."

def extract_keywords(text: str, max_keywords: int = 10) -> list:
    """
    Extract important keywords from text
    
    Args:
        text: Input text
        max_keywords: Maximum number of keywords to return
        
    Returns:
        List of keywords
    """
    # Simple keyword extraction
    words = re.findall(r'\b[a-zA-Z]{4,}\b', text.lower())
    
    # Common stop words to exclude
    stop_words = {
        'that', 'this', 'with', 'from', 'they', 'been', 'have', 'will', 'would', 
        'could', 'should', 'their', 'there', 'where', 'when', 'what', 'which',
        'more', 'most', 'some', 'such', 'very', 'well', 'also', 'just', 'like',
        'than', 'only', 'over', 'after', 'before', 'between', 'through', 'during'
    }
    
    # Count word frequencies
    word_freq = {}
    for word in words:
        if word not in stop_words:
            word_freq[word] = word_freq.get(word, 0) + 1
    
    # Sort by frequency and return top keywords
    keywords = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
    return [word for word, freq in keywords[:max_keywords]]
