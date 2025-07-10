"""
Simple Text Summarizer using lightweight transformer
"""

import regex as re
from typing import Optional

class TextSummarizer:
    """Simple text summarizer with transformer fallback"""
    
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.model_loaded = False
        self._try_load_model()
    
    def _try_load_model(self):
        """Try to load a lightweight transformer model"""
        try:
            from transformers import pipeline
            # Use a small, fast model that works well for summarization
            self.model = pipeline(
                "summarization",
                model="distilbart-cnn-12-6",
                device=-1  # Use CPU
            )
            self.model_loaded = True
            print("✅ BART model loaded successfully")
        except:
            print("⚠️ Using extractive summarization fallback")
            self.model_loaded = False
    
    def summarize(self, text: str, max_length: int = 150) -> str:
        """Generate summary from text"""
        if not text or len(text.strip()) < 50:
            return "Text is too short to summarize."
        
        # Clean text
        text = self._clean_text(text)
        
        if self.model_loaded:
            return self._transformer_summary(text, max_length)
        else:
            return self._extractive_summary(text, max_length)
    
    def _clean_text(self, text: str) -> str:
        """Clean input text"""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove problematic characters
        text = re.sub(r'[^\w\s\.\!\?\,\;\:\-\(\)\"\']+', '', text)
        return text.strip()
    
    def _transformer_summary(self, text: str, max_length: int) -> str:
        """Generate summary using transformer model"""
        try:
            # Limit input length
            max_input = 1000  # tokens
            words = text.split()
            if len(words) > max_input:
                text = ' '.join(words[:max_input])
            
            # Generate summary
            result = self.model(
                text,
                max_length=max_length,
                min_length=30,
                do_sample=False,
                truncation=True
            )
            
            summary = result[0]['summary_text']
            return self._clean_summary(summary)
        
        except Exception as e:
            print(f"Transformer error: {e}")
            return self._extractive_summary(text, max_length)
    
    def _extractive_summary(self, text: str, max_length: int) -> str:
        """Simple extractive summarization"""
        sentences = re.split(r'(?<=[.!?])\s+', text)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 10]
        
        if len(sentences) <= 2:
            return text
        
        # Score sentences
        word_freq = {}
        for word in text.lower().split():
            if len(word) > 3:
                word_freq[word] = word_freq.get(word, 0) + 1
        
        sentence_scores = []
        for i, sentence in enumerate(sentences):
            words = sentence.lower().split()
            score = sum(word_freq.get(word, 0) for word in words if len(word) > 3)
            
            # Boost first sentences
            if i == 0:
                score *= 1.5
            elif i < len(sentences) * 0.3:
                score *= 1.2
            
            sentence_scores.append((score / len(words), i))
        
        # Select top sentences
        sentence_scores.sort(reverse=True)
        
        # Determine number of sentences based on max_length
        if max_length <= 100:
            num_sentences = 2
        elif max_length <= 200:
            num_sentences = 3
        else:
            num_sentences = 4
        
        selected = sorted([idx for _, idx in sentence_scores[:num_sentences]])
        summary = '. '.join([sentences[i] for i in selected])
        
        return self._clean_summary(summary)
    
    def _clean_summary(self, summary: str) -> str:
        """Clean up summary"""
        summary = re.sub(r'\s+', ' ', summary).strip()
        if summary and not summary.endswith('.'):
            summary += '.'
        if summary:
            summary = summary[0].upper() + summary[1:]
        return summary
    
    def get_status(self) -> str:
        """Get current model status"""
        if self.model_loaded:
            return "DistilBART Transformer Model"
        else:
            return "Extractive Summarization"