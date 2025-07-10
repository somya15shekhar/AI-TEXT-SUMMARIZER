import os
import re
from typing import Dict, Any, Optional

# Try to import torch and transformers with fallback
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("⚠️ PyTorch not available - using fallback methods")

try:
    from transformers import pipeline
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("⚠️ Transformers not available - using fallback methods")

class SummarizerEngine:
    """
    Optimized text summarization engine with transformer and extractive fallback
    """
    
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.model_type = "extractive"  # Default fallback
        self._load_model()
    
    def _load_model(self):
        """Load the best available summarization model"""
        # For now, we'll use the extractive method as the primary method
        # This ensures the app works without requiring heavy dependencies
        self.model_type = "extractive"
        print("✅ Using enhanced extractive summarization method")
        
        # Future: Add transformer support when dependencies are available
        # if TORCH_AVAILABLE and TRANSFORMERS_AVAILABLE:
        #     try:
        #         self.model = pipeline(
        #             "summarization",
        #             model="sshleifer/distilbart-cnn-12-6",
        #             torch_dtype=torch.float32,
        #             device_map="cpu"
        #         )
        #         self.model_type = "transformer"
        #         print("✅ DistilBART model loaded successfully")
        #     except Exception as e:
        #         print(f"⚠️ Could not load transformer model: {e}")
        #         self.model_type = "extractive"
    
    def summarize(self, text: str, length: str = "medium") -> str:
        """
        Generate summary using the best available method
        
        Args:
            text: Input text to summarize
            length: Summary length ("short", "medium", "long")
            
        Returns:
            Generated summary text
        """
        if not text or not text.strip():
            return "No text provided for summarization."
        
        # Clean input text
        text = self._clean_text(text)
        
        if self.model_type == "transformer":
            return self._summarize_transformer(text, length)
        else:
            return self._summarize_extractive(text, length)
    
    def _clean_text(self, text: str) -> str:
        """Clean and preprocess text"""
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove special characters that might cause issues
        text = re.sub(r'[^\w\s\.\!\?\,\;\:\-\(\)\"\']+', '', text)
        return text.strip()
    
    def _summarize_transformer(self, text: str, length: str) -> str:
        """Generate summary using transformer model"""
        if not TORCH_AVAILABLE or not TRANSFORMERS_AVAILABLE:
            return self._summarize_extractive(text, length)
        
        # Length configurations optimized for DistilBART
        length_configs = {
            "short": {"max_length": 60, "min_length": 20},
            "medium": {"max_length": 120, "min_length": 40},
            "long": {"max_length": 180, "min_length": 60}
        }
        
        config = length_configs.get(length, length_configs["medium"])
        
        try:
            # Truncate if too long (DistilBART max input is ~1024 tokens)
            max_input_tokens = 900  # Leave some buffer
            words = text.split()
            if len(words) > max_input_tokens:
                text = ' '.join(words[:max_input_tokens])
            
            # Generate summary
            result = self.model(
                text,
                max_length=config["max_length"],
                min_length=config["min_length"],
                do_sample=False,
                truncation=True,
                clean_up_tokenization_spaces=True
            )
            
            summary = result[0]['summary_text']
            
            # Post-process summary
            summary = self._post_process_summary(summary)
            
            return summary
            
        except Exception as e:
            print(f"Transformer error: {e}")
            # Fallback to extractive method
            return self._summarize_extractive(text, length)
    
    def _summarize_extractive(self, text: str, length: str) -> str:
        """Enhanced extractive summarization using advanced scoring"""
        # Length configurations for extractive method
        length_configs = {
            "short": {"sentences": 2, "max_words": 80},
            "medium": {"sentences": 4, "max_words": 150},
            "long": {"sentences": 6, "max_words": 250}
        }
        
        config = length_configs.get(length, length_configs["medium"])
        
        # Split into sentences
        sentences = self._split_sentences(text)
        
        if len(sentences) <= config["sentences"]:
            # If text is already short enough, return as-is
            return '. '.join(sentences)
        
        # Score sentences using multiple methods
        sentence_scores = self._score_sentences_advanced(sentences, text)
        
        # Select top sentences
        top_indices = sorted(
            sentence_scores.keys(),
            key=lambda x: sentence_scores[x],
            reverse=True
        )[:config["sentences"]]
        
        # Maintain original order
        top_indices.sort()
        
        # Create summary
        summary_sentences = [sentences[i] for i in top_indices]
        summary = '. '.join(summary_sentences)
        
        # Trim if too long
        words = summary.split()
        if len(words) > config["max_words"]:
            summary = ' '.join(words[:config["max_words"]]) + '...'
        
        return self._post_process_summary(summary)
    
    def _split_sentences(self, text: str) -> list:
        """Split text into sentences"""
        # Simple sentence splitting
        sentences = re.split(r'(?<=[.!?])\s+', text)
        sentences = [s.strip() for s in sentences if s.strip() and len(s.split()) >= 3]
        return sentences
    
    def _score_sentences_advanced(self, sentences: list, full_text: str) -> dict:
        """Advanced sentence scoring combining multiple techniques"""
        scores = {}
        
        # Word frequency calculation
        words = full_text.lower().split()
        word_freq = {}
        for word in words:
            if len(word) > 3:  # Skip short words
                word_freq[word] = word_freq.get(word, 0) + 1
        
        # Calculate TF-IDF-like scores
        total_sentences = len(sentences)
        
        # Score each sentence
        for i, sentence in enumerate(sentences):
            sent_words = sentence.lower().split()
            
            # 1. Frequency-based scoring
            freq_score = 0
            for word in sent_words:
                if len(word) > 3:
                    freq_score += word_freq.get(word, 0)
            
            # 2. Position bonus (first and last sentences often important)
            position_score = 1.0
            if i < total_sentences * 0.3:  # First 30%
                position_score = 1.4
            elif i > total_sentences * 0.7:  # Last 30%
                position_score = 1.2
            
            # 3. Length penalty/bonus (avoid too short or too long sentences)
            length_score = 1.0
            if 10 <= len(sent_words) <= 25:  # Ideal length
                length_score = 1.3
            elif len(sent_words) < 5:  # Too short
                length_score = 0.5
            elif len(sent_words) > 40:  # Too long
                length_score = 0.8
            
            # 4. Keyword bonus
            keyword_score = 1.0
            important_keywords = [
                'important', 'significant', 'key', 'main', 'primary', 'essential',
                'critical', 'major', 'fundamental', 'crucial', 'vital', 'central',
                'therefore', 'however', 'moreover', 'furthermore', 'consequently',
                'research', 'study', 'found', 'shows', 'indicates', 'reveals',
                'according', 'report', 'analysis', 'conclusion', 'result'
            ]
            for keyword in important_keywords:
                if keyword in sentence.lower():
                    keyword_score *= 1.15
            
            # 5. Numeric data bonus (statistics, dates, percentages)
            numeric_score = 1.0
            if any(char.isdigit() for char in sentence):
                numeric_score = 1.2
            
            # 6. Question/statement detection
            statement_score = 1.0
            if sentence.strip().endswith('?'):
                statement_score = 0.8  # Questions less likely to be key points
            elif sentence.strip().endswith('.') or sentence.strip().endswith('!'):
                statement_score = 1.1  # Complete statements preferred
            
            # Combine all scores
            final_score = (freq_score / max(len(sent_words), 1)) * position_score * length_score * keyword_score * numeric_score * statement_score
            
            scores[i] = final_score
        
        return scores
    
    def _score_sentences(self, sentences: list, full_text: str) -> dict:
        """Legacy scoring method (kept for compatibility)"""
        return self._score_sentences_advanced(sentences, full_text)
    
    def _post_process_summary(self, summary: str) -> str:
        """Post-process summary to improve quality"""
        # Fix common issues
        summary = re.sub(r'\s+', ' ', summary)  # Multiple spaces
        summary = re.sub(r'\.+', '.', summary)  # Multiple periods
        summary = summary.strip()
        
        # Ensure proper ending
        if summary and not summary.endswith(('.', '!', '?')):
            summary += '.'
        
        # Capitalize first letter
        if summary:
            summary = summary[0].upper() + summary[1:] if len(summary) > 1 else summary.upper()
        
        return summary
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model"""
        return {
            "model_type": self.model_type,
            "model_name": "DistilBART-CNN-12-6" if self.model_type == "transformer" else "Extractive",
            "description": "Lightweight transformer model optimized for news summarization" if self.model_type == "transformer" else "Rule-based extractive summarization",
            "max_input_length": 1024 if self.model_type == "transformer" else 10000
        }
