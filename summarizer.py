"""
Abstractive Text Summarizer using Sequence-to-Sequence Transformers
This module implements a proper abstractive summarization system using transformer models.
"""

import os
import re
import time
from typing import Dict, List, Optional, Tuple

class AbstractiveSummarizer:
    """
    Abstractive summarization using transformer models like T5, BART, or Pegasus.
    These models generate new text rather than just extracting existing sentences.
    """
    
    def __init__(self, model_name: str = "t5-small"):
        """
        Initialize the summarizer with a specific model.
        
        Args:
            model_name: Name of the transformer model to use
                       Options: t5-small, t5-base, facebook/bart-large-cnn, 
                               google/pegasus-xsum
        """
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        self.device = "cpu"  # Use CPU for deployment compatibility
        self.max_input_length = 512  # Maximum input tokens
        self.is_loaded = False
        
        # Try to load the model
        self._load_model()
    
    def _load_model(self):
        """Load the transformer model and tokenizer."""
        try:
            # Import transformers library
            from transformers import (
                T5ForConditionalGeneration, T5Tokenizer,
                BartForConditionalGeneration, BartTokenizer,
                PegasusForConditionalGeneration, PegasusTokenizer
            )
            
            print(f"Loading {self.model_name} model...")
            
            # Load different models based on model name
            if "t5" in self.model_name.lower():
                self.model = T5ForConditionalGeneration.from_pretrained(self.model_name)
                self.tokenizer = T5Tokenizer.from_pretrained(self.model_name)
                self.prefix = "summarize: "  # T5 needs task prefix
                
            elif "bart" in self.model_name.lower():
                self.model = BartForConditionalGeneration.from_pretrained(self.model_name)
                self.tokenizer = BartTokenizer.from_pretrained(self.model_name)
                self.prefix = ""  # BART doesn't need prefix
                
            elif "pegasus" in self.model_name.lower():
                self.model = PegasusForConditionalGeneration.from_pretrained(self.model_name)
                self.tokenizer = PegasusTokenizer.from_pretrained(self.model_name)
                self.prefix = ""  # Pegasus doesn't need prefix
            
            else:
                raise ValueError(f"Unsupported model: {self.model_name}")
            
            # Move model to CPU for deployment
            self.model.to(self.device)
            self.model.eval()  # Set to evaluation mode
            
            self.is_loaded = True
            print(f"✅ {self.model_name} loaded successfully!")
            
        except ImportError:
            print("❌ Transformers library not available")
            self.is_loaded = False
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            self.is_loaded = False
    
    def preprocess_text(self, text: str) -> str:
        """
        Clean and preprocess text for better summarization.
        
        Args:
            text: Raw input text
            
        Returns:
            Cleaned text ready for summarization
        """
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters that might confuse the model
        text = re.sub(r'[^\w\s\.\!\?\,\;\:\-\(\)\"\']+', '', text)
        
        # Ensure text ends with proper punctuation
        if not text.endswith(('.', '!', '?')):
            text += '.'
        
        return text.strip()
    
    def chunk_text(self, text: str, max_length: int = 400) -> List[str]:
        """
        Split long text into chunks that fit within model limits.
        
        Args:
            text: Input text to chunk
            max_length: Maximum tokens per chunk
            
        Returns:
            List of text chunks
        """
        if not self.tokenizer:
            # Fallback: split by sentences
            sentences = re.split(r'(?<=[.!?])\s+', text)
            chunks = []
            current_chunk = ""
            
            for sentence in sentences:
                if len(current_chunk.split()) + len(sentence.split()) <= max_length:
                    current_chunk += sentence + " "
                else:
                    if current_chunk:
                        chunks.append(current_chunk.strip())
                    current_chunk = sentence + " "
            
            if current_chunk:
                chunks.append(current_chunk.strip())
            
            return chunks
        
        # Use tokenizer for accurate chunking
        tokens = self.tokenizer.encode(text, return_tensors="pt")
        
        if tokens.shape[1] <= max_length:
            return [text]
        
        # Split into chunks
        chunks = []
        for i in range(0, tokens.shape[1], max_length):
            chunk_tokens = tokens[:, i:i + max_length]
            chunk_text = self.tokenizer.decode(chunk_tokens[0], skip_special_tokens=True)
            chunks.append(chunk_text)
        
        return chunks
    
    def generate_summary(self, text: str, length: str = "medium") -> str:
        """
        Generate abstractive summary using the transformer model.
        
        Args:
            text: Input text to summarize
            length: Summary length - "short", "medium", or "long"
            
        Returns:
            Generated summary text
        """
        if not self.is_loaded:
            return self._fallback_summary(text, length)
        
        try:
            # Preprocess text
            clean_text = self.preprocess_text(text)
            
            # Add task prefix for T5 models
            input_text = self.prefix + clean_text
            
            # Set generation parameters based on length
            length_params = self._get_length_params(length)
            
            # Tokenize input
            inputs = self.tokenizer.encode(
                input_text,
                return_tensors="pt",
                max_length=self.max_input_length,
                truncation=True
            )
            
            # Generate summary
            with torch.no_grad():  # Disable gradient computation for inference
                summary_ids = self.model.generate(
                    inputs,
                    max_length=length_params["max_length"],
                    min_length=length_params["min_length"],
                    num_beams=4,  # Beam search for better quality
                    early_stopping=True,
                    no_repeat_ngram_size=2,  # Avoid repetition
                    temperature=0.7,  # Control randomness
                    do_sample=True  # Enable sampling
                )
            
            # Decode the generated summary
            summary = self.tokenizer.decode(
                summary_ids[0], 
                skip_special_tokens=True
            )
            
            # Post-process summary
            return self._post_process_summary(summary)
            
        except Exception as e:
            print(f"Error generating summary: {e}")
            return self._fallback_summary(text, length)
    
    def _get_length_params(self, length: str) -> Dict[str, int]:
        """Get generation parameters based on desired length."""
        params = {
            "short": {"max_length": 50, "min_length": 20},
            "medium": {"max_length": 100, "min_length": 40},
            "long": {"max_length": 150, "min_length": 60}
        }
        return params.get(length, params["medium"])
    
    def _post_process_summary(self, summary: str) -> str:
        """Clean up the generated summary."""
        # Remove any remaining task prefixes
        if summary.startswith("summarize:"):
            summary = summary.replace("summarize:", "").strip()
        
        # Ensure proper capitalization
        if summary:
            summary = summary[0].upper() + summary[1:]
        
        # Ensure proper ending
        if summary and not summary.endswith(('.', '!', '?')):
            summary += '.'
        
        return summary
    
    def _fallback_summary(self, text: str, length: str) -> str:
        """Simple extractive fallback when model isn't available."""
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        if length == "short":
            return sentences[0] if sentences else text
        elif length == "long":
            return '. '.join(sentences[:3]) + '.' if len(sentences) >= 3 else text
        else:  # medium
            return '. '.join(sentences[:2]) + '.' if len(sentences) >= 2 else text
    
    def summarize_long_text(self, text: str, length: str = "medium") -> str:
        """
        Summarize very long text by chunking and combining summaries.
        
        Args:
            text: Long input text
            length: Desired summary length
            
        Returns:
            Combined summary of all chunks
        """
        if not self.is_loaded:
            return self._fallback_summary(text, length)
        
        # Split text into manageable chunks
        chunks = self.chunk_text(text, max_length=400)
        
        if len(chunks) == 1:
            return self.generate_summary(chunks[0], length)
        
        # Summarize each chunk
        chunk_summaries = []
        for chunk in chunks:
            summary = self.generate_summary(chunk, "short")
            chunk_summaries.append(summary)
        
        # Combine chunk summaries
        combined_text = " ".join(chunk_summaries)
        
        # Generate final summary
        return self.generate_summary(combined_text, length)
    
    def get_model_info(self) -> Dict[str, str]:
        """Get information about the current model."""
        return {
            "model_name": self.model_name,
            "model_type": "Abstractive Transformer",
            "is_loaded": self.is_loaded,
            "device": self.device,
            "max_input_length": self.max_input_length
        }

# Import torch with fallback
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("PyTorch not available - some features may be limited")

def create_summarizer(model_name: str = "t5-small") -> AbstractiveSummarizer:
    """
    Factory function to create a summarizer instance.
    
    Args:
        model_name: Name of the transformer model to use
        
    Returns:
        AbstractiveSummarizer instance
    """
    if not TORCH_AVAILABLE:
        print("Warning: PyTorch not available. Summarizer will use fallback mode.")
    
    return AbstractiveSummarizer(model_name)