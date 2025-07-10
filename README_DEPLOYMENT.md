# AI Text Summarizer - Streamlit Deployment Guide

## Quick Start

This is a simple text summarization app built with Streamlit. It works without any heavy AI dependencies.

## Files for Deployment

### Option 1: Simple Version (Recommended)
- **main.py** - Clean, simple version of the app
- **requirements_deployment.txt** - Rename this to `requirements.txt` for deployment

### Option 2: Full Version
- **app.py** - Full featured version with modules
- **modules/** - Supporting files

## Deployment on Streamlit Cloud

1. **Upload to GitHub:**
   - Create a new repository
   - Upload either `main.py` or `app.py` as your main file
   - Upload `requirements_deployment.txt` and rename it to `requirements.txt`

2. **Deploy on Streamlit Cloud:**
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Connect your GitHub repository
   - Set main file path to `main.py` or `app.py`
   - Click Deploy

## Requirements File

For Streamlit Cloud deployment, use this `requirements.txt`:

```
streamlit>=1.28.0
pandas>=1.5.0
numpy>=1.24.0
```

## Features

- ✅ Manual text paste and summarization
- ✅ Multiple summary lengths (short, medium, long)
- ✅ Batch processing from CSV files
- ✅ Sample articles to test
- ✅ Download summaries as text files
- ✅ Real-time word/character counting
- ✅ Works without heavy AI dependencies

## How It Works

The app uses an enhanced extractive summarization algorithm that:
1. Splits text into sentences
2. Scores sentences based on word frequency, position, and keywords
3. Selects the highest-scoring sentences
4. Creates a coherent summary

## Tips for Best Results

- Use articles with at least 50 words
- The summarizer works best with well-structured text
- Try different length settings for different needs

## No Complex Dependencies

This app is designed to deploy easily on Streamlit Cloud without requiring:
- PyTorch
- Transformers
- Heavy AI models
- Complex installations

Perfect for quick deployment and testing!