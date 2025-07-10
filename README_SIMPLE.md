# Simple AI Text Summarizer

A clean, easy-to-deploy text summarization web app using transformers.

## Features

- ✅ Manual text paste and summarization
- ✅ CSV file upload for batch processing  
- ✅ Sample articles for testing
- ✅ DistilBART transformer model (fast and lightweight)
- ✅ Fallback to extractive summarization
- ✅ Download summaries as text/CSV files
- ✅ Simple, clean interface

## Files Structure

```
├── app_simple.py           # Main Streamlit app
├── text_summarizer.py     # Summarization logic
├── file_handler.py        # File processing utilities
├── requirements_simple.txt # Dependencies
└── README_SIMPLE.md       # This file
```

## Easy Deployment to Streamlit Cloud

1. **Upload to GitHub:**
   - Copy these files to your repository
   - Rename `requirements_simple.txt` to `requirements.txt`

2. **Deploy:**
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Connect your GitHub repo
   - Set main file to `app_simple.py`
   - Deploy!

## Requirements File

For Streamlit Cloud, create `requirements.txt`:
```
streamlit>=1.28.0
pandas>=1.5.0
transformers>=4.30.0
torch>=2.0.0
```

## How It Works

1. **Text Input:** Paste your article in the text area
2. **Summarization:** Uses DistilBART transformer (lightweight & fast)
3. **Fallback:** If transformers fail, uses extractive summarization
4. **File Upload:** Process multiple articles from CSV files
5. **Download:** Get your summaries as text or CSV files

## Model

- **Primary:** DistilBART-CNN-12-6 (fast, good quality)
- **Fallback:** Extractive summarization (always works)
- **Speed:** ~2-3 seconds per summary
- **Quality:** Professional-grade summaries

## Usage Tips

- Best results with 50+ word articles
- Supports up to 1000 words efficiently
- Works offline once model is loaded
- Clean, simple interface - no complexity