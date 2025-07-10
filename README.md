# 🤖 AI Text Summarizer

A fast, efficient text summarization web application built with Streamlit and advanced AI models. Transform long articles into concise summaries using state-of-the-art transformer models.

## ✨ Features

- **📊 Batch Processing**: Upload CSV files to summarize multiple articles at once
- **🎯 Sample Articles**: Test the app with pre-loaded sample articles
- **📈 Real-time Metrics**: Track word count, compression ratio, and processing time
- **💾 Download Results**: Export summaries as text files or CSV
- **🧠 Advanced AI Models**: Uses DistilBART transformer model with extractive fallback

## 🎮 Try It Live

https://appsimplepy-bmop9b7aqi83wa5qtgk9d4.streamlit.app/

NOTE: Due to free resource limits on Streamlit Cloud, the transformer-based abstractive summarizer is replaced with an extractive summarization fallback. It selects key sentences from the input text to generate concise summaries efficiently.

## 🛠️ Installation

1. **Clone the repository**
```bash
git clone https://github.com/somya15shekhar/text_summarizer_project.git
cd your repo
pip install -r requirements.txt
streamlit run app.py

