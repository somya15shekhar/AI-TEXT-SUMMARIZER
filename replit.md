# AI Text Summarizer

## Overview

The AI Text Summarizer is a web application built with Streamlit that transforms long articles into concise summaries using advanced AI models. The application provides multiple summarization options, batch processing capabilities, and real-time metrics tracking. It's designed to be user-friendly with a clean interface and efficient processing.

## User Preferences

Preferred communication style: Simple, everyday language.

## System Architecture

### Frontend Architecture
- **Framework**: Streamlit web framework for rapid development and deployment
- **UI Components**: Clean, responsive interface with custom CSS styling
- **State Management**: Streamlit session state for model caching and user data persistence
- **Layout**: Wide layout with collapsible sidebar for optimal user experience

### Backend Architecture
- **Core Engine**: Modular architecture with separate components for summarization, utilities, and sample data
- **Model Management**: Intelligent model loading with transformer-first approach and extractive fallback
- **Processing Pipeline**: Robust text validation, summarization, and metrics calculation
- **Caching Strategy**: Streamlit's `@st.cache_resource` decorator for efficient model loading

## Key Components

### 1. SummarizerEngine (`modules/summarizer.py`)
- **Primary Function**: Handles text summarization using DistilBART transformer model
- **Fallback Strategy**: Extractive summarization when transformer models are unavailable
- **Device Management**: CPU-optimized for Streamlit Cloud deployment
- **Error Handling**: Graceful degradation with informative error messages

### 2. Validation & Metrics (`modules/utils.py`)
- **Text Validation**: Ensures input text meets quality and length requirements
- **Metrics Calculation**: Computes word count, compression ratio, and processing time
- **Content Analysis**: Validates alphabetic content ratio to filter out low-quality text

### 3. Sample Data (`modules/sample_data.py`)
- **Demo Content**: Pre-loaded sample articles for testing functionality
- **Categories**: Environment, Technology, and Remote Work topics
- **Structure**: Title, category, source, and content fields for comprehensive testing

### 4. Main Application (`app.py`)
- **UI Configuration**: Page setup with custom styling and responsive design
- **Model Loading**: Resource-cached summarization engine initialization
- **User Interface**: Input handling, batch processing, and results display

## Data Flow

1. **Text Input**: User provides text via direct input or CSV upload
2. **Validation**: Text is validated for length, quality, and content requirements
3. **Model Selection**: System attempts to load DistilBART transformer model
4. **Summarization**: Text is processed using the best available method
5. **Metrics Calculation**: Processing statistics are computed and displayed
6. **Results Display**: Summary and metrics are presented with download options

## External Dependencies

### Core Dependencies
- **Streamlit**: Web application framework and deployment platform
- **Pandas**: Data manipulation for CSV processing and batch operations
- **PyTorch**: Deep learning framework for transformer model execution
- **Transformers**: Hugging Face library for pre-trained summarization models

### AI Model Dependencies
- **Primary Model**: `sshleifer/distilbart-cnn-12-6` (DistilBART transformer)
- **Fallback Method**: Custom extractive summarization algorithm
- **Model Loading**: Intelligent fallback system for various deployment environments

## Deployment Strategy

### Target Platform
- **Primary**: Streamlit Cloud for seamless deployment and scaling
- **Architecture**: Single-container deployment with CPU optimization
- **Resource Management**: Efficient model loading and memory usage

### Performance Considerations
- **Model Caching**: Streamlit resource caching prevents repeated model loading
- **CPU Optimization**: Forced CPU usage for consistent cloud deployment
- **Memory Management**: Lightweight DistilBART model for reduced resource usage

### Error Handling
- **Graceful Degradation**: Automatic fallback to extractive summarization
- **User Feedback**: Clear error messages and alternative suggestions
- **Robust Validation**: Comprehensive input validation prevents processing errors

### Scalability Features
- **Batch Processing**: CSV upload for multiple text summarization
- **Efficient Processing**: Optimized algorithms for faster processing times
- **Resource Monitoring**: Built-in metrics tracking for performance analysis