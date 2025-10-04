# ============================================================================
# MBG SENTIMENT ANALYSIS DASHBOARD - Main Application
# Multi-page Streamlit Application with Complete Functionality
# ============================================================================

import streamlit as st
import pandas as pd
import sys
from pathlib import Path

# Page configuration
st.set_page_config(
    page_title="MBG Sentiment Analysis Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f2937;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #6b7280;
        margin-bottom: 2rem;
    }
    .feature-box {
        background-color: #f3f4f6;
        padding: 1.5rem;
        border-radius: 0.5rem;
        border-left: 4px solid #3b82f6;
        margin-bottom: 1rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 0.5rem;
        color: white;
        text-align: center;
    }
    .stButton>button {
        width: 100%;
        background-color: #3b82f6;
        color: white;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
def init_session_state():
    """Initialize all session state variables"""
    defaults = {
        'df_original': None,
        'df_cleaned': None,
        'df_sentiment': None,
        'text_column': None,
        'normalization_dict': {},
        'veto_words': [],
        'trained_models': {},
        'best_model': None,
        'analysis_mode': 'Quick Mode',
        'preprocessing_options': {
            'use_stemming': True,
            'use_stopwords': True,
            'min_text_length': 15
        },
        'model_params': {
            'test_size': 0.2,
            'random_state': 42,
            'smote_k_neighbors': 5,
            'cv_folds': 3
        },
        'selected_models': ['SVM', 'KNN'],
        'bert_model_loaded': False,
        'sentiment_analyzer': None
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

init_session_state()

# Sidebar
with st.sidebar:
    st.image("https://via.placeholder.com/150x50/3b82f6/ffffff?text=MBG+Analytics", use_container_width=True)
    st.markdown("---")
    
    st.markdown("### 📌 Navigation")
    st.info("""
    Use the pages in the sidebar to navigate through the application:
    
    1. **Data Upload** - Load and preprocess data
    2. **Sentiment Analysis** - BERT-based labeling
    3. **Model Training** - Train ML classifiers
    4. **Visualizations** - Interactive charts
    5. **Error Analysis** - Model insights
    6. **Live Predictor** - Test new text
    7. **Export & Reports** - Download results
    """)
    
    st.markdown("---")
    st.markdown("### ⚙️ Global Settings")
    
    # Analysis mode
    st.session_state.analysis_mode = st.radio(
        "Analysis Mode",
        ["Quick Mode", "Full Analysis"],
        help="Quick Mode uses reduced parameters for faster results"
    )
    
    # Data info
    st.markdown("---")
    st.markdown("### 📊 Data Status")
    if st.session_state.df_original is not None:
        st.success(f"✓ Data Loaded: {len(st.session_state.df_original)} rows")
    else:
        st.warning("⚠ No data loaded")
    
    if st.session_state.df_sentiment is not None:
        st.success(f"✓ Sentiment Analyzed")
    
    if st.session_state.trained_models:
        st.success(f"✓ Models Trained: {len(st.session_state.trained_models)}")

# Main content
st.markdown('<div class="main-header">📊 MBG Sentiment Analysis Dashboard</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Complete End-to-End Pipeline for Program Makan Bergizi Gratis Sentiment Analysis</div>', unsafe_allow_html=True)

# Welcome section
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div class="metric-card">
        <h3>🚀 Quick Start</h3>
        <p>Upload your data and start analyzing in minutes</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="metric-card">
        <h3>🤖 AI-Powered</h3>
        <p>BERT models + Custom ML algorithms</p>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="metric-card">
        <h3>📈 Interactive</h3>
        <p>Real-time visualizations and insights</p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# Features overview
st.markdown("## 🎯 Key Features")

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    <div class="feature-box">
        <h4>📁 Data Management</h4>
        <ul>
            <li>CSV upload with sample data</li>
            <li>Dynamic column selection</li>
            <li>Advanced text preprocessing</li>
            <li>Custom normalization dictionaries</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="feature-box">
        <h4>🤖 Sentiment Analysis</h4>
        <ul>
            <li>BERT-based initial labeling</li>
            <li>Custom rule-based refinement</li>
            <li>Editable veto rules</li>
            <li>Batch processing with caching</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="feature-box">
        <h4>🔧 Model Training</h4>
        <ul>
            <li>Multiple ML algorithms (SVM, KNN)</li>
            <li>GridSearchCV hyperparameter tuning</li>
            <li>SMOTE for imbalanced data</li>
            <li>Cross-validation with metrics</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="feature-box">
        <h4>📊 Visualization & Export</h4>
        <ul>
            <li>Interactive Plotly charts</li>
            <li>Word clouds and distributions</li>
            <li>Model comparison dashboard</li>
            <li>CSV, Pickle, and PDF export</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# Quick start guide
st.markdown("## 🚀 Getting Started")

with st.expander("📖 Step-by-Step Guide", expanded=False):
    st.markdown("""
    ### Step 1: Upload Data
    Navigate to **📁 Data Upload & Preprocessing** page:
    - Upload your CSV file or use sample data
    - Select the text column to analyze
    - Configure preprocessing options
    - Review data statistics
    
    ### Step 2: Analyze Sentiment
    Go to **🤖 Sentiment Analysis** page:
    - Run BERT-based sentiment detection
    - Review and adjust veto rules
    - Customize sentiment mapping logic
    - View initial sentiment distribution
    
    ### Step 3: Train Models
    Visit **🔧 Model Training & Evaluation** page:
    - Select models to train (SVM, KNN)
    - Configure hyperparameters
    - Choose Quick Mode or Full Analysis
    - Monitor training progress
    
    ### Step 4: Explore Results
    Check **📈 Visualizations** page:
    - Interactive sentiment distributions
    - Word clouds per category
    - Confusion matrices
    - ROC curves and learning curves
    
    ### Step 5: Analyze Errors
    Review **🔍 Error Analysis** page:
    - Feature importance analysis
    - Misclassification examples
    - Model comparison metrics
    
    ### Step 6: Test New Text
    Use **🎯 Live Predictor** page:
    - Enter custom text
    - Get real-time predictions
    - Compare model outputs
    
    ### Step 7: Export Results
    Download from **💾 Export & Reports** page:
    - Predictions CSV
    - Trained models (pickle)
    - Comprehensive PDF report
    - Visualization images
    """)

# Sample data information
st.markdown("## 📦 Sample Data")
st.info("""
**Demo Dataset Available**: The application includes a sample dataset about MBG (Makan Bergizi Gratis) program 
with 200 pre-labeled Indonesian tweets. Perfect for testing the pipeline without uploading your own data!
""")

# Technical requirements
with st.expander("⚙️ Technical Requirements & Notes"):
    st.markdown("""
    ### System Requirements
    - **RAM**: Minimum 4GB (8GB recommended for large datasets)
    - **Storage**: ~500MB for BERT model cache
    - **Python**: 3.8 or higher
    
    ### Performance Tips
    - **Quick Mode**: Uses reduced GridSearchCV parameters (~2-3 minutes)
    - **Full Analysis**: Complete hyperparameter search (~5-15 minutes)
    - **Max Dataset Size**: 10,000 rows recommended for optimal performance
    - **Batch Size**: Automatically adjusted based on available memory
    
    ### Dependencies
    All required packages are listed in `requirements.txt`:
    - streamlit
    - pandas, numpy
    - transformers, torch
    - scikit-learn, imblearn
    - plotly, wordcloud
    - Sastrawi (Indonesian NLP)
    - reportlab (PDF generation)
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #6b7280; padding: 2rem;'>
    <p><strong>MBG Sentiment Analysis Dashboard v1.0</strong></p>
    <p>Built with Streamlit | Powered by BERT & Scikit-learn</p>
    <p>© 2025 - All Rights Reserved</p>
</div>
""", unsafe_allow_html=True)