# ============================================================================
# SENTIMENT ANALYSIS UTILITIES
# BERT-based analysis with custom rule-based refinement
# ============================================================================

import streamlit as st
from transformers import pipeline
import pandas as pd
import numpy as np

# Default strong negative/positive keywords
DEFAULT_STRONG_NEGATIVE = [
    'tidak_', 'bukan_', 'jangan_', 'kurang_', 'buruk', 'jelek', 'parah',
    'kecewa', 'benci', 'terburuk', 'tipu', 'marah', 'sedih', 'jijik', 
    'menyesal', 'ancur', 'busuk', 'hilang', 'bodoh', 'hambar', 'mustahil'
]

DEFAULT_STRONG_POSITIVE = [
    'bagus', 'baik', 'suka', 'puas', 'mantap', 'sempurna', 'terbaik',
    'recommended', 'senang', 'luar biasa', 'hebat', 'top', 'enak',
    'berguna', 'membantu', 'terima kasih', 'sukses', 'berhasil'
]

# Load BERT model with caching
@st.cache_resource
def load_sentiment_model(model_choice="ayameRushia"):
    """
    Load and cache BERT sentiment analysis model
    Returns: (model, model_type)
    """
    try:
        if model_choice == "ayameRushia":
            model = pipeline(
                "sentiment-analysis",
                model="ayameRushia/bert-base-indonesian-1.5G-sentiment-analysis-smsa",
                device=-1,
                return_all_scores=True
            )
            return model, "ayameRushia"
    except Exception as e:
        st.warning(f"Primary model failed: {str(e)[:100]}. Loading fallback...")
        
    try:
        model = pipeline(
            "sentiment-analysis",
            model="cardiffnlp/twitter-xlm-roberta-base-sentiment",
            device=-1,
            return_all_scores=True
        )
        return model, "xlm-roberta"
    except Exception as e:
        st.error(f"All models failed to load: {e}")
        return None, None


def analyze_sentiment_batch(texts, sentiment_analyzer, batch_size=32, progress_bar=None):
    """
    Batch sentiment analysis with progress tracking
    """
    results = []
    total = len(texts)
    
    for i in range(0, total, batch_size):
        batch = texts[i:i+batch_size]
        
        try:
            batch_results = sentiment_analyzer(batch)
            for scores in batch_results:
                best = max(scores, key=lambda x: x['score'])
                results.append({
                    'label': best['label'],
                    'score': best['score'],
                    'all_scores': scores
                })
        except Exception as e:
            # Process individually if batch fails
            for text in batch:
                try:
                    scores = sentiment_analyzer(text)[0]
                    best = max(scores, key=lambda x: x['score'])
                    results.append({
                        'label': best['label'],
                        'score': best['score'],
                        'all_scores': scores
                    })
                except:
                    results.append({
                        'label': 'NEUTRAL',
                        'score': 0.5,
                        'all_scores': [{'label': 'NEUTRAL', 'score': 0.5}]
                    })
        
        if progress_bar:
            progress = min(i + batch_size, total) / total
            progress_bar.progress(progress)
    
    return results


def map_sentiment_contextual(row, model_type, veto_words=None, 
                            strong_negative=None, strong_positive=None):
    """
    Map sentiment using veto rule logic
    
    Priority:
    1. Veto words (immediate negative)
    2. Keyword comparison (count-based)
    3. BERT model prediction
    """
    if veto_words is None:
        veto_words = []
    if strong_negative is None:
        strong_negative = DEFAULT_STRONG_NEGATIVE
    if strong_positive is None:
        strong_positive = DEFAULT_STRONG_POSITIVE
    
    label = row['raw_sentiment_label'].upper()
    text = str(row['text_cleaned']).lower()
    
    # VETO RULE: If any veto word present, immediately negative
    if any(word in text for word in veto_words):
        return 'negative'
    
    # KEYWORD COMPARISON
    negative_count = sum(1 for word in strong_negative if word in text)
    positive_count = sum(1 for word in strong_positive if word in text)
    
    if negative_count > positive_count:
        return 'negative'
    if positive_count > negative_count:
        return 'positive'
    
    # BERT MODEL FALLBACK
    mapping = {
        "ayameRushia": {
            'POSITIVE': 'positive', 'NEGATIVE': 'negative', 
            'LABEL_0': 'negative', 'LABEL_1': 'neutral', 'LABEL_2': 'positive'
        },
        "xlm-roberta": {
            'POSITIVE': 'positive', 'NEGATIVE': 'negative', 'NEUTRAL': 'neutral'
        }
    }
    
    return mapping.get(model_type, {}).get(label, 'neutral')


def categorize_confidence(score):
    """Categorize confidence levels"""
    if score >= 0.80:
        return 'high'
    if score >= 0.60:
        return 'medium'
    return 'low'


def run_sentiment_analysis(df, text_column, sentiment_analyzer, model_type,
                          veto_words=None, strong_negative=None, strong_positive=None):
    """
    Complete sentiment analysis pipeline
    Returns: DataFrame with sentiment columns
    """
    # Prepare data
    df_work = df.copy()
    
    # Run BERT analysis
    st.info("Running BERT sentiment analysis...")
    progress_bar = st.progress(0)
    
    sentiment_results = analyze_sentiment_batch(
        df_work[text_column].tolist(),
        sentiment_analyzer,
        batch_size=32,
        progress_bar=progress_bar
    )
    
    # Add raw results
    df_work['raw_sentiment_label'] = [r['label'] for r in sentiment_results]
    df_work['raw_sentiment_score'] = [r['score'] for r in sentiment_results]
    
    # Apply contextual mapping
    st.info("Applying custom sentiment rules...")
    df_work['sentiment_category'] = df_work.apply(
        lambda row: map_sentiment_contextual(
            row, model_type, veto_words, strong_negative, strong_positive
        ),
        axis=1
    )
    
    # Add confidence levels
    df_work['confidence_level'] = df_work['raw_sentiment_score'].apply(categorize_confidence)
    
    # Add features
    df_work['text_length'] = df_work[text_column].apply(lambda x: len(str(x)))
    df_work['word_count'] = df_work[text_column].apply(lambda x: len(str(x).split()))
    
    progress_bar.empty()
    
    return df_work


def get_sentiment_distribution(df):
    """Get sentiment distribution statistics"""
    dist = df['sentiment_category'].value_counts()
    total = len(df)
    
    return {
        sentiment: {
            'count': int(count),
            'percentage': round((count / total) * 100, 2)
        }
        for sentiment, count in dist.items()
    }


def predict_single_text(text, cleaned_text, sentiment_analyzer, model_type,
                       veto_words=None, strong_negative=None, strong_positive=None):
    """
    Predict sentiment for a single text
    Returns: dict with predictions
    """
    # BERT prediction
    bert_result = sentiment_analyzer(cleaned_text)[0]
    best_bert = max(bert_result, key=lambda x: x['score'])
    
    # Create temporary dataframe for mapping
    temp_df = pd.DataFrame({
        'text_cleaned': [cleaned_text],
        'raw_sentiment_label': [best_bert['label']]
    })
    
    # Apply contextual mapping
    final_sentiment = map_sentiment_contextual(
        temp_df.iloc[0], model_type, veto_words, strong_negative, strong_positive
    )
    
    # Check for veto words
    veto_triggered = any(word in cleaned_text.lower() for word in (veto_words or []))
    
    # Count keywords
    negative_count = sum(1 for word in (strong_negative or DEFAULT_STRONG_NEGATIVE) 
                        if word in cleaned_text.lower())
    positive_count = sum(1 for word in (strong_positive or DEFAULT_STRONG_POSITIVE) 
                        if word in cleaned_text.lower())
    
    return {
        'original_text': text,
        'cleaned_text': cleaned_text,
        'bert_label': best_bert['label'],
        'bert_score': best_bert['score'],
        'bert_all_scores': bert_result,
        'final_sentiment': final_sentiment,
        'confidence': categorize_confidence(best_bert['score']),
        'veto_triggered': veto_triggered,
        'negative_keywords_count': negative_count,
        'positive_keywords_count': positive_count
    }