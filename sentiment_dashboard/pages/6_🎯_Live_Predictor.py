import streamlit as st
import pandas as pd
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from utils.text_processing import clean_text_advanced
from utils.sentiment_analysis import predict_single_text

st.set_page_config(page_title="Live Predictor", page_icon="🎯", layout="wide")

st.title("🎯 Live Sentiment Predictor")
st.markdown("Test sentiment analysis on new text in real-time")

# Check prerequisites
if not st.session_state.bert_model_loaded:
    st.error("⚠️ BERT model not loaded. Please load the model in the Sentiment Analysis page first!")
    st.stop()

# Input section
st.markdown("### 📝 Input Text")

col1, col2 = st.columns([2, 1])

with col1:
    input_text = st.text_area(
        "Enter Indonesian text to analyze:",
        height=150,
        placeholder="Contoh: Program makan bergizi gratis ini sangat membantu anak-anak Indonesia..."
    )
    
    if st.button("🔍 Analyze Sentiment", type="primary", use_container_width=True):
        if input_text.strip():
            st.session_state.prediction_input = input_text
            st.session_state.run_prediction = True
        else:
            st.warning("Please enter some text first!")

with col2:
    st.markdown("#### Quick Examples")
    examples = {
        "Positive": "Program MBG sangat bagus dan membantu mengatasi stunting pada anak",
        "Negative": "Masalah korupsi harus dihindari, program ini hanya buang-buang uang negara",
        "Neutral": "Kita lihat saja nanti bagaimana pelaksanaan program makan gratis ini"
    }
    
    for sentiment, text in examples.items():
        if st.button(f"📄 {sentiment} Example", use_container_width=True):
            st.session_state.prediction_input = text
            st.session_state.run_prediction = True
            st.rerun()

# Prediction section
if 'prediction_input' in st.session_state and st.session_state.get('run_prediction', False):
    
    st.markdown("---")
    st.markdown("### 🔬 Analysis Results")
    
    with st.spinner("Analyzing..."):
        # Clean text
        cleaned_text = clean_text_advanced(
            st.session_state.prediction_input,
            normalization_dict=st.session_state.normalization_dict,
            use_stemming=st.session_state.preprocessing_options['use_stemming'],
            use_stopwords=st.session_state.preprocessing_options['use_stopwords']
        )
        
        # Get prediction
        prediction = predict_single_text(
            st.session_state.prediction_input,
            cleaned_text,
            st.session_state.sentiment_analyzer,
            st.session_state.model_type,
            veto_words=st.session_state.veto_words
        )
    
    # Display original and cleaned text
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Original Text")
        st.text_area("", prediction['original_text'], height=100, disabled=True, label_visibility="collapsed")
    
    with col2:
        st.markdown("#### Cleaned Text")
        st.text_area("", prediction['cleaned_text'], height=100, disabled=True, label_visibility="collapsed")
    
    # Main prediction result
    st.markdown("---")
    
    sentiment_colors = {
        'positive': 'green',
        'neutral': 'blue',
        'negative': 'red'
    }
    
    sentiment_emojis = {
        'positive': '😊',
        'neutral': '😐',
        'negative': '😞'
    }
    
    final_sentiment = prediction['final_sentiment']
    color = sentiment_colors.get(final_sentiment, 'gray')
    emoji = sentiment_emojis.get(final_sentiment, '🤔')
    
    st.markdown(f"## {emoji} Final Sentiment: :{color}[{final_sentiment.upper()}]")
    
    # Detailed breakdown
    st.markdown("---")
    st.markdown("### 📊 Detailed Breakdown")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("#### BERT Prediction")
        st.metric("Label", prediction['bert_label'])
        st.metric("Confidence", f"{prediction['bert_score']:.3f}")
        
        # Show all BERT scores
        with st.expander("All BERT Scores"):
            for score_data in prediction['bert_all_scores']:
                st.write(f"{score_data['label']}: {score_data['score']:.3f}")
    
    with col2:
        st.markdown("#### Rule Analysis")
        st.metric("Veto Triggered", "Yes" if prediction['veto_triggered'] else "No")
        st.metric("Negative Keywords", prediction['negative_keywords_count'])
        st.metric("Positive Keywords", prediction['positive_keywords_count'])
        
        if prediction['veto_triggered']:
            st.error("⚠️ Veto rule applied - immediately negative!")
    
    with col3:
        st.markdown("#### Final Result")
        st.metric("Sentiment", final_sentiment.upper())
        st.metric("Confidence Level", prediction['confidence'].upper())
        
        # Logic explanation
        if prediction['veto_triggered']:
            logic = "Veto Rule"
        elif prediction['negative_keywords_count'] > prediction['positive_keywords_count']:
            logic = "Keyword (Negative)"
        elif prediction['positive_keywords_count'] > prediction['negative_keywords_count']:
            logic = "Keyword (Positive)"
        else:
            logic = "BERT Fallback"
        
        st.metric("Logic Applied", logic)
    
    # ML Model Predictions (if available)
    if st.session_state.trained_models:
        st.markdown("---")
        st.markdown("### 🤖 ML Model Predictions")
        
        ml_results = []
        
        for model_name, model_data in st.session_state.trained_models.items():
            model = model_data['model']
            
            try:
                ml_pred = model.predict([cleaned_text])[0]
                ml_proba = model.predict_proba([cleaned_text])[0]
                
                sentiment_map_reverse = {0: 'negative', 1: 'neutral', 2: 'positive'}
                ml_sentiment = sentiment_map_reverse[ml_pred]
                
                ml_results.append({
                    'Model': model_name,
                    'Prediction': ml_sentiment,
                    'Confidence': f"{ml_proba[ml_pred]:.3f}",
                    'Neg_Prob': f"{ml_proba[0]:.3f}",
                    'Neu_Prob': f"{ml_proba[1]:.3f}",
                    'Pos_Prob': f"{ml_proba[2]:.3f}"
                })
            except Exception as e:
                st.warning(f"Could not get prediction from {model_name}: {e}")
        
        if ml_results:
            ml_df = pd.DataFrame(ml_results)
            st.dataframe(ml_df, use_container_width=True)
    
    st.session_state.run_prediction = False

# Batch prediction section
st.markdown("---")
st.markdown("### 📦 Batch Prediction")
st.markdown("Upload a CSV file with multiple texts for batch prediction")

uploaded_batch = st.file_uploader(
    "Upload CSV file with text column",
    type=['csv'],
    key="batch_predictor"
)

if uploaded_batch:
    try:
        batch_df = pd.read_csv(uploaded_batch)
        st.success(f"✓ Loaded {len(batch_df)} rows")
        
        text_cols = batch_df.select_dtypes(include=['object']).columns.tolist()
        batch_text_col = st.selectbox("Select text column:", text_cols)
        
        if st.button("🚀 Run Batch Prediction", type="primary"):
            with st.spinner("Processing batch predictions..."):
                progress_bar = st.progress(0)
                
                batch_predictions = []
                total = len(batch_df)
                
                for idx, row in batch_df.iterrows():
                    text = row[batch_text_col]
                    cleaned = clean_text_advanced(text, normalization_dict=st.session_state.normalization_dict)
                    
                    pred = predict_single_text(
                        text, cleaned,
                        st.session_state.sentiment_analyzer,
                        st.session_state.model_type,
                        veto_words=st.session_state.veto_words
                    )
                    
                    batch_predictions.append({
                        'original_text': text,
                        'cleaned_text': cleaned,
                        'sentiment': pred['final_sentiment'],
                        'confidence': pred['confidence'],
                        'bert_score': pred['bert_score']
                    })
                    
                    progress_bar.progress((idx + 1) / total)
                
                progress_bar.empty()
                
                result_df = pd.DataFrame(batch_predictions)
                st.success("✓ Batch prediction completed!")
                
                # Show results
                st.dataframe(result_df, use_container_width=True, height=400)
                
                # Download results
                csv = result_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Download Results",
                    data=csv,
                    file_name=f"batch_predictions_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
    
    except Exception as e:
        st.error(f"Error processing batch file: {e}")

st.markdown("---")
st.caption("💡 Tip: The Live Predictor uses the same BERT model and rules configured in the Sentiment Analysis page")