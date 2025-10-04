import streamlit as st
import pandas as pd
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from utils.sentiment_analysis import (
    load_sentiment_model, run_sentiment_analysis, get_sentiment_distribution,
    DEFAULT_STRONG_NEGATIVE, DEFAULT_STRONG_POSITIVE
)
from utils.text_processing import DEFAULT_VETO_WORDS
from utils.visualization import (
    plot_sentiment_distribution, plot_sentiment_pie, 
    plot_confidence_distribution, plot_sentiment_by_confidence
)

st.set_page_config(page_title="Sentiment Analysis", page_icon="🤖", layout="wide")

st.title("🤖 Sentiment Analysis")
st.markdown("BERT-based sentiment detection with custom rule-based refinement")

# Check prerequisites
if st.session_state.df_cleaned is None:
    st.error("⚠️ No cleaned data available. Please upload and preprocess data first!")
    st.stop()

# Tabs
tab1, tab2, tab3, tab4 = st.tabs(["🔧 Configuration", "🚀 Run Analysis", "📊 Results", "🎯 Rule Tuning"])

# TAB 1: CONFIGURATION
with tab1:
    st.markdown("### Model Configuration")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("#### BERT Model Selection")
        
        model_choice = st.radio(
            "Choose BERT model:",
            ["ayameRushia (Indonesian Specific)", "XLM-RoBERTa (Multilingual)"],
            help="ayameRushia is optimized for Indonesian text"
        )
        
        model_name = "ayameRushia" if "ayameRushia" in model_choice else "xlm-roberta"
        
        if not st.session_state.bert_model_loaded:
            if st.button("🔄 Load BERT Model", type="primary"):
                with st.spinner("Loading BERT model... (this may take a moment)"):
                    analyzer, model_type = load_sentiment_model(model_name)
                    if analyzer:
                        st.session_state.sentiment_analyzer = analyzer
                        st.session_state.model_type = model_type
                        st.session_state.bert_model_loaded = True
                        st.success(f"✓ Model loaded: {model_type}")
                        st.rerun()
                    else:
                        st.error("Failed to load model!")
        else:
            st.success(f"✓ Model loaded: {st.session_state.model_type}")
            if st.button("🔄 Reload Model"):
                st.session_state.bert_model_loaded = False
                st.rerun()
    
    with col2:
        st.markdown("#### Batch Processing")
        
        batch_size = st.slider(
            "Batch Size",
            min_value=8,
            max_value=64,
            value=32,
            step=8,
            help="Larger batches are faster but use more memory"
        )
        
        st.info(f"""
        **Current Configuration:**
        - Dataset size: {len(st.session_state.df_cleaned)} texts
        - Estimated time: ~{len(st.session_state.df_cleaned) / batch_size * 0.5:.1f} seconds
        - Memory usage: ~500MB
        """)

# TAB 2: RUN ANALYSIS
with tab2:
    st.markdown("### 🚀 Run Sentiment Analysis")
    
    if not st.session_state.bert_model_loaded:
        st.warning("⚠️ Please load the BERT model first in the Configuration tab")
    else:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("#### Analysis Parameters")
            
            # Initialize veto words if not exists
            if 'veto_words' not in st.session_state or not st.session_state.veto_words:
                st.session_state.veto_words = DEFAULT_VETO_WORDS.copy()
            
            st.info(f"""
            **Current Settings:**
            - Model: {st.session_state.model_type}
            - Veto words: {len(st.session_state.veto_words)} configured
            - Strong negative keywords: {len(DEFAULT_STRONG_NEGATIVE)}
            - Strong positive keywords: {len(DEFAULT_STRONG_POSITIVE)}
            """)
            
            if st.button("🚀 Start Sentiment Analysis", type="primary", use_container_width=True):
                with st.spinner("Analyzing sentiments... This may take a few minutes"):
                    
                    df_sentiment = run_sentiment_analysis(
                        st.session_state.df_cleaned,
                        'text_cleaned',
                        st.session_state.sentiment_analyzer,
                        st.session_state.model_type,
                        veto_words=st.session_state.veto_words,
                        strong_negative=DEFAULT_STRONG_NEGATIVE,
                        strong_positive=DEFAULT_STRONG_POSITIVE
                    )
                    
                    st.session_state.df_sentiment = df_sentiment
                    st.success("✓ Sentiment analysis completed!")
                    st.balloons()
                    st.rerun()
        
        with col2:
            st.markdown("#### Analysis Steps")
            st.markdown("""
            1. **BERT Prediction**: Initial sentiment labels from BERT model
            2. **Veto Rule Check**: Immediate negative if veto words present
            3. **Keyword Comparison**: Count positive vs negative keywords
            4. **Final Label**: Apply contextual mapping logic
            """)

# TAB 3: RESULTS
with tab3:
    st.markdown("### 📊 Sentiment Analysis Results")
    
    if st.session_state.df_sentiment is None:
        st.warning("⚠️ No analysis results yet. Run sentiment analysis first!")
    else:
        df = st.session_state.df_sentiment
        
        # Summary metrics
        st.markdown("#### Summary Statistics")
        col1, col2, col3, col4 = st.columns(4)
        
        dist = get_sentiment_distribution(df)
        
        with col1:
            neg_data = dist.get('negative', {'count': 0, 'percentage': 0})
            st.metric(
                "Negative", 
                neg_data['count'],
                f"{neg_data['percentage']}%",
                delta_color="inverse"
            )
        
        with col2:
            neu_data = dist.get('neutral', {'count': 0, 'percentage': 0})
            st.metric(
                "Neutral", 
                neu_data['count'],
                f"{neu_data['percentage']}%"
            )
        
        with col3:
            pos_data = dist.get('positive', {'count': 0, 'percentage': 0})
            st.metric(
                "Positive", 
                pos_data['count'],
                f"{pos_data['percentage']}%",
                delta_color="normal"
            )
        
        with col4:
            avg_confidence = df['raw_sentiment_score'].mean()
            st.metric(
                "Avg Confidence",
                f"{avg_confidence:.3f}",
                f"{(avg_confidence - 0.5) * 100:+.1f}%"
            )
        
        st.markdown("---")
        
        # Visualizations
        col1, col2 = st.columns(2)
        
        with col1:
            fig1 = plot_sentiment_distribution(df)
            st.plotly_chart(fig1, use_container_width=True)
        
        with col2:
            fig2 = plot_sentiment_pie(df)
            st.plotly_chart(fig2, use_container_width=True)
        
        col3, col4 = st.columns(2)
        
        with col3:
            fig3 = plot_confidence_distribution(df)
            st.plotly_chart(fig3, use_container_width=True)
        
        with col4:
            fig4 = plot_sentiment_by_confidence(df)
            st.plotly_chart(fig4, use_container_width=True)
        
        st.markdown("---")
        
        # Sample results
        st.markdown("#### Sample Results")
        
        sentiment_filter = st.multiselect(
            "Filter by sentiment:",
            ['negative', 'neutral', 'positive'],
            default=['negative', 'neutral', 'positive']
        )
        
        confidence_filter = st.multiselect(
            "Filter by confidence:",
            ['low', 'medium', 'high'],
            default=['low', 'medium', 'high']
        )
        
        filtered_df = df[
            (df['sentiment_category'].isin(sentiment_filter)) &
            (df['confidence_level'].isin(confidence_filter))
        ]
        
        display_cols = ['text_cleaned', 'sentiment_category', 'confidence_level', 
                       'raw_sentiment_label', 'raw_sentiment_score']
        
        st.dataframe(
            filtered_df[display_cols].head(50),
            use_container_width=True,
            height=400
        )
        
        st.markdown(f"*Showing {min(50, len(filtered_df))} of {len(filtered_df)} filtered results*")

# TAB 4: RULE TUNING
with tab4:
    st.markdown("### 🎯 Sentiment Rule Tuning")
    st.markdown("Customize the rule-based sentiment refinement logic")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Veto Negative Words")
        st.info("If ANY of these words appear, sentiment is immediately set to NEGATIVE")
        
        # Display current veto words
        if 'veto_words' not in st.session_state:
            st.session_state.veto_words = DEFAULT_VETO_WORDS.copy()
        
        st.text_area(
            "Current veto words (one per line):",
            value="\n".join(st.session_state.veto_words),
            height=200,
            key="veto_display",
            disabled=True
        )
        
        # Add new veto word
        new_veto = st.text_input("Add new veto word:")
        if st.button("➕ Add Veto Word"):
            if new_veto and new_veto not in st.session_state.veto_words:
                st.session_state.veto_words.append(new_veto)
                st.success(f"Added: {new_veto}")
                st.rerun()
            elif new_veto in st.session_state.veto_words:
                st.warning("Word already in veto list")
        
        # Remove veto word
        if st.session_state.veto_words:
            remove_veto = st.selectbox("Remove veto word:", st.session_state.veto_words)
            if st.button("➖ Remove Selected"):
                st.session_state.veto_words.remove(remove_veto)
                st.success(f"Removed: {remove_veto}")
                st.rerun()
        
        if st.button("🔄 Reset to Default"):
            st.session_state.veto_words = DEFAULT_VETO_WORDS.copy()
            st.success("Reset to default veto words")
            st.rerun()
    
    with col2:
        st.markdown("#### Strong Keyword Lists")
        
        with st.expander("📕 Strong Negative Keywords", expanded=False):
            st.markdown("Keywords that indicate negative sentiment:")
            neg_df = pd.DataFrame({'Keywords': DEFAULT_STRONG_NEGATIVE})
            st.dataframe(neg_df, use_container_width=True, height=200)
        
        with st.expander("📗 Strong Positive Keywords", expanded=False):
            st.markdown("Keywords that indicate positive sentiment:")
            pos_df = pd.DataFrame({'Keywords': DEFAULT_STRONG_POSITIVE})
            st.dataframe(pos_df, use_container_width=True, height=200)
        
        st.markdown("---")
        st.markdown("#### Logic Flow")
        st.markdown("""
        **Priority Order:**
        1. **Veto Check** (Highest Priority)
           - If any veto word found → NEGATIVE
        2. **Keyword Comparison** (Medium Priority)
           - If negative_count > positive_count → NEGATIVE
           - If positive_count > negative_count → POSITIVE
        3. **BERT Model** (Fallback)
           - Use BERT prediction if no keywords match
        """)
        
        # Test single text
        st.markdown("---")
        st.markdown("#### Test Rule Logic")
        
        test_text = st.text_area(
            "Enter text to test:",
            placeholder="Type Indonesian text here to test sentiment rules..."
        )
        
        if test_text and st.button("Test Sentiment"):
            from utils.text_processing import clean_text_advanced
            
            cleaned = clean_text_advanced(
                test_text,
                normalization_dict=st.session_state.normalization_dict
            )
            
            st.markdown("**Cleaned Text:**")
            st.code(cleaned)
            
            # Check veto
            veto_found = [w for w in st.session_state.veto_words if w in cleaned.lower()]
            
            # Count keywords
            neg_count = sum(1 for w in DEFAULT_STRONG_NEGATIVE if w in cleaned.lower())
            pos_count = sum(1 for w in DEFAULT_STRONG_POSITIVE if w in cleaned.lower())
            
            col_a, col_b, col_c = st.columns(3)
            
            with col_a:
                st.metric("Veto Words Found", len(veto_found))
                if veto_found:
                    st.warning(f"Found: {', '.join(veto_found)}")
            
            with col_b:
                st.metric("Negative Keywords", neg_count)
            
            with col_c:
                st.metric("Positive Keywords", pos_count)
            
            # Determine final sentiment
            if veto_found:
                final = "NEGATIVE (Veto Rule)"
                color = "red"
            elif neg_count > pos_count:
                final = "NEGATIVE (Keyword Count)"
                color = "orange"
            elif pos_count > neg_count:
                final = "POSITIVE (Keyword Count)"
                color = "green"
            else:
                final = "NEUTRAL (BERT Fallback)"
                color = "blue"
            
            st.markdown(f"**Final Sentiment:** :{color}[{final}]")

st.markdown("---")
st.markdown("**Next Step:** Go to **🔧 Model Training** page to train custom ML models")