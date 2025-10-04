import streamlit as st
import pandas as pd
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from utils.visualization import (
    plot_sentiment_distribution, plot_sentiment_pie, plot_confidence_distribution,
    plot_sentiment_by_confidence, generate_wordcloud, plot_confusion_matrix,
    plot_roc_curves, plot_feature_importance, plot_model_comparison,
    plot_text_length_distribution, plot_word_count_distribution
)
from utils.model_training import get_feature_importance, compare_models

st.set_page_config(page_title="Visualizations", page_icon="📈", layout="wide")

st.title("📈 Interactive Visualizations")
st.markdown("Comprehensive visual analysis of sentiment data and model performance")

# Check prerequisites
if st.session_state.df_sentiment is None:
    st.error("⚠️ No sentiment results available. Please run sentiment analysis first!")
    st.stop()

# Sidebar filters
with st.sidebar:
    st.markdown("### 🎛️ Filters")
    
    sentiment_filter = st.multiselect(
        "Filter Sentiments:",
        ['negative', 'neutral', 'positive'],
        default=['negative', 'neutral', 'positive']
    )
    
    confidence_filter = st.multiselect(
        "Filter Confidence:",
        ['low', 'medium', 'high'],
        default=['low', 'medium', 'high']
    )
    
    st.markdown("---")
    st.markdown("### 📊 Quick Stats")
    
    df_filtered = st.session_state.df_sentiment[
        (st.session_state.df_sentiment['sentiment_category'].isin(sentiment_filter)) &
        (st.session_state.df_sentiment['confidence_level'].isin(confidence_filter))
    ]
    
    st.metric("Filtered Rows", len(df_filtered))
    st.metric("Total Rows", len(st.session_state.df_sentiment))

# Apply filters
df = df_filtered

if len(df) == 0:
    st.warning("⚠️ No data matches the selected filters. Please adjust your filter settings.")
    st.stop()

# Tabs for organization
tab1, tab2, tab3, tab4 = st.tabs(["📊 Sentiment Overview", "☁️ Word Clouds", "🤖 Model Performance", "📈 Advanced Charts"])

# TAB 1: SENTIMENT OVERVIEW
with tab1:
    st.markdown("### Sentiment Distribution Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig1 = plot_sentiment_distribution(df, "Sentiment Distribution")
        st.plotly_chart(fig1, use_container_width=True)
    
    with col2:
        fig2 = plot_sentiment_pie(df, "Sentiment Proportion")
        st.plotly_chart(fig2, use_container_width=True)
    
    st.markdown("---")
    
    col3, col4 = st.columns(2)
    
    with col3:
        fig3 = plot_confidence_distribution(df)
        st.plotly_chart(fig3, use_container_width=True)
    
    with col4:
        fig4 = plot_sentiment_by_confidence(df)
        st.plotly_chart(fig4, use_container_width=True)
    
    st.markdown("---")
    
    # Text statistics
    st.markdown("### Text Characteristics")
    
    col5, col6 = st.columns(2)
    
    with col5:
        fig5 = plot_text_length_distribution(df, 'text_cleaned')
        st.plotly_chart(fig5, use_container_width=True)
    
    with col6:
        fig6 = plot_word_count_distribution(df, 'text_cleaned')
        st.plotly_chart(fig6, use_container_width=True)
    
    # Summary statistics
    st.markdown("---")
    st.markdown("### Summary Statistics")
    
    col7, col8, col9, col10 = st.columns(4)
    
    with col7:
        st.metric("Avg Text Length", f"{df['text_cleaned'].str.len().mean():.0f} chars")
    with col8:
        st.metric("Avg Word Count", f"{df['text_cleaned'].str.split().str.len().mean():.1f}")
    with col9:
        st.metric("Avg Confidence", f"{df['raw_sentiment_score'].mean():.3f}")
    with col10:
        st.metric("Unique Words", len(set(' '.join(df['text_cleaned'].dropna()).split())))

# TAB 2: WORD CLOUDS
with tab2:
    st.markdown("### Word Clouds by Sentiment Category")
    st.info("Word clouds show the most frequent words in each sentiment category")
    
    sentiment_options = ['negative', 'neutral', 'positive']
    colormap_options = {
        'negative': 'Reds',
        'neutral': 'Blues',
        'positive': 'Greens'
    }
    
    for sentiment in sentiment_options:
        if sentiment not in df['sentiment_category'].unique():
            continue
        
        st.markdown(f"#### {sentiment.upper()} Sentiment")
        
        text_data = " ".join(df[df['sentiment_category'] == sentiment]['text_cleaned'].dropna())
        
        if text_data and len(text_data.split()) > 10:
            img_base64 = generate_wordcloud(text_data, colormap=colormap_options[sentiment])
            
            if img_base64:
                st.markdown(
                    f'<img src="{img_base64}" style="width:100%; border-radius:10px; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">',
                    unsafe_allow_html=True
                )
                
                # Top 10 words
                words = text_data.split()
                word_freq = pd.Series(words).value_counts().head(10)
                
                col1, col2 = st.columns([2, 1])
                with col2:
                    st.markdown(f"**Top 10 Words in {sentiment}:**")
                    for word, count in word_freq.items():
                        st.text(f"{word}: {count}")
            else:
                st.warning(f"Could not generate word cloud for {sentiment}")
        else:
            st.warning(f"Insufficient data for {sentiment} word cloud (need at least 10 words)")
        
        st.markdown("---")

# TAB 3: MODEL PERFORMANCE
with tab3:
    st.markdown("### Machine Learning Model Performance")
    
    if not st.session_state.trained_models:
        st.warning("⚠️ No trained models available. Please train models first in the Model Training page.")
    else:
        # Model comparison
        st.markdown("#### Model Comparison")
        
        comparison_df = compare_models(st.session_state.trained_models)
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.dataframe(comparison_df, use_container_width=True, height=200)
        
        with col2:
            fig_comp = plot_model_comparison(comparison_df)
            st.plotly_chart(fig_comp, use_container_width=True)
        
        st.markdown("---")
        
        # Individual model visualizations
        st.markdown("#### Detailed Model Visualizations")
        
        selected_model = st.selectbox(
            "Select model to visualize:",
            list(st.session_state.trained_models.keys())
        )
        
        model_result = st.session_state.trained_models[selected_model]
        metrics = model_result['metrics']
        
        # Confusion Matrix
        st.markdown(f"##### Confusion Matrix - {selected_model}")
        class_names = ['negative', 'neutral', 'positive']
        fig_cm = plot_confusion_matrix(
            metrics['confusion_matrix'],
            class_names,
            title=f"Confusion Matrix - {selected_model}"
        )
        st.plotly_chart(fig_cm, use_container_width=True)
        
        # ROC Curves
        st.markdown(f"##### ROC Curves - {selected_model}")
        fig_roc = plot_roc_curves(metrics['roc_data'], class_names)
        st.plotly_chart(fig_roc, use_container_width=True)
        
        # Feature Importance
        st.markdown(f"##### Feature Importance - {selected_model}")
        
        importance = get_feature_importance(model_result['model'], top_n=15)
        
        if importance:
            tabs_fi = st.tabs(["Negative", "Neutral", "Positive"])
            
            for tab, sentiment in zip(tabs_fi, class_names):
                with tab:
                    fig_fi = plot_feature_importance(importance, sentiment, top_n=15)
                    if fig_fi:
                        st.plotly_chart(fig_fi, use_container_width=True)
                    else:
                        st.warning(f"No feature importance data for {sentiment}")
        else:
            st.info(f"Feature importance analysis not available for {selected_model}")

# TAB 4: ADVANCED CHARTS
with tab4:
    st.markdown("### Advanced Analysis Charts")
    
    # Sentiment by text length
    st.markdown("#### Sentiment Distribution by Text Length")
    
    df_plot = df.copy()
    df_plot['text_len_category'] = pd.cut(
        df_plot['text_cleaned'].str.len(),
        bins=[0, 50, 100, 200, 500, 10000],
        labels=['Very Short', 'Short', 'Medium', 'Long', 'Very Long']
    )
    
    import plotly.express as px
    
    fig_len = px.histogram(
        df_plot,
        x='text_len_category',
        color='sentiment_category',
        barmode='group',
        title='Sentiment Distribution by Text Length Category',
        color_discrete_map={'positive': '#2ecc71', 'neutral': '#95a5a6', 'negative': '#e74c3c'}
    )
    fig_len.update_layout(
        xaxis_title="Text Length Category",
        yaxis_title="Count",
        template="plotly_white"
    )
    st.plotly_chart(fig_len, use_container_width=True)
    
    st.markdown("---")
    
    # Confidence score distribution
    st.markdown("#### Confidence Score Distribution")
    
    fig_conf = px.violin(
        df,
        y='raw_sentiment_score',
        x='sentiment_category',
        color='sentiment_category',
        box=True,
        points="all",
        title='Confidence Score Distribution by Sentiment',
        color_discrete_map={'positive': '#2ecc71', 'neutral': '#95a5a6', 'negative': '#e74c3c'}
    )
    fig_conf.update_layout(
        xaxis_title="Sentiment Category",
        yaxis_title="Confidence Score",
        template="plotly_white"
    )
    st.plotly_chart(fig_conf, use_container_width=True)
    
    st.markdown("---")
    
    # Word count vs confidence
    st.markdown("#### Word Count vs Confidence Score")
    
    df_plot['word_count'] = df_plot['text_cleaned'].str.split().str.len()
    
    fig_scatter = px.scatter(
        df_plot,
        x='word_count',
        y='raw_sentiment_score',
        color='sentiment_category',
        title='Relationship between Word Count and Confidence',
        color_discrete_map={'positive': '#2ecc71', 'neutral': '#95a5a6', 'negative': '#e74c3c'},
        opacity=0.6
    )
    fig_scatter.update_layout(
        xaxis_title="Word Count",
        yaxis_title="Confidence Score",
        template="plotly_white"
    )
    st.plotly_chart(fig_scatter, use_container_width=True)
    
    st.markdown("---")
    
    # Sentiment trend (if timestamp available)
    if 'timestamp' in df.columns or 'date' in df.columns:
        st.markdown("#### Sentiment Trend Over Time")
        
        time_col = 'timestamp' if 'timestamp' in df.columns else 'date'
        df_plot[time_col] = pd.to_datetime(df_plot[time_col], errors='coerce')
        
        df_trend = df_plot.groupby([pd.Grouper(key=time_col, freq='D'), 'sentiment_category']).size().reset_index(name='count')
        
        fig_trend = px.line(
            df_trend,
            x=time_col,
            y='count',
            color='sentiment_category',
            title='Sentiment Trend Over Time',
            color_discrete_map={'positive': '#2ecc71', 'neutral': '#95a5a6', 'negative': '#e74c3c'}
        )
        fig_trend.update_layout(
            xaxis_title="Date",
            yaxis_title="Count",
            template="plotly_white"
        )
        st.plotly_chart(fig_trend, use_container_width=True)

st.markdown("---")
st.caption("💡 Tip: Hover over charts for interactive details. Click the camera icon to download charts as images.")