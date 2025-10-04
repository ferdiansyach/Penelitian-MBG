# ============================================================================
# VISUALIZATION UTILITIES
# Interactive Plotly charts for sentiment analysis
# ============================================================================

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import io
import base64


def plot_sentiment_distribution(df, title="Sentiment Distribution"):
    """
    Create interactive bar chart for sentiment distribution
    """
    sentiment_counts = df['sentiment_category'].value_counts()
    colors_map = {'positive': '#2ecc71', 'neutral': '#95a5a6', 'negative': '#e74c3c'}
    
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=sentiment_counts.index,
        y=sentiment_counts.values,
        text=[f"{v}<br>({v/len(df)*100:.1f}%)" for v in sentiment_counts.values],
        textposition='outside',
        marker_color=[colors_map.get(s, '#3498db') for s in sentiment_counts.index],
        hovertemplate='<b>%{x}</b><br>Count: %{y}<br><extra></extra>'
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title="Sentiment Category",
        yaxis_title="Count",
        template="plotly_white",
        height=500
    )
    
    return fig


def plot_sentiment_pie(df, title="Sentiment Proportion"):
    """
    Create interactive pie chart for sentiment proportion
    """
    sentiment_counts = df['sentiment_category'].value_counts()
    colors_map = {'positive': '#2ecc71', 'neutral': '#95a5a6', 'negative': '#e74c3c'}
    
    fig = go.Figure(data=[go.Pie(
        labels=sentiment_counts.index,
        values=sentiment_counts.values,
        hole=0.4,
        marker=dict(colors=[colors_map.get(s, '#3498db') for s in sentiment_counts.index]),
        textinfo='label+percent',
        hovertemplate='<b>%{label}</b><br>Count: %{value}<br>Percentage: %{percent}<extra></extra>'
    )])
    
    fig.update_layout(
        title=title,
        template="plotly_white",
        height=500
    )
    
    return fig


def plot_confidence_distribution(df):
    """
    Plot confidence level distribution
    """
    confidence_counts = df['confidence_level'].value_counts()
    colors = ['#e74c3c', '#f39c12', '#2ecc71']
    
    fig = go.Figure(data=[go.Bar(
        x=confidence_counts.index,
        y=confidence_counts.values,
        marker_color=colors[:len(confidence_counts)],
        text=[f"{v} ({v/len(df)*100:.1f}%)" for v in confidence_counts.values],
        textposition='outside'
    )])
    
    fig.update_layout(
        title="Confidence Level Distribution",
        xaxis_title="Confidence Level",
        yaxis_title="Count",
        template="plotly_white",
        height=400
    )
    
    return fig


def plot_confusion_matrix(cm, class_names, title="Confusion Matrix"):
    """
    Create interactive heatmap for confusion matrix
    """
    fig = go.Figure(data=go.Heatmap(
        z=cm,
        x=class_names,
        y=class_names,
        colorscale='Blues',
        text=cm,
        texttemplate='%{text}',
        textfont={"size": 16},
        hovertemplate='True: %{y}<br>Predicted: %{x}<br>Count: %{z}<extra></extra>'
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title="Predicted Label",
        yaxis_title="True Label",
        template="plotly_white",
        height=500
    )
    
    return fig


def plot_roc_curves(roc_data, class_names):
    """
    Plot ROC curves for multi-class classification
    """
    fig = go.Figure()
    colors = ['#e74c3c', '#95a5a6', '#2ecc71']
    
    for i, (color, class_name) in enumerate(zip(colors, class_names)):
        fig.add_trace(go.Scatter(
            x=roc_data['fpr'][i],
            y=roc_data['tpr'][i],
            mode='lines',
            name=f'{class_name} (AUC = {roc_data["auc"][i]:.3f})',
            line=dict(color=color, width=2)
        ))
    
    # Add diagonal line
    fig.add_trace(go.Scatter(
        x=[0, 1],
        y=[0, 1],
        mode='lines',
        name='Random Classifier',
        line=dict(color='gray', width=2, dash='dash')
    ))
    
    fig.update_layout(
        title="ROC Curves - Multi-class Classification",
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        template="plotly_white",
        height=600,
        legend=dict(x=0.6, y=0.1)
    )
    
    return fig


def plot_text_length_distribution(df, text_column):
    """
    Plot text length distribution
    """
    df['text_len'] = df[text_column].str.len()
    
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=df['text_len'],
        nbinsx=50,
        marker_color='#3498db',
        hovertemplate='Length: %{x}<br>Count: %{y}<extra></extra>'
    ))
    
    fig.update_layout(
        title="Text Length Distribution",
        xaxis_title="Text Length (characters)",
        yaxis_title="Frequency",
        template="plotly_white",
        height=400
    )
    
    return fig


def plot_word_count_distribution(df, text_column):
    """
    Plot word count distribution
    """
    df['word_cnt'] = df[text_column].str.split().str.len()
    
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=df['word_cnt'],
        nbinsx=30,
        marker_color='#9b59b6',
        hovertemplate='Words: %{x}<br>Count: %{y}<extra></extra>'
    ))
    
    fig.update_layout(
        title="Word Count Distribution",
        xaxis_title="Number of Words",
        yaxis_title="Frequency",
        template="plotly_white",
        height=400
    )
    
    return fig


def generate_wordcloud(text_data, colormap='viridis'):
    """
    Generate word cloud and return as base64 image
    """
    if not text_data or len(text_data.split()) < 10:
        return None
    
    wordcloud = WordCloud(
        width=800,
        height=400,
        background_color='white',
        colormap=colormap,
        max_words=100,
        collocations=False
    ).generate(text_data)
    
    # Convert to image
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    
    # Save to buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    plt.close()
    
    # Convert to base64
    img_base64 = base64.b64encode(buf.read()).decode()
    return f"data:image/png;base64,{img_base64}"


def plot_model_comparison(comparison_df):
    """
    Create model comparison bar chart
    """
    fig = go.Figure()
    
    metrics = ['Accuracy', 'F1 (Macro)', 'Precision', 'Recall']
    colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12']
    
    for metric, color in zip(metrics, colors):
        if metric in comparison_df.columns:
            values = comparison_df[metric].apply(lambda x: float(x) if isinstance(x, str) else x)
            fig.add_trace(go.Bar(
                name=metric,
                x=comparison_df['Model'],
                y=values,
                marker_color=color,
                text=[f"{v:.3f}" for v in values],
                textposition='outside'
            ))
    
    fig.update_layout(
        title="Model Performance Comparison",
        xaxis_title="Model",
        yaxis_title="Score",
        barmode='group',
        template="plotly_white",
        height=500,
        yaxis=dict(range=[0, 1.1])
    )
    
    return fig


def plot_feature_importance(importance_data, class_name, top_n=15):
    """
    Plot feature importance for a specific class
    """
    if importance_data is None or class_name not in importance_data:
        return None
    
    data = importance_data[class_name]
    features = data['features'][:top_n]
    scores = data['scores'][:top_n]
    
    fig = go.Figure(go.Bar(
        x=scores,
        y=features,
        orientation='h',
        marker_color='skyblue',
        text=[f"{s:.3f}" for s in scores],
        textposition='outside'
    ))
    
    fig.update_layout(
        title=f"Top {top_n} Important Features for '{class_name}' Class",
        xaxis_title="Coefficient Score",
        yaxis_title="Feature",
        template="plotly_white",
        height=500,
        yaxis=dict(autorange="reversed")
    )
    
    return fig


def plot_sentiment_by_confidence(df):
    """
    Stacked bar chart of sentiment by confidence level
    """
    crosstab = pd.crosstab(df['confidence_level'], df['sentiment_category'], normalize='index') * 100
    
    fig = go.Figure()
    colors_map = {'positive': '#2ecc71', 'neutral': '#95a5a6', 'negative': '#e74c3c'}
    
    for sentiment in crosstab.columns:
        fig.add_trace(go.Bar(
            name=sentiment,
            x=crosstab.index,
            y=crosstab[sentiment],
            marker_color=colors_map.get(sentiment, '#3498db'),
            text=[f"{v:.1f}%" for v in crosstab[sentiment]],
            textposition='inside'
        ))
    
    fig.update_layout(
        title="Sentiment Distribution by Confidence Level",
        xaxis_title="Confidence Level",
        yaxis_title="Percentage (%)",
        barmode='stack',
        template="plotly_white",
        height=500
    )
    
    return fig