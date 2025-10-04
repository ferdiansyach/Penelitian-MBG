import streamlit as st
import pandas as pd
import numpy as np
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from utils.model_training import get_misclassified_samples, get_feature_importance

st.set_page_config(page_title="Error Analysis", page_icon="🔍", layout="wide")

st.title("🔍 Error Analysis & Model Insights")
st.markdown("Deep dive into model performance, misclassifications, and improvement opportunities")

# Check prerequisites
if not st.session_state.trained_models:
    st.error("⚠️ No trained models available. Please train models first in the Model Training page.")
    st.stop()

if st.session_state.X_test is None or st.session_state.y_test is None:
    st.error("⚠️ Test data not available. Please retrain models.")
    st.stop()

# Tabs
tab1, tab2, tab3, tab4 = st.tabs(["❌ Misclassifications", "📊 Error Patterns", "🎯 Model Comparison", "💡 Insights"])

# TAB 1: MISCLASSIFICATIONS
with tab1:
    st.markdown("### Misclassified Samples Analysis")
    
    # Model selector
    selected_model = st.selectbox(
        "Select model to analyze:",
        list(st.session_state.trained_models.keys()),
        key="error_model_select"
    )
    
    model_result = st.session_state.trained_models[selected_model]
    predictions = model_result['metrics']['predictions']
    
    # Get misclassified samples
    error_df = get_misclassified_samples(
        st.session_state.X_test,
        st.session_state.y_test,
        predictions,
        st.session_state.sentiment_map,
        max_samples=100
    )
    
    if error_df.empty:
        st.success("🎉 Perfect predictions! No misclassifications found.")
    else:
        # Error statistics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Errors", len(error_df))
        with col2:
            error_rate = len(error_df) / len(st.session_state.y_test) * 100
            st.metric("Error Rate", f"{error_rate:.2f}%")
        with col3:
            accuracy = model_result['metrics']['accuracy']
            st.metric("Accuracy", f"{accuracy:.4f}")
        with col4:
            st.metric("Correct Predictions", len(st.session_state.y_test) - len(error_df))
        
        st.markdown("---")
        
        # Filter options
        col1, col2 = st.columns(2)
        
        with col1:
            true_label_filter = st.multiselect(
                "Filter by True Label:",
                ['negative', 'neutral', 'positive'],
                default=['negative', 'neutral', 'positive']
            )
        
        with col2:
            pred_label_filter = st.multiselect(
                "Filter by Predicted Label:",
                ['negative', 'neutral', 'positive'],
                default=['negative', 'neutral', 'positive']
            )
        
        # Apply filters
        filtered_errors = error_df[
            (error_df['True Label'].isin(true_label_filter)) &
            (error_df['Predicted Label'].isin(pred_label_filter))
        ]
        
        st.markdown(f"#### Showing {len(filtered_errors)} of {len(error_df)} Errors")
        
        # Display errors with color coding
        def highlight_errors(row):
            if row['True Label'] == 'negative':
                color = 'background-color: #ffe6e6'
            elif row['True Label'] == 'positive':
                color = 'background-color: #e6ffe6'
            else:
                color = 'background-color: #e6e6ff'
            return [color] * len(row)
        
        st.dataframe(
            filtered_errors.style.apply(highlight_errors, axis=1),
            use_container_width=True,
            height=500
        )
        
        # Download errors
        csv = filtered_errors.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Misclassifications",
            data=csv,
            file_name=f"errors_{selected_model}_{pd.Timestamp.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )

# TAB 2: ERROR PATTERNS
with tab2:
    st.markdown("### Error Pattern Analysis")
    
    selected_model = st.selectbox(
        "Select model to analyze:",
        list(st.session_state.trained_models.keys()),
        key="pattern_model_select"
    )
    
    model_result = st.session_state.trained_models[selected_model]
    predictions = model_result['metrics']['predictions']
    cm = model_result['metrics']['confusion_matrix']
    
    # Confusion matrix analysis
    st.markdown("#### Confusion Matrix Breakdown")
    
    class_names = ['negative', 'neutral', 'positive']
    
    # Create detailed confusion matrix dataframe
    cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)
    cm_df.index.name = 'True Label'
    cm_df.columns.name = 'Predicted Label'
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.dataframe(cm_df, use_container_width=True)
    
    with col2:
        st.markdown("**Key Observations:**")
        
        # Find most confused pairs
        cm_copy = cm.copy()
        np.fill_diagonal(cm_copy, 0)
        max_confusion_idx = np.unravel_index(cm_copy.argmax(), cm_copy.shape)
        
        st.write(f"• Most confused: **{class_names[max_confusion_idx[0]]}** → **{class_names[max_confusion_idx[1]]}**")
        st.write(f"  Count: {cm_copy[max_confusion_idx]}")
        
        # Per-class accuracy
        st.markdown("**Per-Class Accuracy:**")
        for i, class_name in enumerate(class_names):
            class_correct = cm[i, i]
            class_total = cm[i, :].sum()
            class_acc = class_correct / class_total * 100 if class_total > 0 else 0
            st.write(f"• {class_name}: {class_acc:.1f}%")
    
    st.markdown("---")
    
    # Error distribution by class
    st.markdown("#### Error Distribution by Class")
    
    import plotly.graph_objects as go
    
    # Calculate errors per class
    error_data = []
    for i, true_label in enumerate(class_names):
        for j, pred_label in enumerate(class_names):
            if i != j:  # Only errors
                error_data.append({
                    'True': true_label,
                    'Predicted': pred_label,
                    'Count': cm[i, j]
                })
    
    error_df_plot = pd.DataFrame(error_data)
    
    fig = go.Figure()
    
    for pred_label in class_names:
        data = error_df_plot[error_df_plot['Predicted'] == pred_label]
        fig.add_trace(go.Bar(
            name=f'Predicted as {pred_label}',
            x=data['True'],
            y=data['Count'],
            text=data['Count'],
            textposition='auto'
        ))
    
    fig.update_layout(
        title='Misclassification Patterns',
        xaxis_title='True Label',
        yaxis_title='Number of Errors',
        barmode='group',
        template='plotly_white',
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # Text length analysis of errors
    st.markdown("#### Error Analysis by Text Characteristics")
    
    error_df = get_misclassified_samples(
        st.session_state.X_test,
        st.session_state.y_test,
        predictions,
        st.session_state.sentiment_map,
        max_samples=1000
    )
    
    if not error_df.empty:
        error_df['text_length'] = error_df['Text'].str.len()
        error_df['word_count'] = error_df['Text'].str.split().str.len()
        
        # Get correct predictions for comparison
        correct_mask = st.session_state.y_test == predictions
        correct_texts = st.session_state.X_test[correct_mask]
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Avg Length (Errors)", f"{error_df['text_length'].mean():.0f} chars")
            st.metric("Avg Words (Errors)", f"{error_df['word_count'].mean():.1f}")
        
        with col2:
            st.metric("Avg Length (Correct)", f"{correct_texts.str.len().mean():.0f} chars")
            st.metric("Avg Words (Correct)", f"{correct_texts.str.split().str.len().mean():.1f}")
        
        # Distribution comparison
        import plotly.express as px
        
        comparison_df = pd.DataFrame({
            'Length': list(error_df['text_length']) + list(correct_texts.str.len()),
            'Type': ['Error']*len(error_df) + ['Correct']*len(correct_texts)
        })
        
        fig_dist = px.histogram(
            comparison_df,
            x='Length',
            color='Type',
            barmode='overlay',
            title='Text Length Distribution: Errors vs Correct Predictions',
            nbins=30,
            opacity=0.7
        )
        
        st.plotly_chart(fig_dist, use_container_width=True)

# TAB 3: MODEL COMPARISON
with tab3:
    st.markdown("### Model-to-Model Comparison")
    
    if len(st.session_state.trained_models) < 2:
        st.warning("⚠️ Need at least 2 trained models for comparison. Currently have: " + 
                  str(len(st.session_state.trained_models)))
    else:
        # Select two models to compare
        col1, col2 = st.columns(2)
        
        model_names = list(st.session_state.trained_models.keys())
        
        with col1:
            model1 = st.selectbox("Model 1:", model_names, key="compare_model1")
        
        with col2:
            model2 = st.selectbox("Model 2:", [m for m in model_names if m != model1], key="compare_model2")
        
        # Get predictions from both models
        pred1 = st.session_state.trained_models[model1]['metrics']['predictions']
        pred2 = st.session_state.trained_models[model2]['metrics']['predictions']
        
        # Agreement analysis
        agreement = (pred1 == pred2).sum()
        disagreement = len(pred1) - agreement
        
        st.markdown("#### Agreement Analysis")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Agreements", agreement)
        with col2:
            st.metric("Disagreements", disagreement)
        with col3:
            agreement_rate = agreement / len(pred1) * 100
            st.metric("Agreement Rate", f"{agreement_rate:.1f}%")
        
        st.markdown("---")
        
        # Find cases where models disagree
        st.markdown("#### Cases Where Models Disagree")
        
        disagree_mask = pred1 != pred2
        disagree_indices = np.where(disagree_mask)[0]
        
        if len(disagree_indices) > 0:
            reverse_map = {v: k for k, v in st.session_state.sentiment_map.items()}
            
            disagree_df = pd.DataFrame({
                'Text': st.session_state.X_test.iloc[disagree_indices].values,
                'True Label': [reverse_map[y] for y in st.session_state.y_test.iloc[disagree_indices].values],
                f'{model1} Prediction': [reverse_map[p] for p in pred1[disagree_indices]],
                f'{model2} Prediction': [reverse_map[p] for p in pred2[disagree_indices]]
            })
            
            # Add correctness indicators
            disagree_df[f'{model1} Correct'] = [
                '✓' if reverse_map[pred1[i]] == reverse_map[st.session_state.y_test.iloc[i]] else '✗'
                for i in disagree_indices
            ]
            disagree_df[f'{model2} Correct'] = [
                '✓' if reverse_map[pred2[i]] == reverse_map[st.session_state.y_test.iloc[i]] else '✗'
                for i in disagree_indices
            ]
            
            st.dataframe(disagree_df.head(50), use_container_width=True, height=400)
            
            # Download disagreements
            csv = disagree_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Disagreements",
                data=csv,
                file_name=f"disagreements_{model1}_vs_{model2}.csv",
                mime="text/csv"
            )
        else:
            st.success("🎉 Both models agree on all predictions!")
        
        st.markdown("---")
        
        # Performance comparison
        st.markdown("#### Performance Comparison")
        
        metrics1 = st.session_state.trained_models[model1]['metrics']
        metrics2 = st.session_state.trained_models[model2]['metrics']
        
        comparison_data = {
            'Metric': ['Accuracy', 'F1 (Macro)', 'F1 (Weighted)', 'Precision', 'Recall'],
            model1: [
                metrics1['accuracy'],
                metrics1['f1_macro'],
                metrics1['f1_weighted'],
                metrics1['precision'],
                metrics1['recall']
            ],
            model2: [
                metrics2['accuracy'],
                metrics2['f1_macro'],
                metrics2['f1_weighted'],
                metrics2['precision'],
                metrics2['recall']
            ]
        }
        
        comp_df = pd.DataFrame(comparison_data)
        comp_df['Difference'] = comp_df[model1] - comp_df[model2]
        comp_df['Winner'] = comp_df['Difference'].apply(
            lambda x: model1 if x > 0 else (model2 if x < 0 else 'Tie')
        )
        
        # Format for display
        for col in [model1, model2, 'Difference']:
            comp_df[col] = comp_df[col].apply(lambda x: f"{x:.4f}")
        
        st.dataframe(comp_df, use_container_width=True)

# TAB 4: INSIGHTS & RECOMMENDATIONS
with tab4:
    st.markdown("### Insights & Recommendations")
    
    # Analyze all models
    best_model_name = max(
        st.session_state.trained_models.keys(),
        key=lambda x: st.session_state.trained_models[x]['metrics']['accuracy']
    )
    
    best_model_result = st.session_state.trained_models[best_model_name]
    best_metrics = best_model_result['metrics']
    
    st.markdown(f"#### Overall Best Model: **{best_model_name}**")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Accuracy", f"{best_metrics['accuracy']:.4f}")
    with col2:
        st.metric("F1 Score", f"{best_metrics['f1_macro']:.4f}")
    with col3:
        st.metric("Precision", f"{best_metrics['precision']:.4f}")
    with col4:
        st.metric("Recall", f"{best_metrics['recall']:.4f}")
    
    st.markdown("---")
    
    # Generate insights
    st.markdown("#### 🔍 Key Findings")
    
    insights = []
    
    # Check class imbalance in errors
    cm = best_metrics['confusion_matrix']
    class_names = ['negative', 'neutral', 'positive']
    
    # Per-class performance
    for i, class_name in enumerate(class_names):
        class_correct = cm[i, i]
        class_total = cm[i, :].sum()
        if class_total > 0:
            class_acc = class_correct / class_total * 100
            if class_acc < 70:
                insights.append(f"⚠️ **Low performance on {class_name} class** ({class_acc:.1f}% accuracy) - Consider collecting more {class_name} samples")
            elif class_acc > 90:
                insights.append(f"✓ **Excellent performance on {class_name} class** ({class_acc:.1f}% accuracy)")
    
    # Check most confused pairs
    cm_copy = cm.copy()
    np.fill_diagonal(cm_copy, 0)
    max_confusion_idx = np.unravel_index(cm_copy.argmax(), cm_copy.shape)
    confusion_count = cm_copy[max_confusion_idx]
    
    if confusion_count > 10:
        insights.append(
            f"⚠️ **High confusion between {class_names[max_confusion_idx[0]]} and {class_names[max_confusion_idx[1]]}** "
            f"({confusion_count} cases) - Review veto rules and keywords"
        )
    
    # Check overall accuracy
    if best_metrics['accuracy'] < 0.7:
        insights.append("⚠️ **Overall accuracy below 70%** - Consider using Full Analysis mode or collecting more diverse data")
    elif best_metrics['accuracy'] > 0.85:
        insights.append("✓ **Excellent overall accuracy** - Model is performing well")
    
    # Check F1 score
    if best_metrics['f1_macro'] < best_metrics['accuracy'] - 0.1:
        insights.append("⚠️ **F1 score significantly lower than accuracy** - Class imbalance detected, SMOTE may need adjustment")
    
    # Display insights
    for insight in insights:
        if '⚠️' in insight:
            st.warning(insight)
        else:
            st.success(insight)
    
    st.markdown("---")
    
    # Recommendations
    st.markdown("#### 💡 Recommendations for Improvement")
    
    recommendations = []
    
    # Data-related recommendations
    error_df = get_misclassified_samples(
        st.session_state.X_test,
        st.session_state.y_test,
        best_metrics['predictions'],
        st.session_state.sentiment_map,
        max_samples=1000
    )
    
    if not error_df.empty:
        # Check if errors have specific patterns
        error_df['text_length'] = error_df['Text'].str.len()
        avg_error_length = error_df['text_length'].mean()
        
        correct_mask = st.session_state.y_test == best_metrics['predictions']
        correct_texts = st.session_state.X_test[correct_mask]
        avg_correct_length = correct_texts.str.len().mean()
        
        if avg_error_length < avg_correct_length * 0.7:
            recommendations.append({
                'Category': 'Data Quality',
                'Issue': 'Errors occur more on shorter texts',
                'Recommendation': 'Consider filtering texts below a minimum length or collecting more varied short text samples'
            })
        
        if avg_error_length > avg_correct_length * 1.3:
            recommendations.append({
                'Category': 'Data Quality',
                'Issue': 'Errors occur more on longer texts',
                'Recommendation': 'Model may struggle with complex sentences - consider text segmentation'
            })
    
    # Model-related recommendations
    if best_metrics['accuracy'] < 0.75:
        recommendations.append({
            'Category': 'Model Training',
            'Issue': 'Suboptimal accuracy',
            'Recommendation': 'Try Full Analysis mode with extended hyperparameter search'
        })
    
    # Check if Quick Mode was used
    if st.session_state.analysis_mode == 'Quick Mode':
        recommendations.append({
            'Category': 'Model Training',
            'Issue': 'Quick Mode used',
            'Recommendation': 'For production use, consider Full Analysis mode for optimal hyperparameters'
        })
    
    # Rule-based recommendations
    cm_copy = cm.copy()
    np.fill_diagonal(cm_copy, 0)
    if cm_copy.sum() > len(st.session_state.y_test) * 0.3:
        recommendations.append({
            'Category': 'Sentiment Rules',
            'Issue': 'High misclassification rate',
            'Recommendation': 'Review and expand veto words list, adjust keyword lists in Sentiment Analysis page'
        })
    
    # Dataset recommendations
    if len(st.session_state.df_sentiment) < 500:
        recommendations.append({
            'Category': 'Data Collection',
            'Issue': 'Small dataset',
            'Recommendation': 'Collect more data (target: 1000+ samples) for more robust model training'
        })
    
    # Feature engineering recommendations
    recommendations.append({
        'Category': 'Feature Engineering',
        'Issue': 'Using only TF-IDF features',
        'Recommendation': 'Consider adding sentiment lexicon scores, emoji counts, or other linguistic features'
    })
    
    # Display recommendations
    if recommendations:
        rec_df = pd.DataFrame(recommendations)
        
        # Color code by category
        def color_category(row):
            colors = {
                'Data Quality': 'background-color: #fff3cd',
                'Model Training': 'background-color: #d1ecf1',
                'Sentiment Rules': 'background-color: #f8d7da',
                'Data Collection': 'background-color: #d4edda',
                'Feature Engineering': 'background-color: #e2e3e5'
            }
            color = colors.get(row['Category'], '')
            return [color] * len(row)
        
        st.dataframe(
            rec_df.style.apply(color_category, axis=1),
            use_container_width=True,
            height=300
        )
    else:
        st.success("No specific recommendations - model is performing well!")
    
    st.markdown("---")
    
    # Action items
    st.markdown("#### 📋 Suggested Action Items")
    
    action_items = []
    
    # Priority 1: Critical issues
    if best_metrics['accuracy'] < 0.7:
        action_items.append("🔴 **HIGH PRIORITY**: Improve model accuracy (currently below 70%)")
    
    # Priority 2: Important improvements
    if cm_copy.max() > 20:
        action_items.append("🟠 **MEDIUM PRIORITY**: Address confusion between specific sentiment pairs")
    
    if st.session_state.analysis_mode == 'Quick Mode':
        action_items.append("🟠 **MEDIUM PRIORITY**: Run Full Analysis mode for production deployment")
    
    # Priority 3: Nice to have
    action_items.append("🟢 **LOW PRIORITY**: Regularly update veto words and keyword lists based on new patterns")
    action_items.append("🟢 **LOW PRIORITY**: Monitor performance on new data and retrain periodically")
    
    for item in action_items:
        if '🔴' in item:
            st.error(item)
        elif '🟠' in item:
            st.warning(item)
        else:
            st.info(item)
    
    st.markdown("---")
    
    # Export analysis
    st.markdown("#### 📥 Export Error Analysis")
    
    if st.button("📄 Generate Error Analysis Report", use_container_width=True):
        # Compile all error analysis into a text report
        report_lines = [
            "="*70,
            "ERROR ANALYSIS REPORT",
            "="*70,
            f"\nGenerated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Best Model: {best_model_name}",
            f"Overall Accuracy: {best_metrics['accuracy']:.4f}",
            f"F1 Score: {best_metrics['f1_macro']:.4f}",
            "\n" + "="*70,
            "KEY FINDINGS",
            "="*70
        ]
        
        for insight in insights:
            report_lines.append(insight.replace('⚠️', '[WARNING]').replace('✓', '[SUCCESS]'))
        
        report_lines.extend([
            "\n" + "="*70,
            "RECOMMENDATIONS",
            "="*70
        ])
        
        for rec in recommendations:
            report_lines.append(f"\n[{rec['Category']}]")
            report_lines.append(f"Issue: {rec['Issue']}")
            report_lines.append(f"Recommendation: {rec['Recommendation']}")
        
        report_lines.extend([
            "\n" + "="*70,
            "ACTION ITEMS",
            "="*70
        ])
        
        for item in action_items:
            report_lines.append(item.replace('🔴', '[HIGH]').replace('🟠', '[MEDIUM]').replace('🟢', '[LOW]'))
        
        report_text = "\n".join(report_lines)
        
        st.download_button(
            label="📥 Download Report (TXT)",
            data=report_text,
            file_name=f"error_analysis_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.txt",
            mime="text/plain"
        )

st.markdown("---")
st.caption("💡 Use these insights to iterate on your model and improve performance systematically")