import streamlit as st
import pandas as pd
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from utils.model_training import (
    prepare_data, train_all_models, compare_models, get_feature_importance
)
from utils.visualization import (
    plot_confusion_matrix, plot_model_comparison, plot_roc_curves
)

st.set_page_config(page_title="Model Training", page_icon="🔧", layout="wide")

st.title("🔧 Model Training & Evaluation")
st.markdown("Train machine learning models with hyperparameter optimization")

# Check prerequisites
if st.session_state.df_sentiment is None:
    st.error("⚠️ No sentiment-labeled data available. Please run sentiment analysis first!")
    st.stop()

# Tabs
tab1, tab2, tab3 = st.tabs(["⚙️ Configuration", "🚀 Training", "📊 Results"])

# TAB 1: CONFIGURATION
with tab1:
    st.markdown("### Training Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Model Selection")
        
        available_models = ['SVM', 'KNN']
        selected_models = st.multiselect(
            "Select models to train:",
            available_models,
            default=st.session_state.selected_models,
            help="You can select one or more models"
        )
        st.session_state.selected_models = selected_models
        
        if not selected_models:
            st.warning("⚠️ Please select at least one model")
        
        st.markdown("---")
        st.markdown("#### Analysis Mode")
        
        mode_desc = {
            "Quick Mode": "Reduced parameters, ~2-3 minutes",
            "Full Analysis": "Complete grid search, ~5-15 minutes"
        }
        
        for mode, desc in mode_desc.items():
            is_selected = st.session_state.analysis_mode == mode
            if st.button(
                f"{'✓' if is_selected else '○'} {mode}", 
                key=f"mode_{mode}",
                use_container_width=True,
                type="primary" if is_selected else "secondary"
            ):
                st.session_state.analysis_mode = mode
                st.rerun()
            st.caption(desc)
    
    with col2:
        st.markdown("#### Hyperparameters")
        
        test_size = st.slider(
            "Test Split Ratio",
            min_value=0.1,
            max_value=0.4,
            value=st.session_state.model_params['test_size'],
            step=0.05,
            help="Portion of data used for testing"
        )
        
        cv_folds = st.slider(
            "Cross-Validation Folds",
            min_value=3,
            max_value=10,
            value=st.session_state.model_params['cv_folds'],
            step=1,
            help="Number of folds for cross-validation"
        )
        
        smote_k = st.slider(
            "SMOTE K-Neighbors",
            min_value=1,
            max_value=10,
            value=st.session_state.model_params['smote_k_neighbors'],
            step=1,
            help="Number of neighbors for SMOTE oversampling"
        )
        
        random_state = st.number_input(
            "Random State",
            min_value=1,
            max_value=999,
            value=st.session_state.model_params['random_state'],
            help="Seed for reproducibility"
        )
        
        # Update session state
        st.session_state.model_params.update({
            'test_size': test_size,
            'cv_folds': cv_folds,
            'smote_k_neighbors': smote_k,
            'random_state': random_state
        })
        
        st.markdown("---")
        st.info(f"""
        **Current Configuration:**
        - Models: {', '.join(selected_models) if selected_models else 'None'}
        - Mode: {st.session_state.analysis_mode}
        - Dataset: {len(st.session_state.df_sentiment)} samples
        - Train/Test: {int((1-test_size)*100)}%/{int(test_size*100)}%
        """)

# TAB 2: TRAINING
with tab2:
    st.markdown("### 🚀 Model Training")
    
    if not st.session_state.selected_models:
        st.warning("⚠️ Please select at least one model in the Configuration tab")
    else:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("#### Ready to Train")
            
            st.info(f"""
            **Training Setup:**
            - Selected models: {', '.join(st.session_state.selected_models)}
            - Analysis mode: {st.session_state.analysis_mode}
            - Cross-validation: {st.session_state.model_params['cv_folds']} folds
            - SMOTE oversampling: {st.session_state.model_params['smote_k_neighbors']} neighbors
            
            **Estimated time:** 
            {'~2-3 minutes (Quick Mode)' if st.session_state.analysis_mode == 'Quick Mode' else '~5-15 minutes (Full Analysis)'}
            """)
            
            if st.button("🚀 Start Training", type="primary", use_container_width=True):
                
                # Prepare data
                with st.spinner("Preparing data..."):
                    X_train, X_test, y_train, y_test, sentiment_map = prepare_data(
                        st.session_state.df_sentiment,
                        'text_cleaned',
                        test_size=st.session_state.model_params['test_size'],
                        random_state=st.session_state.model_params['random_state']
                    )
                    
                    st.session_state.X_train = X_train
                    st.session_state.X_test = X_test
                    st.session_state.y_train = y_train
                    st.session_state.y_test = y_test
                    st.session_state.sentiment_map = sentiment_map
                
                st.success("✓ Data prepared successfully!")
                
                # Training progress
                st.markdown("#### Training Progress")
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                def update_progress(current, total, message):
                    progress = current / total
                    progress_bar.progress(progress)
                    status_text.text(f"{message} ({current}/{total})")
                
                # Train models
                results = train_all_models(
                    X_train, X_test, y_train, y_test,
                    st.session_state.selected_models,
                    analysis_mode=st.session_state.analysis_mode,
                    cv_folds=st.session_state.model_params['cv_folds'],
                    smote_k=st.session_state.model_params['smote_k_neighbors'],
                    progress_callback=update_progress
                )
                
                st.session_state.trained_models = results
                
                # Find best model
                best_model_name = max(results.keys(), 
                                     key=lambda x: results[x]['metrics']['accuracy'])
                st.session_state.best_model = best_model_name
                
                progress_bar.empty()
                status_text.empty()
                
                st.success("✓ All models trained successfully!")
                st.balloons()
                
                # Show quick summary
                st.markdown("#### Quick Summary")
                comparison_df = compare_models(results)
                st.dataframe(comparison_df, use_container_width=True)
                
                st.info(f"🏆 Best performing model: **{best_model_name}** "
                       f"(Accuracy: {results[best_model_name]['metrics']['accuracy']:.4f})")
        
        with col2:
            st.markdown("#### Training Details")
            
            st.markdown("""
            **What happens during training:**
            
            1. **Data Split**
               - Stratified train/test split
               - Preserves class distribution
            
            2. **Feature Extraction**
               - TF-IDF vectorization
               - N-gram combinations
            
            3. **SMOTE Oversampling**
               - Balance class distribution
               - Synthetic sample generation
            
            4. **GridSearchCV**
               - Test multiple parameters
               - Cross-validation
               - Select best configuration
            
            5. **Evaluation**
               - Test set predictions
               - Comprehensive metrics
               - Confusion matrix
               - ROC curves
            """)

# TAB 3: RESULTS
with tab3:
    st.markdown("### 📊 Training Results")
    
    if not st.session_state.trained_models:
        st.warning("⚠️ No trained models yet. Please train models first!")
    else:
        results = st.session_state.trained_models
        
        # Model selector
        st.markdown("#### Model Comparison")
        
        comparison_df = compare_models(results)
        st.dataframe(comparison_df, use_container_width=True, height=150)
        
        fig_comparison = plot_model_comparison(comparison_df)
        st.plotly_chart(fig_comparison, use_container_width=True)
        
        st.markdown("---")
        
        # Detailed results for each model
        st.markdown("#### Detailed Model Results")
        
        selected_model = st.selectbox(
            "Select model for detailed analysis:",
            list(results.keys())
        )
        
        model_result = results[selected_model]
        metrics = model_result['metrics']
        
        # Metrics overview
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Accuracy", f"{metrics['accuracy']:.4f}")
        with col2:
            st.metric("F1 (Macro)", f"{metrics['f1_macro']:.4f}")
        with col3:
            st.metric("Precision", f"{metrics['precision']:.4f}")
        with col4:
            st.metric("Recall", f"{metrics['recall']:.4f}")
        
        # Best parameters
        with st.expander("🎯 Best Hyperparameters", expanded=True):
            st.json(model_result['best_params'])
        
        # Confusion Matrix
        st.markdown("#### Confusion Matrix")
        class_names = ['negative', 'neutral', 'positive']
        fig_cm = plot_confusion_matrix(
            metrics['confusion_matrix'], 
            class_names,
            title=f"Confusion Matrix - {selected_model}"
        )
        st.plotly_chart(fig_cm, use_container_width=True)
        
        # Classification Report
        with st.expander("📋 Detailed Classification Report"):
            report_df = pd.DataFrame(metrics['classification_report']).transpose()
            st.dataframe(report_df, use_container_width=True)
        
        # ROC Curves
        st.markdown("#### ROC Curves")
        fig_roc = plot_roc_curves(metrics['roc_data'], class_names)
        st.plotly_chart(fig_roc, use_container_width=True)
        
        # Feature Importance (if available)
        st.markdown("#### Feature Importance")
        importance = get_feature_importance(model_result['model'], top_n=15)
        
        if importance:
            from utils.visualization import plot_feature_importance
            
            tabs_fi = st.tabs(["Negative", "Neutral", "Positive"])
            
            for tab, sentiment in zip(tabs_fi, class_names):
                with tab:
                    fig_fi = plot_feature_importance(importance, sentiment)
                    if fig_fi:
                        st.plotly_chart(fig_fi, use_container_width=True)
        else:
            st.info(f"Feature importance not available for {selected_model}")

st.markdown("---")
st.markdown("**Next Step:** Go to **📈 Visualizations** page for comprehensive visual analysis")