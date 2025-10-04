import streamlit as st
import pandas as pd
import pickle
import sys
from pathlib import Path
from datetime import datetime
import io

sys.path.append(str(Path(__file__).parent.parent))

from utils.sentiment_analysis import get_sentiment_distribution

st.set_page_config(page_title="Export & Reports", page_icon="💾", layout="wide")

st.title("💾 Export & Reports")
st.markdown("Download analysis results, trained models, and comprehensive reports")

# Check what's available
has_data = st.session_state.df_original is not None
has_cleaned = st.session_state.df_cleaned is not None
has_sentiment = st.session_state.df_sentiment is not None
has_models = bool(st.session_state.trained_models)

# Status overview
st.markdown("### 📊 Export Status")

col1, col2, col3, col4 = st.columns(4)

with col1:
    if has_data:
        st.success("✓ Original Data")
    else:
        st.error("✗ Original Data")

with col2:
    if has_cleaned:
        st.success("✓ Cleaned Data")
    else:
        st.error("✗ Cleaned Data")

with col3:
    if has_sentiment:
        st.success("✓ Sentiment Results")
    else:
        st.error("✗ Sentiment Results")

with col4:
    if has_models:
        st.success(f"✓ {len(st.session_state.trained_models)} Models")
    else:
        st.error("✗ Trained Models")

st.markdown("---")

# Tabs for different export types
tab1, tab2, tab3, tab4 = st.tabs(["📥 Data Export", "🤖 Model Export", "📄 PDF Report", "📊 Visualizations"])

# TAB 1: DATA EXPORT
with tab1:
    st.markdown("### 📥 Export Data Files")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Original Data")
        if has_data:
            csv_original = st.session_state.df_original.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Original Data",
                data=csv_original,
                file_name=f"original_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True
            )
            st.info(f"Rows: {len(st.session_state.df_original)}")
        else:
            st.warning("No original data available")
        
        st.markdown("#### Cleaned Data")
        if has_cleaned:
            csv_cleaned = st.session_state.df_cleaned.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Cleaned Data",
                data=csv_cleaned,
                file_name=f"cleaned_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True
            )
            st.info(f"Rows: {len(st.session_state.df_cleaned)}")
        else:
            st.warning("No cleaned data available")
    
    with col2:
        st.markdown("#### Sentiment Results")
        if has_sentiment:
            # Full results
            csv_sentiment = st.session_state.df_sentiment.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Full Results",
                data=csv_sentiment,
                file_name=f"sentiment_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True
            )
            
            # Summary only
            summary_cols = ['text_cleaned', 'sentiment_category', 'confidence_level', 'raw_sentiment_score']
            if all(col in st.session_state.df_sentiment.columns for col in summary_cols):
                csv_summary = st.session_state.df_sentiment[summary_cols].to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Download Summary Only",
                    data=csv_summary,
                    file_name=f"sentiment_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                    use_container_width=True
                )
            
            st.info(f"Rows: {len(st.session_state.df_sentiment)}")
        else:
            st.warning("No sentiment results available")
        
        st.markdown("#### Custom Dictionary")
        if st.session_state.normalization_dict:
            dict_text = "\n".join([f"{k}:{v}" for k, v in st.session_state.normalization_dict.items()])
            st.download_button(
                label="📥 Download Normalization Dict",
                data=dict_text,
                file_name="normalization_dictionary.txt",
                mime="text/plain",
                use_container_width=True
            )
            st.info(f"Entries: {len(st.session_state.normalization_dict)}")

# TAB 2: MODEL EXPORT
with tab2:
    st.markdown("### 🤖 Export Trained Models")
    
    if not has_models:
        st.warning("⚠️ No trained models available. Train models first in the Model Training page.")
    else:
        st.info(f"Found {len(st.session_state.trained_models)} trained model(s)")
        
        for model_name, model_data in st.session_state.trained_models.items():
            with st.expander(f"📦 {model_name} Model", expanded=True):
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.markdown(f"**Performance:**")
                    metrics = model_data['metrics']
                    st.write(f"- Accuracy: {metrics['accuracy']:.4f}")
                    st.write(f"- F1 Score: {metrics['f1_macro']:.4f}")
                    st.write(f"- Precision: {metrics['precision']:.4f}")
                    st.write(f"- Recall: {metrics['recall']:.4f}")
                    
                    st.markdown(f"**Best Parameters:**")
                    st.json(model_data['best_params'])
                
                with col2:
                    # Export model
                    model_bytes = io.BytesIO()
                    pickle.dump(model_data['model'], model_bytes)
                    model_bytes.seek(0)
                    
                    st.download_button(
                        label=f"📥 Download {model_name}",
                        data=model_bytes,
                        file_name=f"{model_name.lower()}_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl",
                        mime="application/octet-stream",
                        use_container_width=True
                    )
                    
                    st.caption("Load with: `pickle.load(file)`")
        
        st.markdown("---")
        
        # Export all models together
        st.markdown("#### Export All Models")
        all_models_bytes = io.BytesIO()
        pickle.dump(st.session_state.trained_models, all_models_bytes)
        all_models_bytes.seek(0)
        
        st.download_button(
            label="📥 Download All Models (Bundle)",
            data=all_models_bytes,
            file_name=f"all_models_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl",
            mime="application/octet-stream",
            use_container_width=True
        )

# TAB 3: PDF REPORT
with tab3:
    st.markdown("### 📄 Generate PDF Report")
    
    if not has_sentiment:
        st.warning("⚠️ No sentiment results available. Complete sentiment analysis first.")
    else:
        st.markdown("#### Report Configuration")
        
        col1, col2 = st.columns(2)
        
        with col1:
            report_title = st.text_input(
                "Report Title",
                value="MBG Sentiment Analysis Report"
            )
            
            report_author = st.text_input(
                "Author/Organization",
                value="Sentiment Analysis Team"
            )
            
            include_sections = st.multiselect(
                "Include Sections:",
                [
                    "Executive Summary",
                    "Data Overview",
                    "Sentiment Distribution",
                    "Model Performance",
                    "Sample Results",
                    "Methodology"
                ],
                default=[
                    "Executive Summary",
                    "Data Overview",
                    "Sentiment Distribution",
                    "Model Performance"
                ]
            )
        
        with col2:
            st.markdown("#### Report Preview")
            
            dist = get_sentiment_distribution(st.session_state.df_sentiment)
            
            st.info(f"""
            **Report will include:**
            - Title: {report_title}
            - Author: {report_author}
            - Total Samples: {len(st.session_state.df_sentiment)}
            - Negative: {dist.get('negative', {}).get('percentage', 0)}%
            - Neutral: {dist.get('neutral', {}).get('percentage', 0)}%
            - Positive: {dist.get('positive', {}).get('percentage', 0)}%
            - Models Trained: {len(st.session_state.trained_models) if has_models else 0}
            - Sections: {len(include_sections)}
            """)
        
        if st.button("📄 Generate PDF Report", type="primary", use_container_width=True):
            with st.spinner("Generating PDF report..."):
                try:
                    from reportlab.lib.pagesizes import letter, A4
                    from reportlab.lib import colors
                    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
                    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
                    from reportlab.lib.units import inch
                    
                    # Create PDF in memory
                    pdf_buffer = io.BytesIO()
                    doc = SimpleDocTemplate(pdf_buffer, pagesize=letter)
                    story = []
                    styles = getSampleStyleSheet()
                    
                    # Title
                    title_style = ParagraphStyle(
                        'CustomTitle',
                        parent=styles['Heading1'],
                        fontSize=24,
                        textColor=colors.HexColor('#1f2937'),
                        spaceAfter=30,
                        alignment=1  # Center
                    )
                    story.append(Paragraph(report_title, title_style))
                    story.append(Spacer(1, 0.2*inch))
                    
                    # Metadata
                    story.append(Paragraph(f"<b>Author:</b> {report_author}", styles['Normal']))
                    story.append(Paragraph(f"<b>Generated:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles['Normal']))
                    story.append(Spacer(1, 0.3*inch))
                    
                    # Executive Summary
                    if "Executive Summary" in include_sections:
                        story.append(Paragraph("<b>Executive Summary</b>", styles['Heading2']))
                        summary_text = f"""
                        This report presents the results of sentiment analysis on {len(st.session_state.df_sentiment)} 
                        text samples regarding the MBG (Makan Bergizi Gratis) program. The analysis used BERT-based 
                        natural language processing combined with custom rule-based refinement. 
                        Results show {dist.get('positive', {}).get('percentage', 0):.1f}% positive sentiment, 
                        {dist.get('neutral', {}).get('percentage', 0):.1f}% neutral, and 
                        {dist.get('negative', {}).get('percentage', 0):.1f}% negative sentiment.
                        """
                        story.append(Paragraph(summary_text, styles['Normal']))
                        story.append(Spacer(1, 0.2*inch))
                    
                    # Data Overview
                    if "Data Overview" in include_sections:
                        story.append(Paragraph("<b>Data Overview</b>", styles['Heading2']))
                        data_table = [
                            ['Metric', 'Value'],
                            ['Total Samples', str(len(st.session_state.df_sentiment))],
                            ['Negative Count', str(dist.get('negative', {}).get('count', 0))],
                            ['Neutral Count', str(dist.get('neutral', {}).get('count', 0))],
                            ['Positive Count', str(dist.get('positive', {}).get('count', 0))],
                            ['Average Confidence', f"{st.session_state.df_sentiment['raw_sentiment_score'].mean():.3f}"]
                        ]
                        t = Table(data_table, colWidths=[3*inch, 2*inch])
                        t.setStyle(TableStyle([
                            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                            ('FONTSIZE', (0, 0), (-1, 0), 12),
                            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                            ('GRID', (0, 0), (-1, -1), 1, colors.black)
                        ]))
                        story.append(t)
                        story.append(Spacer(1, 0.3*inch))
                    
                    # Model Performance
                    if "Model Performance" in include_sections and has_models:
                        story.append(Paragraph("<b>Model Performance</b>", styles['Heading2']))
                        
                        model_data = []
                        model_data.append(['Model', 'Accuracy', 'F1 Score', 'Precision', 'Recall'])
                        
                        for model_name, model_info in st.session_state.trained_models.items():
                            metrics = model_info['metrics']
                            model_data.append([
                                model_name,
                                f"{metrics['accuracy']:.4f}",
                                f"{metrics['f1_macro']:.4f}",
                                f"{metrics['precision']:.4f}",
                                f"{metrics['recall']:.4f}"
                            ])
                        
                        t = Table(model_data, colWidths=[1.5*inch, 1.2*inch, 1.2*inch, 1.2*inch, 1.2*inch])
                        t.setStyle(TableStyle([
                            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                            ('FONTSIZE', (0, 0), (-1, 0), 10),
                            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                            ('GRID', (0, 0), (-1, -1), 1, colors.black)
                        ]))
                        story.append(t)
                        story.append(Spacer(1, 0.3*inch))
                    
                    # Methodology
                    if "Methodology" in include_sections:
                        story.append(PageBreak())
                        story.append(Paragraph("<b>Methodology</b>", styles['Heading2']))
                        methodology_text = """
                        <b>1. Data Preprocessing:</b><br/>
                        - Text normalization using custom Indonesian dictionary<br/>
                        - Sastrawi stemming for root word extraction<br/>
                        - Stopword removal while preserving sentiment indicators<br/>
                        - Negation handling with underscore notation<br/><br/>
                        
                        <b>2. Sentiment Analysis:</b><br/>
                        - BERT-based initial labeling (Indonesian-optimized model)<br/>
                        - Custom veto rule application for immediate negative classification<br/>
                        - Keyword-based sentiment refinement<br/>
                        - Confidence level categorization<br/><br/>
                        
                        <b>3. Machine Learning:</b><br/>
                        - TF-IDF feature extraction with n-grams<br/>
                        - SMOTE oversampling for class balance<br/>
                        - GridSearchCV hyperparameter optimization<br/>
                        - Stratified k-fold cross-validation<br/>
                        """
                        story.append(Paragraph(methodology_text, styles['Normal']))
                    
                    # Build PDF
                    doc.build(story)
                    pdf_buffer.seek(0)
                    
                    st.success("✓ PDF report generated successfully!")
                    
                    st.download_button(
                        label="📥 Download PDF Report",
                        data=pdf_buffer,
                        file_name=f"sentiment_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                        mime="application/pdf",
                        use_container_width=True
                    )
                    
                except ImportError:
                    st.error("reportlab not installed. Install with: pip install reportlab")
                except Exception as e:
                    st.error(f"Error generating PDF: {e}")

# TAB 4: VISUALIZATIONS
with tab4:
    st.markdown("### 📊 Export Visualizations")
    
    if not has_sentiment:
        st.warning("⚠️ No results to visualize. Complete sentiment analysis first.")
    else:
        st.info("📌 Tip: Right-click on any Plotly chart in the Visualizations page and select 'Download plot as PNG'")
        
        st.markdown("#### Available Visualizations")
        st.markdown("""
        Navigate to the **📈 Visualizations** page to view and download:
        - Sentiment distribution charts
        - Word clouds per category
        - Confidence level analysis
        - Model comparison charts
        - Confusion matrices
        - ROC curves
        - Feature importance plots
        
        **To download a chart:**
        1. Hover over the chart
        2. Click the camera icon (📷) in the top-right
        3. Or right-click → "Save image as..."
        """)

st.markdown("---")
st.markdown("### 📋 Export Summary")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Data Files", "3" if has_sentiment else "0")
with col2:
    st.metric("Trained Models", len(st.session_state.trained_models) if has_models else 0)
with col3:
    st.metric("Reports Available", "PDF" if has_sentiment else "None")

st.caption("💡 Export all results before closing the application to preserve your analysis")