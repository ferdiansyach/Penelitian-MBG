import streamlit as st
import pandas as pd
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from utils.text_processing import (
    clean_text_advanced, batch_clean_texts, get_text_statistics,
    DEFAULT_NORMALIZATION_DICT, export_custom_dictionary, import_custom_dictionary
)
from utils.visualization import (
    plot_text_length_distribution, plot_word_count_distribution
)

st.set_page_config(page_title="Data Upload", page_icon="📁", layout="wide")

st.title("📁 Data Upload & Preprocessing")
st.markdown("Upload your dataset and configure text preprocessing options")

# Sample data function
def create_sample_data():
    """Create sample MBG sentiment data"""
    data = {
        'cleaned_text': [
            'program makan bergizi gratis ini sangat bagus dan membantu anak sekolah',
            'saya suka sekali dengan adanya mbg semoga cepat terealisasi',
            'anggaran untuk makan gratis terlalu besar lebih baik untuk yang lain',
            'tidak setuju dengan program mbg hanya buang buang uang negara',
            'program mbg ini bagus tapi harus diawasi agar tidak ada korupsi',
            'netral saja lihat dulu bagaimana pelaksanaannya nanti',
            'mantap prabowo gibran programnya pro rakyat kecil',
            'apa benar makan gratis bisa mengatasi stunting saya ragu',
            'jangan sampai program ini gagal di tengah jalan seperti yang sudah sudah',
            'keren anak anak jadi semangat sekolah karena ada makan siang gratis',
            'mbg program yang sangat bermanfaat untuk kesehatan anak indonesia',
            'pemerintah harus transparan dalam pelaksanaan program makan gratis',
            'saya khawatir program ini hanya janji politik saja tidak terealisasi',
            'bagus sekali inisiatif untuk mengurangi stunting di indonesia',
            'buang buang anggaran negara untuk program yang tidak efektif',
            'semoga program mbg berjalan lancar dan tepat sasaran',
            'masalah korupsi harus dihindari dalam program makan bergizi gratis',
            'anak anak indonesia berhak mendapat makanan bergizi setiap hari',
            'saya tidak yakin program ini bisa berkelanjutan dalam jangka panjang',
            'luar biasa pemerintah serius menangani masalah gizi buruk anak'
        ] * 10  # 200 samples
    }
    return pd.DataFrame(data)

# Tabs for organization
tab1, tab2, tab3, tab4 = st.tabs(["📤 Upload", "🔧 Preprocessing", "📊 Statistics", "💾 Export"])

# TAB 1: UPLOAD
with tab1:
    st.markdown("### Upload Your Data")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        upload_option = st.radio(
            "Choose data source:",
            ["Upload CSV File", "Use Sample Data"],
            horizontal=True
        )
        
        if upload_option == "Upload CSV File":
            uploaded_file = st.file_uploader(
                "Choose a CSV file",
                type=['csv'],
                help="Upload a CSV file containing text data for sentiment analysis"
            )
            
            if uploaded_file:
                try:
                    df = pd.read_csv(uploaded_file)
                    st.success(f"✓ File uploaded successfully: {len(df)} rows, {len(df.columns)} columns")
                    
                    # Column selection
                    text_columns = df.select_dtypes(include=['object']).columns.tolist()
                    
                    if text_columns:
                        selected_column = st.selectbox(
                            "Select the text column to analyze:",
                            text_columns,
                            help="Choose the column containing the text data"
                        )
                        
                        if st.button("Load Data", type="primary"):
                            st.session_state.df_original = df
                            st.session_state.text_column = selected_column
                            st.success(f"✓ Data loaded! Column '{selected_column}' selected for analysis.")
                            st.rerun()
                    else:
                        st.error("No text columns found in the uploaded file!")
                        
                except Exception as e:
                    st.error(f"Error reading file: {e}")
        
        else:  # Use Sample Data
            st.info("📦 Sample dataset contains 200 Indonesian tweets about MBG program")
            
            if st.button("Load Sample Data", type="primary"):
                df = create_sample_data()
                st.session_state.df_original = df
                st.session_state.text_column = 'cleaned_text'
                st.success("✓ Sample data loaded successfully!")
                st.rerun()
    
    with col2:
        st.markdown("### 📋 Requirements")
        st.info("""
        **CSV Format:**
        - UTF-8 encoding
        - Contains text column(s)
        - No missing values in text
        
        **Recommended:**
        - Max 10,000 rows
        - Clean text format
        - Indonesian language
        """)
    
    # Display loaded data
    if st.session_state.df_original is not None:
        st.markdown("---")
        st.markdown("### 📄 Loaded Data Preview")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Rows", len(st.session_state.df_original))
        with col2:
            st.metric("Total Columns", len(st.session_state.df_original.columns))
        with col3:
            st.metric("Text Column", st.session_state.text_column)
        
        st.dataframe(
            st.session_state.df_original.head(10),
            use_container_width=True,
            height=300
        )

# TAB 2: PREPROCESSING
with tab2:
    st.markdown("### 🔧 Text Preprocessing Configuration")
    
    if st.session_state.df_original is None:
        st.warning("⚠️ Please upload data first in the Upload tab")
    else:
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("#### Basic Options")
            
            use_stemming = st.checkbox(
                "Use Stemming",
                value=st.session_state.preprocessing_options['use_stemming'],
                help="Apply Sastrawi stemmer to reduce words to root form"
            )
            
            use_stopwords = st.checkbox(
                "Remove Stopwords",
                value=st.session_state.preprocessing_options['use_stopwords'],
                help="Remove common Indonesian stopwords (preserving sentiment words)"
            )
            
            min_length = st.slider(
                "Minimum Text Length (characters)",
                min_value=5,
                max_value=50,
                value=st.session_state.preprocessing_options['min_text_length'],
                help="Filter out texts shorter than this length after cleaning"
            )
            
            # Update session state
            st.session_state.preprocessing_options.update({
                'use_stemming': use_stemming,
                'use_stopwords': use_stopwords,
                'min_text_length': min_length
            })
        
        with col2:
            st.markdown("#### Normalization Dictionary")
            
            dict_option = st.radio(
                "Dictionary Option:",
                ["Use Default", "Customize", "Upload Custom"],
                help="Choose how to handle text normalization"
            )
            
            if dict_option == "Use Default":
                st.session_state.normalization_dict = DEFAULT_NORMALIZATION_DICT.copy()
                st.info(f"Using default dictionary with {len(DEFAULT_NORMALIZATION_DICT)} entries")
                
                with st.expander("View Default Dictionary (first 20)"):
                    items = list(DEFAULT_NORMALIZATION_DICT.items())[:20]
                    for key, value in items:
                        st.text(f"{key} → {value}")
            
            elif dict_option == "Customize":
                st.markdown("**Add Custom Mappings:**")
                
                col_a, col_b = st.columns(2)
                with col_a:
                    custom_key = st.text_input("Slang/Typo")
                with col_b:
                    custom_value = st.text_input("Normalized Form")
                
                if st.button("Add Mapping"):
                    if custom_key and custom_value:
                        if 'normalization_dict' not in st.session_state:
                            st.session_state.normalization_dict = DEFAULT_NORMALIZATION_DICT.copy()
                        st.session_state.normalization_dict[custom_key] = custom_value
                        st.success(f"Added: {custom_key} → {custom_value}")
                    else:
                        st.error("Both fields required!")
                
                # Show current custom entries
                if st.session_state.normalization_dict:
                    custom_entries = {k: v for k, v in st.session_state.normalization_dict.items() 
                                    if k not in DEFAULT_NORMALIZATION_DICT}
                    if custom_entries:
                        st.markdown(f"**Custom Entries ({len(custom_entries)}):**")
                        st.json(custom_entries)
            
            else:  # Upload Custom
                dict_file = st.file_uploader(
                    "Upload dictionary file (key:value per line)",
                    type=['txt'],
                    help="Format: slang:normalized (one per line)"
                )
                if dict_file:
                    try:
                        content = dict_file.read().decode('utf-8')
                        custom_dict = {}
                        for line in content.split('\n'):
                            if ':' in line:
                                key, value = line.strip().split(':', 1)
                                custom_dict[key.strip()] = value.strip()
                        st.session_state.normalization_dict = {**DEFAULT_NORMALIZATION_DICT, **custom_dict}
                        st.success(f"✓ Loaded {len(custom_dict)} custom mappings")
                    except Exception as e:
                        st.error(f"Error loading file: {e}")
        
        st.markdown("---")
        
        # Run preprocessing
        if st.button("🚀 Run Text Cleaning", type="primary", use_container_width=True):
            with st.spinner("Cleaning text data..."):
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                def update_progress(current, total):
                    progress = current / total
                    progress_bar.progress(progress)
                    status_text.text(f"Processed {current}/{total} texts ({progress*100:.1f}%)")
                
                cleaned_texts = batch_clean_texts(
                    st.session_state.df_original[st.session_state.text_column].tolist(),
                    normalization_dict=st.session_state.normalization_dict,
                    use_stemming=use_stemming,
                    use_stopwords=use_stopwords,
                    min_length=min_length,
                    progress_callback=update_progress
                )
                
                # Create cleaned dataframe
                df_cleaned = st.session_state.df_original.copy()
                df_cleaned['text_cleaned'] = cleaned_texts
                
                # Filter empty texts
                original_count = len(df_cleaned)
                df_cleaned = df_cleaned[df_cleaned['text_cleaned'].str.strip().str.len() > 0].reset_index(drop=True)
                filtered_count = original_count - len(df_cleaned)
                
                st.session_state.df_cleaned = df_cleaned
                
                progress_bar.empty()
                status_text.empty()
                
                st.success(f"✓ Text cleaning completed!")
                st.info(f"Filtered out {filtered_count} texts (too short/empty). Remaining: {len(df_cleaned)}")
                
                # Show comparison
                st.markdown("### 📊 Before & After Comparison")
                comparison_df = pd.DataFrame({
                    'Original': df_cleaned[st.session_state.text_column].head(5),
                    'Cleaned': df_cleaned['text_cleaned'].head(5)
                })
                st.dataframe(comparison_df, use_container_width=True)

# TAB 3: STATISTICS
with tab3:
    st.markdown("### 📊 Data Statistics")
    
    if st.session_state.df_cleaned is None:
        st.warning("⚠️ Please run text cleaning first in the Preprocessing tab")
    else:
        df = st.session_state.df_cleaned
        
        # Basic stats
        col1, col2, col3, col4 = st.columns(4)
        
        stats = get_text_statistics(df, 'text_cleaned')
        
        with col1:
            st.metric("Total Texts", stats['total_texts'])
        with col2:
            st.metric("Avg Text Length", f"{stats['avg_length']:.0f} chars")
        with col3:
            st.metric("Avg Words", f"{stats['avg_words']:.1f}")
        with col4:
            st.metric("Unique Words", stats['unique_words'])
        
        st.markdown("---")
        
        # Visualizations
        col1, col2 = st.columns(2)
        
        with col1:
            fig1 = plot_text_length_distribution(df, 'text_cleaned')
            st.plotly_chart(fig1, use_container_width=True)
        
        with col2:
            fig2 = plot_word_count_distribution(df, 'text_cleaned')
            st.plotly_chart(fig2, use_container_width=True)
        
        # Word frequency
        st.markdown("### 🔤 Most Common Words")
        
        all_words = ' '.join(df['text_cleaned'].dropna()).split()
        word_freq = pd.Series(all_words).value_counts().head(20)
        
        st.bar_chart(word_freq)

# TAB 4: EXPORT
with tab4:
    st.markdown("### 💾 Export Cleaned Data")
    
    if st.session_state.df_cleaned is None:
        st.warning("⚠️ No cleaned data available to export")
    else:
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Export Options")
            
            # CSV export
            csv = st.session_state.df_cleaned.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Cleaned CSV",
                data=csv,
                file_name=f"cleaned_data_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True
            )
            
            # Dictionary export
            if st.button("📥 Export Normalization Dictionary", use_container_width=True):
                dict_text = "\n".join([f"{k}:{v}" for k, v in st.session_state.normalization_dict.items()])
                st.download_button(
                    label="Download Dictionary",
                    data=dict_text,
                    file_name="normalization_dict.txt",
                    mime="text/plain"
                )
        
        with col2:
            st.markdown("#### Export Summary")
            st.info(f"""
            **Cleaned Dataset:**
            - Total rows: {len(st.session_state.df_cleaned)}
            - Columns: {len(st.session_state.df_cleaned.columns)}
            - Dictionary entries: {len(st.session_state.normalization_dict)}
            """)

st.markdown("---")
st.markdown("**Next Step:** Go to **🤖 Sentiment Analysis** page to analyze the cleaned text")