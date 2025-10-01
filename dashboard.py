import streamlit as st
import pandas as pd
import numpy as np
import re
import warnings
import torch
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from transformers import pipeline
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from wordcloud import WordCloud
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

# ==============================================================================
# KONFIGURASI HALAMAN DAN GAYA (CSS)
# ==============================================================================
st.set_page_config(
    page_title="Dashboard Analisis Sentimen MBG",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1E88E5; /* Biru */
        text-align: center;
        padding: 1rem 0;
        border-bottom: 3px solid #1E88E5;
        margin-bottom: 2rem;
    }
    .metric-card {
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        margin: 0.5rem 0;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    .positive-card { background: linear-gradient(135deg, #43A047 0%, #66BB6A 100%); } /* Hijau */
    .negative-card { background: linear-gradient(135deg, #E53935 0%, #EF5350 100%); } /* Merah */
    .neutral-card { background: linear-gradient(135deg, #757575 0%, #9E9E9E 100%); } /* Abu-abu */
    .total-card { background: linear-gradient(135deg, #1E88E5 0%, #42A5F5 100%); } /* Biru */

    .stButton>button {
        background-color: #1E88E5;
        color: white;
        border-radius: 8px;
        border: none;
        padding: 10px 20px;
        font-size: 16px;
    }
    .stButton>button:hover {
        background-color: #1565C0;
    }
</style>
""", unsafe_allow_html=True)


# ==============================================================================
# DATA & KONFIGURASI (Kamus Normalisasi, dll.)
# ==============================================================================
# Diambil dari skrip asli Anda
normalization_dict = {
    'yg': 'yang', 'dgn': 'dengan', 'utk': 'untuk', 'kpd': 'kepada', 'dr': 'dari',
    'krn': 'karena', 'bkn': 'bukan', 'jg': 'juga', 'sdh': 'sudah', 'blm': 'belum',
    'tdk': 'tidak', 'ga': 'tidak', 'gak': 'tidak', 'nggak': 'tidak', 'jgn': 'jangan',
    'bnyk': 'banyak', 'byk': 'banyak', 'aja': 'saja', 'kalo': 'kalau', 'pake': 'pakai',
    'bgt': 'sangat', 'banget': 'sangat', 'mantul': 'mantap', 'keren': 'bagus',
    'oke': 'baik', 'pro': 'mendukung', 'kontra': 'menolak', 'hoax': 'bohong',
    'mbg': 'makan bergizi gratis', 'wkwk': '', 'haha': '', 'hehe': '',
    # Tambahkan normalisasi lain yang relevan jika perlu
}

# ==============================================================================
# FUNGSI-FUNGSI UTAMA (dengan caching untuk performa)
# ==============================================================================

@st.cache_data
def to_csv(df):
    """Mengonversi DataFrame menjadi file CSV (format UTF-8)."""
    return df.to_csv(index=False).encode('utf-8')

@st.cache_resource
def load_resources():
    """Memuat semua resource berat seperti model, stemmer, dan stopwords."""
    stem_factory = StemmerFactory()
    stemmer = stem_factory.create_stemmer()

    stop_factory = StopWordRemoverFactory()
    default_stopwords = set(stop_factory.get_stop_words())
    sentiment_words = {
        'tidak', 'bukan', 'jangan', 'kurang', 'buruk', 'jelek', 'bagus', 'baik',
        'suka', 'benci', 'senang', 'sedih', 'marah', 'kecewa', 'puas', 'mantap',
        'gagal', 'sangat', 'sekali', 'banget', 'terima', 'kasih'
    }
    custom_stopwords = default_stopwords - sentiment_words

    device = 0 if torch.cuda.is_available() else -1
    sentiment_analyzer = pipeline(
        "sentiment-analysis",
        model="ayameRushia/bert-base-indonesian-1.5G-sentiment-analysis-smsa",
        device=device,
        return_all_scores=True
    )
    return stemmer, custom_stopwords, sentiment_analyzer

# Memuat resource di awal
stemmer, custom_stopwords, sentiment_analyzer = load_resources()

def clean_text_advanced(text, _stemmer, _custom_stopwords):
    if pd.isna(text): return ''
    text = str(text).lower()
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#(\w+)', r'\1', text)
    negation_patterns = [(r'\b(tidak|ga|gak|nggak|bukan|jangan|kurang)\s+(\w+)', r'\1_\2')]
    for pattern, replacement in negation_patterns:
        text = re.sub(pattern, replacement, text)
    text = re.sub(r'[^\w\s_]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    tokens = text.split()
    tokens = [normalization_dict.get(tok, tok) for tok in tokens]
    tokens = [tok for tok in tokens if tok not in _custom_stopwords and len(tok) > 2]
    no_stem_words = {'korupsi', 'anggaran', 'program', 'kebijakan', 'pemerintah', 'masyarakat', 'bansos', 'prabowo', 'gibran'}
    tokens = [tok if tok in no_stem_words or '_' in tok else _stemmer.stem(tok) for tok in tokens]
    return ' '.join(tokens)

def map_sentiment_contextual(row):
    label = row['raw_sentiment_label'].upper()
    text = str(row['text_cleaned']).lower()
    veto_negative_words = [
        'masalah', 'gagal', 'korupsi', 'korup', 'kkn', 'bohong', 'hoax', 'tipu', 'penipuan',
        'curang', 'pembodohan', 'sia sia', 'buang', 'boros', 'rugi', 'tidak berguna', 'kacau'
    ]
    strong_negative = ['tidak_', 'bukan_', 'jangan_', 'kurang_', 'kecewa', 'benci', 'marah', 'tolak', 'kontra']
    strong_positive = ['bagus', 'baik', 'suka', 'puas', 'mantap', 'dukung', 'setuju', 'bermanfaat', 'membantu']

    if any(word in text for word in veto_negative_words): return 'negative'
    negative_count = sum(1 for word in strong_negative if word in text)
    positive_count = sum(1 for word in strong_positive if word in text)

    if negative_count > positive_count: return 'negative'
    if positive_count > negative_count: return 'positive'
    
    mapping = {'POSITIVE': 'positive', 'NEGATIVE': 'negative', 'NEUTRAL': 'neutral'}
    return mapping.get(label, 'neutral')

def run_full_analysis(df, source_column):
    """Menjalankan seluruh pipeline analisis pada DataFrame yang diberikan."""
    progress_bar = st.progress(0, text="Memulai analisis...")

    # 1. Cleaning Text
    progress_bar.progress(10, text="Tahap 1/4: Membersihkan teks...")
    df['text_cleaned'] = df[source_column].apply(lambda x: clean_text_advanced(x, stemmer, custom_stopwords))
    df.dropna(subset=['text_cleaned'], inplace=True)
    df = df[df['text_cleaned'].str.strip().str.len() > 15].reset_index(drop=True)

    # 2. Analisis dengan Model BERT
    progress_bar.progress(30, text="Tahap 2/4: Menganalisis sentimen dengan model BERT (Mungkin perlu waktu)...")
    texts_to_analyze = df['text_cleaned'].tolist()
    sentiment_results = []
    batch_size = 16 # Batch size lebih kecil untuk desktop GPU
    for i in range(0, len(texts_to_analyze), batch_size):
        batch = texts_to_analyze[i:i+batch_size]
        results = sentiment_analyzer(batch)
        sentiment_results.extend(results)
        progress_bar.progress(30 + int(40 * (i + len(batch)) / len(texts_to_analyze)), text=f"Tahap 2/4: Menganalisis sentimen... ({i+len(batch)}/{len(texts_to_analyze)})")

    df['raw_sentiment_label'] = [max(res, key=lambda x: x['score'])['label'] for res in sentiment_results]
    df['raw_sentiment_score'] = [max(res, key=lambda x: x['score'])['score'] for res in sentiment_results]

    # 3. Pemetaan Sentimen Kontekstual
    progress_bar.progress(75, text="Tahap 3/4: Menyempurnakan label sentimen dengan aturan kontekstual...")
    df['sentiment'] = df.apply(map_sentiment_contextual, axis=1)

    # 4. Pelatihan Model Klasifikasi Kustom (SVC)
    progress_bar.progress(85, text="Tahap 4/4: Melatih model klasifikasi kustom (SVM)...")
    sentiment_map = {'negative': 0, 'neutral': 1, 'positive': 2}
    df['sentiment_code'] = df['sentiment'].map(sentiment_map)
    X = df['text_cleaned'].dropna()
    y = df.loc[X.index, 'sentiment_code']

    # Hanya jalankan jika ada cukup data dan kelas
    model_results = {}
    if y.nunique() > 1 and len(y) > 100:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
        
        pipeline_svc = ImbPipeline(steps=[
            ('tfidf', TfidfVectorizer(ngram_range=(1, 2), max_df=0.75, min_df=3)),
            ('smote', SMOTE(random_state=42)),
            ('classifier', SVC(kernel='linear', C=1, probability=True, random_state=42))
        ])
        
        pipeline_svc.fit(X_train, y_train)
        y_pred = pipeline_svc.predict(X_test)
        
        report = classification_report(y_test, y_pred, target_names=['negative', 'neutral', 'positive'], output_dict=True)
        cm = confusion_matrix(y_test, y_pred)
        
        model_results = {
            'model': pipeline_svc,
            'report': report,
            'cm': cm,
            'X_test': X_test,
            'y_test': y_test,
            'y_pred': y_pred
        }
    else:
        st.warning("Tidak cukup data atau variasi kelas untuk melatih model klasifikasi kustom.")

    progress_bar.progress(100, text="Analisis Selesai!")
    progress_bar.empty()
    
    return df, model_results

# ==============================================================================
# TAMPILAN APLIKASI STREAMLIT
# ==============================================================================

# Inisialisasi session state
if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = None
if 'model_results' not in st.session_state:
    st.session_state.model_results = None

# --- SIDEBAR ---
st.sidebar.title("Kontrol Dasbor")
st.sidebar.header("Unggah Data Anda")
uploaded_file = st.sidebar.file_uploader(
    "Pilih file CSV",
    type=['csv'],
    help="Unggah file CSV yang berisi kolom teks untuk dianalisis."
)

if uploaded_file:
    df_raw = pd.read_csv(uploaded_file)
    st.sidebar.success(f"File '{uploaded_file.name}' berhasil diunggah. Terdapat {len(df_raw)} baris.")
    
    available_columns = df_raw.columns.tolist()
    source_column = st.sidebar.selectbox(
        "Pilih kolom yang berisi teks:",
        options=available_columns,
        index=0
    )

    if st.sidebar.button("🚀 Mulai Analisis Menyeluruh"):
        with st.spinner("Harap tunggu, analisis sedang berlangsung..."):
            final_df, model_res = run_full_analysis(df_raw.copy(), source_column)
            st.session_state.analysis_results = final_df
            st.session_state.model_results = model_res
        st.success("Analisis menyeluruh selesai! Hasil tersedia di tab.")

# --- KONTEN UTAMA ---
st.markdown('<div class="main-header">📊 Dashboard Analisis Sentimen Program MBG</div>', unsafe_allow_html=True)

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🏠 Beranda", 
    "📊 Dasbor Utama", 
    "☁️ Word Clouds", 
    "🛠️ Hasil Model Kustom", 
    "💬 Coba Analisis Sendiri"
])

with tab1:
    st.header("Selamat Datang di Dashboard Analisis Sentimen")
    st.write("""
        Aplikasi ini dirancang untuk menganalisis sentimen publik terhadap **Program Makan Bergizi Gratis (MBG)** dari data teks, seperti media sosial. 
        Gunakan dasbor ini untuk mendapatkan wawasan mendalam dari data Anda.

        **Bagaimana Cara Menggunakannya?**
        1.  **Unggah Data**: Gunakan panel di sebelah kiri untuk mengunggah file CSV Anda.
        2.  **Pilih Kolom**: Pilih kolom yang berisi teks yang ingin Anda analisis.
        3.  **Mulai Analisis**: Klik tombol "Mulai Analisis Menyeluruh" untuk menjalankan seluruh proses.
        4.  **Jelajahi Hasil**: Setelah analisis selesai, jelajahi berbagai tab untuk melihat visualisasi dan hasil model.

        **Fitur Utama:**
        - **Dasbor Interaktif**: Visualisasikan distribusi sentimen dengan grafik yang jelas.
        - **Word Clouds**: Identifikasi kata-kata kunci yang paling sering muncul untuk setiap sentimen.
        - **Model Kustom**: Lihat performa model Machine Learning (SVM) yang dilatih khusus pada data Anda.
        - **Analisis Interaktif**: Coba analisis sentimen pada kalimat Anda sendiri secara real-time.
    """)
    st.info("Untuk memulai, silakan unggah file CSV di sidebar kiri.")

if st.session_state.analysis_results is not None:
    df_final = st.session_state.analysis_results
    
    with tab2:
        st.header("📊 Dasbor Utama Hasil Analisis")
        
        # Metrik Utama
        total_texts = len(df_final)
        positive_count = len(df_final[df_final['sentiment'] == 'positive'])
        negative_count = len(df_final[df_final['sentiment'] == 'negative'])
        neutral_count = len(df_final[df_final['sentiment'] == 'neutral'])

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown(f'<div class="metric-card total-card"><h4>Total Teks Dianalisis</h4><h1>{total_texts}</h1></div>', unsafe_allow_html=True)
        with col2:
            st.markdown(f'<div class="metric-card positive-card"><h4>Sentimen Positif</h4><h1>{positive_count}</h1></div>', unsafe_allow_html=True)
        with col3:
            st.markdown(f'<div class="metric-card negative-card"><h4>Sentimen Negatif</h4><h1>{negative_count}</h1></div>', unsafe_allow_html=True)
        with col4:
            st.markdown(f'<div class="metric-card neutral-card"><h4>Sentimen Netral</h4><h1>{neutral_count}</h1></div>', unsafe_allow_html=True)
        
        st.markdown("---")

        # Visualisasi Distribusi
        col1, col2 = st.columns([2, 1])
        with col1:
            sentiment_counts = df_final['sentiment'].value_counts().reset_index()
            sentiment_counts.columns = ['sentiment', 'count']
            
            fig_bar = px.bar(
                sentiment_counts, 
                x='sentiment', 
                y='count', 
                color='sentiment',
                title="<b>Distribusi Jumlah Sentimen</b>",
                text='count',
                color_discrete_map={'positive': '#43A047', 'negative': '#E53935', 'neutral': '#757575'}
            )
            fig_bar.update_layout(xaxis_title="Sentimen", yaxis_title="Jumlah Teks", showlegend=False)
            st.plotly_chart(fig_bar, use_container_width=True)

        with col2:
            fig_pie = px.pie(
                sentiment_counts, 
                names='sentiment', 
                values='count', 
                title="<b>Proporsi Sentimen</b>",
                hole=0.4,
                color='sentiment',
                color_discrete_map={'positive': '#43A047', 'negative': '#E53935', 'neutral': '#757575'}
            )
            fig_pie.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig_pie, use_container_width=True)

        # Tampilkan Data Hasil
        st.markdown("---")
        st.subheader("Data Hasil Analisis")
        st.dataframe(df_final[[source_column, 'text_cleaned', 'sentiment']], use_container_width=True)
        
        # Tombol Download
        csv_data = to_csv(df_final)
        st.download_button(
            label="📥 Download Data Hasil (CSV)",
            data=csv_data,
            file_name='hasil_analisis_sentimen_mbg.csv',
            mime='text/csv',
        )

    with tab3:
        st.header("☁️ Word Clouds per Kategori Sentimen")
        st.write("Visualisasi kata-kata yang paling sering muncul untuk setiap kategori sentimen.")
        
        col1, col2, col3 = st.columns(3)
        sentiments = ['positive', 'negative', 'neutral']
        cols = [col1, col2, col3]
        
        for sent, col in zip(sentiments, cols):
            with col:
                st.subheader(f"Sentimen: {sent.capitalize()}")
                text_data = " ".join(df_final[df_final['sentiment'] == sent]['text_cleaned'].dropna())
                
                if text_data and len(text_data.strip()) > 0:
                    try:
                        wordcloud = WordCloud(
                            width=800, height=600,
                            background_color='white',
                            colormap='viridis',
                            max_words=100,
                            collocations=False
                        ).generate(text_data)
                        
                        fig, ax = plt.subplots(figsize=(10, 8))
                        ax.imshow(wordcloud, interpolation='bilinear')
                        ax.axis('off')
                        st.pyplot(fig)
                    except ValueError:
                        st.warning(f"Tidak cukup kata unik untuk membuat word cloud '{sent}'.")
                else:
                    st.info(f"Tidak ada data untuk sentimen '{sent}'.")

    with tab4:
        st.header("🛠️ Hasil Model Klasifikasi Kustom (SVM)")
        model_res = st.session_state.model_results
        
        if model_res:
            st.write("Model Support Vector Machine (SVM) dilatih pada data yang telah dianalisis untuk mengklasifikasikan teks baru. Berikut adalah evaluasi performanya pada data uji.")
            
            report_df = pd.DataFrame(model_res['report']).transpose()
            st.subheader("Laporan Klasifikasi")
            st.dataframe(report_df.round(3), use_container_width=True)

            st.subheader("Confusion Matrix")
            fig_cm = go.Figure(data=go.Heatmap(
                z=model_res['cm'],
                x=['negative', 'neutral', 'positive'],
                y=['negative', 'neutral', 'positive'],
                colorscale='Blues',
                text=model_res['cm'],
                texttemplate="%{text}"
            ))
            fig_cm.update_layout(
                title='<b>Confusion Matrix</b>',
                xaxis_title='Prediksi Model',
                yaxis_title='Label Aktual'
            )
            st.plotly_chart(fig_cm, use_container_width=True)
            
            st.subheader("Fitur/Kata Paling Penting")
            try:
                model = model_res['model'].named_steps['classifier']
                vectorizer = model_res['model'].named_steps['tfidf']
                feature_names = vectorizer.get_feature_names_out()
                
                if hasattr(model, 'coef_'):
                    coef = model.coef_
                    
                    feature_data = []
                    for i, label in enumerate(['negative', 'neutral', 'positive']):
                        top_features_idx = coef[i].argsort()[-15:]
                        top_features = feature_names[top_features_idx]
                        top_scores = coef[i][top_features_idx]
                        
                        for feature, score in zip(top_features, top_scores):
                            feature_data.append({'Sentimen': label, 'Fitur': feature, 'Bobot': score})
                    
                    feature_df = pd.DataFrame(feature_data)
                    
                    fig_features = px.bar(
                        feature_df.sort_values('Bobot', ascending=True),
                        x='Bobot',
                        y='Fitur',
                        color='Sentimen',
                        orientation='h',
                        title='<b>Top 15 Fitur Paling Berpengaruh per Sentimen</b>',
                        height=800,
                        color_discrete_map={'positive': '#43A047', 'negative': '#E53935', 'neutral': '#757575'}
                    )
                    st.plotly_chart(fig_features, use_container_width=True)
                else:
                    st.warning("Model yang digunakan tidak memiliki atribut 'coef_' untuk analisis fitur.")
            except Exception as e:
                st.error(f"Gagal menganalisis fitur penting: {e}")

        else:
            st.warning("Model kustom belum dilatih. Jalankan analisis menyeluruh terlebih dahulu.")

with tab5:
    st.header("💬 Coba Analisis Sentimen Sendiri")
    st.write("Masukkan sebuah kalimat untuk mendapatkan analisis sentimen instan menggunakan model BERT.")
    
    text_to_try = st.text_input("Kalimat:", "Makan gratis ini bagus, tapi anggarannya harus diawasi agar tidak ada korupsi.")
    
    if st.button("Analisis Kalimat"):
        if text_to_try:
            with st.spinner("Menganalisis..."):
                cleaned_input = clean_text_advanced(text_to_try, stemmer, custom_stopwords)
                result = sentiment_analyzer(cleaned_input)[0]
                
                result_df = pd.DataFrame(result).sort_values(by='score', ascending=False)
                final_label = result_df.iloc[0]['label']
                final_score = result_df.iloc[0]['score']

            st.subheader("Hasil:")
            if final_label == 'POSITIVE':
                st.success(f"**Sentimen: Positif** (Skor: {final_score:.2%})")
            elif final_label == 'NEGATIVE':
                st.error(f"**Sentimen: Negatif** (Skor: {final_score:.2%})")
            else:
                st.warning(f"**Sentimen: Netral** (Skor: {final_score:.2%})")

            # Tampilkan dalam bentuk bar chart
            fig = px.bar(result_df, x='score', y='label', orientation='h', labels={'score': 'Skor Kepercayaan', 'label': 'Sentimen'},
                         color='label', color_discrete_map={'POSITIVE': '#43A047', 'NEGATIVE': '#E53935', 'NEUTRAL': '#757575'})
            fig.update_layout(showlegend=False, yaxis={'categoryorder':'total ascending'})
            st.plotly_chart(fig, use_container_width=True)
            
            with st.expander("Lihat Detail Proses"):
                st.write("**Teks yang Dibersihkan:**")
                st.code(cleaned_input)