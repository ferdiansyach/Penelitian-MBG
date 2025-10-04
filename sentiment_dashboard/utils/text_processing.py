# ============================================================================
# TEXT PROCESSING UTILITIES
# Functions for text cleaning, normalization, and preprocessing
# ============================================================================

import re
import pandas as pd
import streamlit as st
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

# Default normalization dictionary
DEFAULT_NORMALIZATION_DICT = {
    # Basic words
    'yg': 'yang', 'dgn': 'dengan', 'utk': 'untuk', 'kpd': 'kepada', 
    'dr': 'dari', 'karna': 'karena', 'krn': 'karena', 'bkn': 'bukan', 
    'jg': 'juga', 'sdh': 'sudah', 'udh': 'sudah', 'udah': 'sudah',
    'blm': 'belum', 'thn': 'tahun', 'tgl': 'tanggal', 'dll': 'dan lain lain',
    'tdk': 'tidak', 'ga': 'tidak', 'gak': 'tidak', 'nggak': 'tidak',
    'jgn': 'jangan', 'bnyk': 'banyak', 'byk': 'banyak', 'dpt': 'dapat',
    
    # Positive slang
    'mantul': 'mantap', 'keren': 'bagus', 'oke': 'baik', 'mantap': 'baik',
    'top': 'baik', 'recommended': 'bagus', 'gokil': 'luar biasa',
    'amazing': 'luar biasa', 'perfect': 'sempurna', 'josss': 'bagus',
    'ciamik': 'bagus', 'maknyus': 'enak', 'puas': 'puas', 'suka': 'suka',
    
    # Negative slang
    'jelek': 'jelek', 'parah': 'parah', 'gaje': 'buruk', 'zonk': 'buruk',
    'hate': 'benci', 'worst': 'terburuk', 'useless': 'tidak berguna',
    'gagal': 'gagal', 'ancur': 'hancur', 'bobrok': 'buruk', 'busuk': 'busuk',
    'kecewa': 'kecewa', 'sedih': 'sedih', 'marah': 'marah',
    
    # MBG specific
    'mbg': 'makan bergizi gratis', 'gratisan': 'gratis', 
    'prabowo': 'prabowo', 'gibran': 'gibran', 'jkw': 'jokowi',
    'pemda': 'pemerintah daerah', 'apbn': 'anggaran pendapatan dan belanja negara',
    
    # Internet slang
    'wkwk': '', 'wkwkwk': '', 'haha': '', 'hehe': '', 
    'bro': 'kawan', 'sist': 'saudari', 'gan': 'juragan'
}

# Default veto negative words
DEFAULT_VETO_WORDS = [
    'masalah', 'gagal', 'korupsi', 'bohong', 'korban', 'mati', 
    'pembodohan', 'janji palsu', 'omong kosong', 'penipuan', 
    'curang', 'buang', 'mentah', 'kacau', 'paksa'
]

# Initialize tools
@st.cache_resource
def get_text_processing_tools():
    """Initialize and cache Sastrawi tools"""
    stem_factory = StemmerFactory()
    stemmer = stem_factory.create_stemmer()
    
    stop_factory = StopWordRemoverFactory()
    default_stopwords = set(stop_factory.get_stop_words())
    
    sentiment_words = {
        'tidak', 'bukan', 'jangan', 'kurang', 'buruk', 'jelek', 'bagus', 'baik',
        'suka', 'benci', 'senang', 'sedih', 'marah', 'kecewa', 'puas', 'mantap',
        'hebat', 'luar', 'biasa', 'sempurna', 'terburuk', 'terbaik', 'parah',
        'gagal', 'sangat', 'sekali', 'banget', 'enak', 'recommended', 'membantu',
        'berguna', 'sia', 'tipu', 'bohong', 'terima', 'kasih', 'gratis'
    }
    
    custom_stopwords = default_stopwords - sentiment_words
    
    return stemmer, custom_stopwords


def clean_text_advanced(text, normalization_dict=None, use_stemming=True, 
                       use_stopwords=True, min_length=15):
    """
    Advanced text cleaning with customizable options
    """
    if pd.isna(text):
        return ''
    
    if normalization_dict is None:
        normalization_dict = DEFAULT_NORMALIZATION_DICT
    
    stemmer, custom_stopwords = get_text_processing_tools()
    
    # Convert to lowercase
    text = str(text).lower()
    
    # Remove URLs
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    
    # Remove mentions
    text = re.sub(r'@\w+', '', text)
    
    # Clean hashtags (keep the word)
    text = re.sub(r'#(\w+)', r'\1', text)
    
    # Handle repeated characters
    text = re.sub(r'(.)\1{2,}', r'\1\1', text)
    
    # Preserve negations
    negation_patterns = [
        (r'\b(tidak|ga|gak|nggak|bukan|jangan|kurang)\s+(\w+)', r'\1_\2')
    ]
    for pattern, replacement in negation_patterns:
        text = re.sub(pattern, replacement, text)
    
    # Handle emphasis
    text = re.sub(r'[!]{2,}', ' sangat_penting ', text)
    
    # Remove numbers
    text = re.sub(r'\d+', '', text)
    
    # Remove special characters (keep underscores for negations)
    text = re.sub(r'[^\w\s_]', ' ', text)
    
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Tokenize
    tokens = text.split()
    
    # Normalize tokens
    tokens = [normalization_dict.get(tok, tok) for tok in tokens]
    
    # Filter short tokens (with exceptions)
    meaningful_short = {'ga', 'ng', 'ok', 'yg', 'dr', 'tp', 'mbg'}
    tokens = [tok for tok in tokens if len(tok) >= 3 or tok in meaningful_short]
    
    # Remove stopwords if enabled
    if use_stopwords:
        tokens = [tok for tok in tokens if tok not in custom_stopwords]
    
    # Stemming if enabled
    if use_stemming:
        no_stem_words = {
            'tidak', 'bukan', 'jangan', 'kurang', 'sangat', 'sekali', 'korupsi', 
            'anggaran', 'program', 'masalah', 'kebijakan', 'pemerintah', 
            'gagal', 'bohong', 'korban', 'mati', 'curang', 'buang', 'mentah'
        }
        tokens = [tok if tok in no_stem_words or '_' in tok 
                 else stemmer.stem(tok) for tok in tokens]
    
    # Join and check minimum length
    cleaned = ' '.join(tokens)
    
    return cleaned if len(cleaned) >= min_length else ''


def batch_clean_texts(texts, normalization_dict=None, use_stemming=True,
                     use_stopwords=True, min_length=15, progress_callback=None):
    """
    Clean multiple texts with progress tracking
    """
    results = []
    total = len(texts)
    
    for i, text in enumerate(texts):
        cleaned = clean_text_advanced(
            text, 
            normalization_dict=normalization_dict,
            use_stemming=use_stemming,
            use_stopwords=use_stopwords,
            min_length=min_length
        )
        results.append(cleaned)
        
        if progress_callback and (i + 1) % 100 == 0:
            progress_callback(i + 1, total)
    
    return results


def get_text_statistics(df, text_column):
    """
    Calculate text statistics for visualization
    """
    stats = {
        'total_texts': len(df),
        'avg_length': df[text_column].str.len().mean(),
        'max_length': df[text_column].str.len().max(),
        'min_length': df[text_column].str.len().min(),
        'avg_words': df[text_column].str.split().str.len().mean(),
        'total_words': df[text_column].str.split().str.len().sum(),
        'unique_words': len(set(' '.join(df[text_column].dropna()).split()))
    }
    
    return stats


def export_custom_dictionary(normalization_dict, filename='custom_dict.txt'):
    """
    Export normalization dictionary to text file
    """
    with open(filename, 'w', encoding='utf-8') as f:
        for key, value in normalization_dict.items():
            f.write(f"{key}:{value}\n")


def import_custom_dictionary(filename):
    """
    Import normalization dictionary from text file
    """
    custom_dict = {}
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            if ':' in line:
                key, value = line.strip().split(':', 1)
                custom_dict[key.strip()] = value.strip()
    return custom_dict