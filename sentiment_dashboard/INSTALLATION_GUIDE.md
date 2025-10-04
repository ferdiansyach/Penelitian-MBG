# 🚀 MBG Sentiment Analysis Dashboard - Installation Guide

## Complete Setup Instructions

This guide will walk you through setting up the complete Streamlit dashboard from scratch.

---

## 📋 Prerequisites

Before you begin, ensure you have:

- **Python 3.8+** installed ([Download here](https://www.python.org/downloads/))
- **4GB RAM minimum** (8GB recommended)
- **~500MB free disk space** (for BERT model cache)
- **pip** package manager (comes with Python)
- Text editor or IDE (VS Code, PyCharm, etc.)

---

## 📁 Step 1: Create Project Structure

Create the following folder structure:

```
sentiment_dashboard/
├── app.py
├── requirements.txt
├── README.md
├── utils/
│   ├── __init__.py
│   ├── text_processing.py
│   ├── sentiment_analysis.py
│   ├── model_training.py
│   └── visualization.py
└── pages/
    ├── 1_📁_Data_Upload.py
    ├── 2_🤖_Sentiment_Analysis.py
    ├── 3_🔧_Model_Training.py
    ├── 4_📈_Visualizations.py
    ├── 5_🔍_Error_Analysis.py
    ├── 6_🎯_Live_Predictor.py
    └── 7_💾_Export_Reports.py
```

### Commands to create structure:

**Windows:**

```cmd
mkdir sentiment_dashboard
cd sentiment_dashboard
mkdir utils pages
type nul > app.py
type nul > requirements.txt
cd utils
type nul > __init__.py
type nul > text_processing.py
type nul > sentiment_analysis.py
type nul > model_training.py
type nul > visualization.py
cd ..
```

**Mac/Linux:**

```bash
mkdir -p sentiment_dashboard/utils sentiment_dashboard/pages
cd sentiment_dashboard
touch app.py requirements.txt
cd utils
touch __init__.py text_processing.py sentiment_analysis.py model_training.py visualization.py
cd ..
```

---

## 📝 Step 2: Copy All Code Files

Copy the provided code into each file:

1. **app.py** - Main application file
2. **requirements.txt** - Dependencies list
3. **utils/**init**.py** - Leave empty (just create the file)
4. **utils/text_processing.py** - Text processing utilities
5. **utils/sentiment_analysis.py** - Sentiment analysis functions
6. **utils/model_training.py** - ML training pipeline
7. **utils/visualization.py** - Plotly visualization functions
8. **pages/1_📁_Data_Upload.py** - Data upload page
9. **pages/2_🤖_Sentiment_Analysis.py** - Sentiment analysis page
10. **pages/3_🔧_Model_Training.py** - Model training page
11. **pages/4_📈_Visualizations.py** - Visualizations page
12. **pages/5_🔍_Error_Analysis.py** - Error analysis page
13. **pages/6_🎯_Live_Predictor.py** - Live predictor page
14. **pages/7_💾_Export_Reports.py** - Export & reports page

---

## 🐍 Step 3: Set Up Python Environment

### Option A: Using venv (Recommended)

**Windows:**

```cmd
python -m venv venv
venv\Scripts\activate
```

**Mac/Linux:**

```bash
python3 -m venv venv
source venv/bin/activate
```

### Option B: Using conda

```bash
conda create -n sentiment python=3.9
conda activate sentiment
```

---

## 📦 Step 4: Install Dependencies

With your virtual environment activated:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

This will install:

- streamlit
- pandas, numpy
- transformers, torch
- scikit-learn, imbalanced-learn
- Sastrawi (Indonesian NLP)
- plotly, matplotlib, seaborn, wordcloud
- reportlab, Pillow

**Note:** Installation may take 5-10 minutes depending on your internet connection.

---

## 🎬 Step 5: Run the Application

From the `sentiment_dashboard` directory:

```bash
streamlit run app.py
```

The dashboard will automatically open in your default browser at:

```
http://localhost:8501
```

If it doesn't open automatically, manually navigate to that URL.

---

## ✅ Step 6: Verify Installation

### Quick Test Checklist:

1. **Homepage loads** - You should see the main dashboard
2. **Sidebar navigation** - All 7 pages listed
3. **Load sample data**:
   - Go to "📁 Data Upload"
   - Click "Load Sample Data"
   - Verify 200 rows loaded
4. **Run preprocessing**:
   - Go to "Preprocessing" tab
   - Click "Run Text Cleaning"
   - Verify cleaned data appears
5. **Load BERT model**:
   - Go to "🤖 Sentiment Analysis"
   - Click "Load BERT Model"
   - Wait for model to download/load (first time only)
6. **Run Quick Analysis**:
   - Stay on Sentiment Analysis page
   - Click "Start Sentiment Analysis"
   - Verify results appear

If all steps work, installation is successful!

---

## 🔧 Troubleshooting

### Problem: ImportError for transformers

**Solution:**

```bash
pip install transformers torch --upgrade
```

### Problem: BERT model fails to load

**Solutions:**

1. Check internet connection
2. Clear Streamlit cache:
   ```bash
   streamlit cache clear
   ```
3. Try the fallback model (XLM-RoBERTa option)

### Problem: Sastrawi not found

**Solution:**

```bash
pip install PySastrawi --upgrade
```

### Problem: Plotly charts don't appear

**Solutions:**

1. Update Plotly:
   ```bash
   pip install plotly --upgrade
   ```
2. Clear browser cache
3. Try different browser (Chrome recommended)

### Problem: PDF generation fails

**Solution:**

```bash
pip install reportlab Pillow --upgrade
```

### Problem: Port 8501 already in use

**Solution:**
Run on different port:

```bash
streamlit run app.py --server.port 8502
```

### Problem: Memory errors during training

**Solutions:**

1. Use "Quick Mode" instead of "Full Analysis"
2. Reduce dataset size (sample fewer rows)
3. Close other applications
4. Increase available RAM

---

## 🔄 Updating the Application

To update dependencies:

```bash
pip install -r requirements.txt --upgrade
```

To clear all caches:

```bash
streamlit cache clear
```

---

## 🌐 Running on Different Devices

### Access from other devices on your network:

1. Find your local IP address:

   - **Windows:** `ipconfig`
   - **Mac/Linux:** `ifconfig` or `ip addr`

2. Run Streamlit with network access:

   ```bash
   streamlit run app.py --server.address 0.0.0.0
   ```

3. Access from other devices:
   ```
   http://YOUR_LOCAL_IP:8501
   ```

---

## 📚 Next Steps

After successful installation:

1. **Read the README.md** for detailed usage instructions
2. **Try the sample workflow**:
   - Load sample data
   - Run sentiment analysis
   - Train models (Quick Mode)
   - Explore visualizations
   - Test live predictor
3. **Upload your own data** and analyze
4. **Customize** normalization dictionaries and veto rules
5. **Export** your results

---

## 🆘 Getting Help

If you encounter issues:

1. Check this troubleshooting guide
2. Verify Python version: `python --version` (should be 3.8+)
3. Verify all files are in correct locations
4. Check Streamlit documentation: https://docs.streamlit.io
5. Ensure all code was copied completely without truncation

---

## 📊 System Requirements Summary

| Component | Minimum                | Recommended    |
| --------- | ---------------------- | -------------- |
| Python    | 3.8                    | 3.9+           |
| RAM       | 4GB                    | 8GB            |
| Storage   | 1GB                    | 2GB            |
| Internet  | Required for first run | -              |
| Browser   | Any modern browser     | Chrome/Firefox |

---

## ✨ Features Overview

After installation, you'll have access to:

- ✅ CSV upload with sample data
- ✅ Advanced Indonesian text preprocessing
- ✅ BERT-based sentiment analysis
- ✅ Custom rule-based refinement
- ✅ Machine learning model training (SVM, KNN)
- ✅ GridSearchCV hyperparameter optimization
- ✅ Interactive Plotly visualizations
- ✅ Real-time sentiment prediction
- ✅ Comprehensive error analysis
- ✅ Export to CSV, Pickle, and PDF

---

## 🎉 Installation Complete!

You're all set! Start exploring the dashboard and analyzing sentiment data.

For detailed usage instructions, see the **README.md** file.

Happy analyzing! 📊
