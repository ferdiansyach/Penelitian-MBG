# 📊 MBG Sentiment Analysis Dashboard

Complete end-to-end sentiment analysis pipeline for Indonesian text data with interactive Streamlit interface.

## 🎯 Features

### Data Management

- CSV upload with sample dataset
- Dynamic column selection
- Advanced text preprocessing with Sastrawi
- Custom normalization dictionaries
- Real-time statistics and visualizations

### Sentiment Analysis

- BERT-based initial labeling (Indonesian-optimized)
- Custom rule-based refinement with veto logic
- Editable sentiment rules
- Batch processing with progress tracking
- Confidence level categorization

### Machine Learning

- Multiple algorithms (SVM, KNN)
- GridSearchCV hyperparameter tuning
- SMOTE for imbalanced data handling
- Cross-validation metrics
- Quick Mode vs Full Analysis

### Visualization

- Interactive Plotly charts
- Word clouds per sentiment category
- Confusion matrices
- ROC curves
- Feature importance analysis
- Model comparison dashboard

### Real-time Prediction

- Single text sentiment prediction
- Batch prediction from CSV
- ML model comparison
- Rule logic visualization

### Export & Reporting

- CSV with predictions
- Trained models (pickle format)
- PDF comprehensive reports
- Visualization downloads

## 📁 Project Structure

```
sentiment_dashboard/
├── app.py                          # Main application
├── requirements.txt                # Python dependencies
├── README.md                       # This file
├── utils/
│   ├── __init__.py
│   ├── text_processing.py          # Text cleaning utilities
│   ├── sentiment_analysis.py       # BERT & sentiment logic
│   ├── model_training.py           # ML training pipeline
│   └── visualization.py            # Plotly charts
└── pages/
    ├── 1_📁_Data_Upload.py
    ├── 2_🤖_Sentiment_Analysis.py
    ├── 3_🔧_Model_Training.py
    ├── 4_📈_Visualizations.py
    ├── 5_🔍_Error_Analysis.py
    ├── 6_🎯_Live_Predictor.py
    └── 7_💾_Export_Reports.py
```

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- 4GB RAM minimum (8GB recommended)
- ~500MB free disk space for BERT model cache

### Step-by-Step Setup

1. **Clone or download the project files**

   ```bash
   mkdir sentiment_dashboard
   cd sentiment_dashboard
   ```

2. **Create virtual environment (recommended)**

   ```bash
   python -m venv venv

   # Windows
   venv\Scripts\activate

   # Mac/Linux
   source venv/bin/activate
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

4. **Create directory structure**

   ```bash
   mkdir utils pages
   ```

5. **Copy all provided Python files to their respective directories**

   - `app.py` → root directory
   - `utils/*.py` → utils/ folder
   - `pages/*.py` → pages/ folder

6. **Run the application**
   ```bash
   streamlit run app.py
   ```

The dashboard will open in your browser at `http://localhost:8501`

## 📖 Usage Guide

### Quick Start (5 minutes)

1. **Launch the app**: `streamlit run app.py`
2. **Load sample data**: Go to "Data Upload" page → Select "Use Sample Data" → Click "Load Sample Data"
3. **Run sentiment analysis**: Go to "Sentiment Analysis" page → Load BERT Model → Start Analysis
4. **Train models**: Go to "Model Training" page → Select "Quick Mode" → Start Training
5. **Explore results**: Navigate through Visualizations and Error Analysis pages

### Detailed Workflow

#### 1. Data Upload & Preprocessing

- Upload your CSV file or use the included sample dataset
- Select the text column containing data to analyze
- Configure preprocessing options:
  - Enable/disable stemming
  - Enable/disable stopword removal
  - Set minimum text length
- Customize normalization dictionary for slang/typos
- Review data statistics and distributions
- Export cleaned data

#### 2. Sentiment Analysis

- Load BERT model (cached after first load)
- Configure batch processing parameters
- Run sentiment analysis with progress tracking
- Review sentiment distribution and confidence levels
- Customize veto rules and keyword lists
- Test sentiment logic on sample text

#### 3. Model Training

- Select models to train (SVM, KNN, or both)
- Choose analysis mode:
  - **Quick Mode**: Reduced parameters, faster (~2-3 min)
  - **Full Analysis**: Complete grid search (~5-15 min)
- Configure hyperparameters:
  - Test/train split ratio
  - Cross-validation folds
  - SMOTE k-neighbors
- Monitor training progress
- Compare model performance

#### 4. Visualizations

- Interactive Plotly charts for all metrics
- Word clouds per sentiment category
- Confusion matrices with heatmaps
- ROC curves for multi-class classification
- Feature importance analysis
- Sentiment distribution by confidence level

#### 5. Error Analysis

- Review misclassified samples
- Analyze model weaknesses
- Compare BERT vs ML predictions
- Identify patterns in errors

#### 6. Live Predictor

- Test single text predictions in real-time
- View detailed breakdown of prediction logic
- Compare BERT and ML model outputs
- Run batch predictions from CSV files

#### 7. Export & Reports

- Download predictions as CSV
- Export trained models (pickle format)
- Generate comprehensive PDF reports
- Save visualizations as images

## ⚙️ Configuration Options

### Preprocessing Options

- **Stemming**: Reduces words to root form using Sastrawi
- **Stopword Removal**: Removes common words while preserving sentiment words
- **Min Text Length**: Filters out texts below specified character count
- **Normalization Dictionary**: Custom mappings for slang/typos

### Sentiment Rules

- **Veto Words**: Words that immediately classify text as negative
- **Strong Negative Keywords**: Words indicating negative sentiment
- **Strong Positive Keywords**: Words indicating positive sentiment
- **Priority Logic**: Veto → Keyword Comparison → BERT Fallback

### Model Parameters

- **Test Split**: Portion of data for testing (default: 20%)
- **CV Folds**: Cross-validation folds (default: 3)
- **SMOTE K-Neighbors**: Neighbors for oversampling (default: 5)
- **Random State**: Seed for reproducibility (default: 42)

## 🎯 Analysis Modes

### Quick Mode (Recommended for Testing)

- Reduced GridSearchCV parameters
- Faster training (~2-3 minutes)
- Good baseline performance
- Suitable for datasets up to 5,000 rows

### Full Analysis Mode

- Complete hyperparameter grid search
- Longer training time (~5-15 minutes)
- Optimized performance
- Best for final production models

## 📊 Supported Models

### SVM (Support Vector Machine)

- Linear kernel with probability estimates
- TF-IDF feature extraction with n-grams
- Regularization parameter tuning
- Best for: Balanced accuracy and interpretability

### KNN (K-Nearest Neighbors)

- Distance-based classification
- TF-IDF vectorization
- Neighbor count optimization
- Best for: Simple, intuitive predictions

## 🔧 Troubleshooting

### BERT Model Loading Issues

**Problem**: Model fails to load or takes too long
**Solutions**:

- Ensure stable internet connection (first load downloads ~500MB)
- Clear cache: Delete `.streamlit/cache` folder
- Try fallback model: Use XLM-RoBERTa option
- Increase system RAM allocation

### Memory Errors During Training

**Problem**: Out of memory during GridSearchCV
**Solutions**:

- Use Quick Mode instead of Full Analysis
- Reduce dataset size (sample data)
- Lower batch size in configuration
- Close other applications

### Slow Performance

**Problem**: Dashboard is slow or unresponsive
**Solutions**:

- Limit dataset to <10,000 rows
- Use Quick Mode for faster results
- Clear browser cache
- Restart Streamlit server

### Missing Visualizations

**Problem**: Charts don't appear or show errors
**Solutions**:

- Verify all dependencies installed: `pip list`
- Update Plotly: `pip install --upgrade plotly`
- Check browser console for JavaScript errors
- Try different browser (Chrome recommended)

## 📝 Input Data Format

### CSV Requirements

- **Encoding**: UTF-8
- **Structure**: At least one text column
- **Language**: Indonesian (optimized for this language)
- **Size**: Max 10,000 rows recommended
- **Text Quality**: Clean text performs better

### Example CSV Structure

```csv
cleaned_text
"program makan bergizi gratis sangat membantu anak"
"saya tidak setuju dengan kebijakan ini"
"netral saja menurut saya"
```

## 🎨 Customization

### Adding Custom Normalization Rules

1. Go to Data Upload page
2. Select "Customize" under Dictionary Option
3. Add slang:normalized mappings
4. Or upload custom dictionary file (format: `key:value` per line)

### Modifying Veto Rules

1. Go to Sentiment Analysis page
2. Navigate to "Rule Tuning" tab
3. Add/remove veto words
4. Test logic with sample text

### Adjusting Model Parameters

Edit `utils/model_training.py` for advanced customization:

- TF-IDF parameters (max_df, min_df, ngram_range)
- Additional algorithms
- Custom scoring metrics
- Different resampling strategies

## 📈 Performance Metrics

### Sentiment Analysis

- **Accuracy**: Overall correct predictions
- **F1-Score**: Balance of precision and recall
- **Precision**: Correct positive predictions ratio
- **Recall**: Found positive cases ratio
- **Confidence**: BERT model certainty (0-1)

### Model Evaluation

- **Confusion Matrix**: Prediction vs actual breakdown
- **ROC Curve**: True/false positive rate tradeoff
- **Feature Importance**: Most influential words per class
- **Cross-Validation Score**: Average performance across folds

## 🛡️ Best Practices

1. **Data Quality**

   - Clean and preprocess text before analysis
   - Remove duplicates and irrelevant data
   - Ensure adequate samples per class

2. **Model Training**

   - Start with Quick Mode for testing
   - Use Full Analysis for production
   - Validate with held-out test set
   - Monitor for overfitting

3. **Rule Tuning**

   - Test veto words on sample data first
   - Balance keyword lists (positive/negative)
   - Regularly review misclassifications
   - Update rules based on new patterns

4. **Performance**
   - Limit dataset size for interactive use
   - Cache BERT model for faster subsequent runs
   - Export trained models for reuse
   - Clear old session state periodically

## 🤝 Contributing

To extend this dashboard:

1. Add new pages in `pages/` folder (use naming: `N_emoji_Name.py`)
2. Add utility functions in `utils/` modules
3. Update session state management in `app.py`
4. Test with both Quick and Full Analysis modes

## 📄 License

This project is provided as-is for educational and research purposes.

## 🆘 Support

For issues or questions:

1. Check Troubleshooting section above
2. Review Streamlit documentation: https://docs.streamlit.io
3. Verify all dependencies are correctly installed
4. Check Python version compatibility (3.8+)

## 🔄 Version History

**v1.0** - Initial Release

- Complete sentiment analysis pipeline
- Multi-page Streamlit interface
- BERT integration with custom rules
- ML model training with GridSearchCV
- Interactive visualizations
- Real-time prediction
- Export functionality
