# ============================================================================
# UTILS PACKAGE INITIALIZATION
# Makes imports cleaner across the application
# ============================================================================

# You can leave this file empty, or add convenient imports:

from .text_processing import (
    clean_text_advanced,
    batch_clean_texts,
    get_text_statistics,
    DEFAULT_NORMALIZATION_DICT,
    DEFAULT_VETO_WORDS
)

from .sentiment_analysis import (
    load_sentiment_model,
    analyze_sentiment_batch,
    map_sentiment_contextual,
    run_sentiment_analysis,
    get_sentiment_distribution,
    predict_single_text
)

from .model_training import (
    prepare_data,
    train_all_models,
    compare_models,
    get_feature_importance,
    get_misclassified_samples
)

from .visualization import (
    plot_sentiment_distribution,
    plot_sentiment_pie,
    plot_confusion_matrix,
    plot_roc_curves,
    generate_wordcloud
)

__version__ = "1.0.0"
__author__ = "Sentiment Analysis Team"