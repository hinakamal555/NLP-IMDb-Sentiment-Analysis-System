"""
Evaluate trained sentiment analysis models.
Reports accuracy, precision, recall, and F1 score.
"""

import sys
import pickle
import numpy as np
from datetime import datetime
from pathlib import Path
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from data.download_data import prepare_dataset, EXTRACT_DIR
from src.preprocessing import TextPreprocessor


MODELS_DIR = Path(__file__).parent.parent / "models"
LOGS_DIR = Path(__file__).parent.parent / "logs"
LOGS_DIR.mkdir(exist_ok=True)

# Create log file
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = LOGS_DIR / f"evaluation_log_{timestamp}.txt"

def log_message(message, print_msg=True):
    """Log message to both console and file."""
    if print_msg:
        print(message)
    with open(log_file, 'a', encoding='utf-8') as f:
        f.write(message + '\n')


def load_model(model_name):
    """Load a trained model."""
    model_path = MODELS_DIR / f"{model_name}.pkl"
    if not model_path.exists():
        print(f"Model not found: {model_path}")
        return None
    
    with open(model_path, 'rb') as f:
        return pickle.load(f)


def evaluate_traditional_model(model_name, X_test, y_test):
    """
    Evaluate a traditional ML model.
    
    Args:
        model_name: Name of the model
        X_test: Test features
        y_test: True labels
    
    Returns:
        Dictionary with evaluation metrics
    """
    log_message(f"\n{'='*60}")
    log_message(f"Evaluating: {model_name.replace('_', ' ').title()}")
    log_message('='*60)
    
    model = load_model(model_name)
    if model is None:
        return None
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    log_message(f"\nMetrics:")
    log_message(f"  Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
    log_message(f"  Precision: {precision:.4f} ({precision*100:.2f}%)")
    log_message(f"  Recall:    {recall:.4f} ({recall*100:.2f}%)")
    log_message(f"  F1 Score:  {f1:.4f}")
    
    log_message(f"\nClassification Report:")
    report = classification_report(y_test, y_pred, target_names=['Negative', 'Positive'])
    log_message(report)
    
    return {
        'model': model_name,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    }


def evaluate_bilstm(test_texts, y_test):
    """
    Evaluate BiLSTM model.
    
    Args:
        test_texts: Test texts (preprocessed)
        y_test: True labels
    
    Returns:
        Dictionary with evaluation metrics
    """
    log_message(f"\n{'='*60}")
    log_message(f"Evaluating: BiLSTM")
    log_message('='*60)
    
    try:
        from tensorflow.keras.models import load_model as keras_load_model
        from tensorflow.keras.preprocessing.sequence import pad_sequences
        
        model_path = MODELS_DIR / "bilstm.h5"
        tokenizer_path = MODELS_DIR / "bilstm_tokenizer.pkl"
        
        if not model_path.exists() or not tokenizer_path.exists():
            log_message("BiLSTM model not found. Train it first using train.py")
            return None
        
        # Load model and tokenizer
        model = keras_load_model(model_path)
        with open(tokenizer_path, 'rb') as f:
            tokenizer = pickle.load(f)
        
        # Prepare sequences
        max_len = 200
        X_test = tokenizer.texts_to_sequences(test_texts)
        X_test = pad_sequences(X_test, maxlen=max_len)
        
        # Predict
        y_pred_prob = model.predict(X_test, verbose=0)
        y_pred = (y_pred_prob > 0.5).astype(int).flatten()
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        log_message(f"\nMetrics:")
        log_message(f"  Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
        log_message(f"  Precision: {precision:.4f} ({precision*100:.2f}%)")
        log_message(f"  Recall:    {recall:.4f} ({recall*100:.2f}%)")
        log_message(f"  F1 Score:  {f1:.4f}")
        
        log_message(f"\nClassification Report:")
        report = classification_report(y_test, y_pred, target_names=['Negative', 'Positive'])
        log_message(report)
        
        return {
            'model': 'bilstm',
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        }
        
    except ImportError:
        log_message("TensorFlow not installed. Skipping BiLSTM evaluation.")
        return None


def main():
    """Main evaluation pipeline."""
    log_message("="*60)
    log_message("MovieSentiment Model Evaluation")
    log_message("="*60)
    log_message(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log_message("")
    
    # Check if dataset exists
    if not EXTRACT_DIR.exists():
        log_message("Dataset not found. Please run data/download_data.py first.")
        sys.exit(1)
    
    # Load and preprocess test data
    log_message("Loading test dataset...")
    _, (test_texts, test_labels) = prepare_dataset()
    
    log_message("\nPreprocessing texts...")
    preprocessor = TextPreprocessor(remove_stopwords=True, lemmatize=True)
    test_texts_processed = preprocessor.preprocess_batch(test_texts)
    log_message(f"Preprocessed {len(test_texts_processed)} test samples")
    
    results = []
    
    # Load vectorizer for traditional models
    vectorizer_path = MODELS_DIR / "tfidf_vectorizer.pkl"
    if vectorizer_path.exists():
        with open(vectorizer_path, 'rb') as f:
            vectorizer = pickle.load(f)
        X_test = vectorizer.transform(test_texts_processed)
        
        # Evaluate traditional models
        traditional_models = ['naive_bayes', 'logistic_regression', 'svm']
        for model_name in traditional_models:
            if (MODELS_DIR / f"{model_name}.pkl").exists():
                result = evaluate_traditional_model(model_name, X_test, test_labels)
                if result:
                    results.append(result)
    else:
        log_message("Vectorizer not found. Train models first using train.py")
    
    # Evaluate BiLSTM
    bilstm_result = evaluate_bilstm(test_texts_processed, test_labels)
    if bilstm_result:
        results.append(bilstm_result)
    
    # Summary
    log_message("\n" + "="*60)
    log_message("EVALUATION SUMMARY")
    log_message("="*60)
    log_message(f"{'Model':<25} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1-Score':<12}")
    log_message("-"*70)
    for result in results:
        log_message(f"{result['model']:<25} {result['accuracy']:<12.4f} "
                   f"{result['precision']:<12.4f} {result['recall']:<12.4f} "
                   f"{result['f1_score']:<12.4f}")
    
    # Find best model
    if results:
        best_model = max(results, key=lambda x: x['accuracy'])
        log_message(f"\nBest performing model: {best_model['model']} "
                   f"(Accuracy: {best_model['accuracy']:.4f})")
    
    log_message(f"\nEnd time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log_message("="*60)


if __name__ == "__main__":
    main()

