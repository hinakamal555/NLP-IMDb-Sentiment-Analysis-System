"""
Train sentiment analysis models on movie reviews.
Supports: Naive Bayes, Logistic Regression, SVM, and BiLSTM.

Fixed random seeds ensure reproducibility across runs.
"""

import os
import sys
import pickle
import numpy as np
import time
from datetime import datetime
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split

# Set random seeds for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# Set TensorFlow seed (if available)
try:
    import tensorflow as tf
    tf.random.set_seed(RANDOM_SEED)
except ImportError:
    pass

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from data.download_data import prepare_dataset, EXTRACT_DIR
from src.preprocessing import TextPreprocessor


# Directories
MODELS_DIR = Path(__file__).parent.parent / "models"
MODELS_DIR.mkdir(exist_ok=True)
LOGS_DIR = Path(__file__).parent.parent / "logs"
LOGS_DIR.mkdir(exist_ok=True)

# Create log file
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = LOGS_DIR / f"training_log_{timestamp}.txt"

def log_message(message, print_msg=True):
    """Log message to both console and file."""
    if print_msg:
        print(message)
    with open(log_file, 'a', encoding='utf-8') as f:
        f.write(message + '\n')


def load_and_preprocess_data():
    """Load and preprocess the IMDb dataset."""
    log_message("Loading dataset...")
    if not EXTRACT_DIR.exists():
        log_message("Dataset not found. Please run data/download_data.py first.")
        sys.exit(1)
    
    (train_texts, train_labels), (test_texts, test_labels) = prepare_dataset()
    
    log_message("\nPreprocessing texts...")
    start_time = time.time()
    preprocessor = TextPreprocessor(remove_stopwords=True, lemmatize=True)
    
    train_texts = preprocessor.preprocess_batch(train_texts)
    test_texts = preprocessor.preprocess_batch(test_texts)
    
    preprocessing_time = time.time() - start_time
    log_message(f"Preprocessing completed in {preprocessing_time:.2f} seconds")
    
    return train_texts, train_labels, test_texts, test_labels


def train_traditional_models(X_train, y_train, X_test, y_test):
    """
    Train traditional ML models (Naive Bayes, Logistic Regression, SVM).
    
    Args:
        X_train: Training feature vectors
        y_train: Training labels
        X_test: Test feature vectors
        y_test: Test labels
    """
    models = {
        'naive_bayes': MultinomialNB(),
        'logistic_regression': LogisticRegression(max_iter=1000, random_state=42),
        'svm': LinearSVC(random_state=42, max_iter=2000)
    }
    
    training_results = []
    
    for model_name, model in models.items():
        log_message(f"\n{'='*60}")
        log_message(f"Training {model_name.replace('_', ' ').title()}...")
        log_message(f"{'='*60}")
        
        start_time = time.time()
        model.fit(X_train, y_train)
        training_time = time.time() - start_time
        
        train_score = model.score(X_train, y_train)
        test_score = model.score(X_test, y_test)
        
        log_message(f"  Training time: {training_time:.2f} seconds")
        log_message(f"  Train accuracy: {train_score:.4f} ({train_score*100:.2f}%)")
        log_message(f"  Test accuracy: {test_score:.4f} ({test_score*100:.2f}%)")
        
        # Save model
        model_path = MODELS_DIR / f"{model_name}.pkl"
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        log_message(f"  Model saved to: models/{model_name}.pkl")
        
        training_results.append({
            'model': model_name,
            'train_accuracy': train_score,
            'test_accuracy': test_score,
            'training_time': training_time
        })
    
    return training_results


def train_bilstm(train_texts, y_train, test_texts, y_test):
    """
    Train BiLSTM model (optional, requires TensorFlow/Keras).
    
    Args:
        train_texts: Training texts (preprocessed)
        y_train: Training labels
        test_texts: Test texts (preprocessed)
        y_test: Test labels
    """
    try:
        from tensorflow.keras.preprocessing.text import Tokenizer
        from tensorflow.keras.preprocessing.sequence import pad_sequences
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, Dense, Dropout
        
        log_message("\n" + "="*60)
        log_message("Training BiLSTM Model")
        log_message("="*60)
        
        # Tokenize and pad sequences
        max_words = 10000
        max_len = 200
        
        log_message(f"\nModel parameters:")
        log_message(f"  Max vocabulary size: {max_words}")
        log_message(f"  Max sequence length: {max_len}")
        log_message(f"  Embedding dimension: 128")
        log_message(f"  LSTM units: 64 -> 32")
        log_message(f"  Epochs: 3")
        log_message(f"  Batch size: 128")
        
        tokenizer = Tokenizer(num_words=max_words)
        tokenizer.fit_on_texts(train_texts)
        
        X_train = tokenizer.texts_to_sequences(train_texts)
        X_test = tokenizer.texts_to_sequences(test_texts)
        
        X_train = pad_sequences(X_train, maxlen=max_len)
        X_test = pad_sequences(X_test, maxlen=max_len)
        
        # Build BiLSTM model
        model = Sequential([
            Embedding(max_words, 128, input_length=max_len),
            Bidirectional(LSTM(64, return_sequences=True)),
            Bidirectional(LSTM(32)),
            Dense(64, activation='relu'),
            Dropout(0.5),
            Dense(1, activation='sigmoid')
        ])
        
        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        log_message("\nModel architecture:")
        # Capture model summary
        import io
        stream = io.StringIO()
        model.summary(print_fn=lambda x: stream.write(x + '\n'))
        summary_str = stream.getvalue()
        log_message(summary_str, print_msg=False)
        print(summary_str)
        
        # Train model
        log_message("\nTraining BiLSTM...")
        start_time = time.time()
        history = model.fit(
            X_train, np.array(y_train),
            validation_data=(X_test, np.array(y_test)),
            epochs=3,
            batch_size=128,
            verbose=1
        )
        training_time = time.time() - start_time
        
        # Log training results
        final_train_acc = history.history['accuracy'][-1]
        final_val_acc = history.history['val_accuracy'][-1]
        
        log_message(f"\nBiLSTM Training Results:")
        log_message(f"  Total training time: {training_time:.2f} seconds")
        log_message(f"  Final train accuracy: {final_train_acc:.4f} ({final_train_acc*100:.2f}%)")
        log_message(f"  Final validation accuracy: {final_val_acc:.4f} ({final_val_acc*100:.2f}%)")
        
        log_message("\nEpoch-wise training history:")
        for epoch in range(len(history.history['accuracy'])):
            log_message(f"  Epoch {epoch+1}: train_acc={history.history['accuracy'][epoch]:.4f}, "
                       f"val_acc={history.history['val_accuracy'][epoch]:.4f}, "
                       f"train_loss={history.history['loss'][epoch]:.4f}, "
                       f"val_loss={history.history['val_loss'][epoch]:.4f}")
        
        # Save model and tokenizer
        model.save(MODELS_DIR / "bilstm.h5")
        with open(MODELS_DIR / "bilstm_tokenizer.pkl", 'wb') as f:
            pickle.dump(tokenizer, f)
        
        log_message(f"\nBiLSTM model saved to models/bilstm.h5")
        log_message(f"Tokenizer saved to models/bilstm_tokenizer.pkl")
        
        return {
            'model': 'bilstm',
            'train_accuracy': final_train_acc,
            'test_accuracy': final_val_acc,
            'training_time': training_time
        }
        
    except ImportError:
        log_message("\nSkipping BiLSTM training (TensorFlow not installed)")
        log_message("To train BiLSTM, install: pip install tensorflow")
        return None


def main():
    """Main training pipeline."""
    overall_start = time.time()
    
    log_message("="*60)
    log_message("MovieSentiment Training")
    log_message("="*60)
    log_message(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log_message(f"Random seed: {RANDOM_SEED} (for reproducibility)")
    log_message("")
    
    # Load and preprocess data
    train_texts, train_labels, test_texts, test_labels = load_and_preprocess_data()
    
    # Create TF-IDF vectors for traditional ML models
    log_message("\n" + "="*60)
    log_message("Vectorizing texts (TF-IDF)")
    log_message("="*60)
    log_message("TF-IDF Parameters:")
    log_message("  Max features: 5000")
    log_message("  N-gram range: (1, 2)")
    
    start_time = time.time()
    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    X_train = vectorizer.fit_transform(train_texts)
    X_test = vectorizer.transform(test_texts)
    vectorization_time = time.time() - start_time
    
    log_message(f"Vectorization completed in {vectorization_time:.2f} seconds")
    log_message(f"Training matrix shape: {X_train.shape}")
    log_message(f"Testing matrix shape: {X_test.shape}")
    
    # Save vectorizer
    vectorizer_path = MODELS_DIR / "tfidf_vectorizer.pkl"
    with open(vectorizer_path, 'wb') as f:
        pickle.dump(vectorizer, f)
    log_message(f"Vectorizer saved to models/tfidf_vectorizer.pkl")
    
    # Train traditional ML models
    log_message("\n" + "="*60)
    log_message("Training Traditional ML Models")
    log_message("="*60)
    results = train_traditional_models(X_train, train_labels, X_test, test_labels)
    
    # Train BiLSTM (optional)
    bilstm_result = train_bilstm(train_texts, train_labels, test_texts, test_labels)
    if bilstm_result:
        results.append(bilstm_result)
    
    # Summary
    overall_time = time.time() - overall_start
    log_message("\n" + "="*60)
    log_message("TRAINING SUMMARY")
    log_message("="*60)
    log_message(f"Total training time: {overall_time:.2f} seconds ({overall_time/60:.2f} minutes)")
    log_message(f"\nModel Performance Summary:")
    log_message(f"{'Model':<25} {'Train Acc':<12} {'Test Acc':<12} {'Time (s)':<12}")
    log_message("-"*60)
    for result in results:
        log_message(f"{result['model']:<25} {result['train_accuracy']:<12.4f} "
                   f"{result['test_accuracy']:<12.4f} {result['training_time']:<12.2f}")
    
    log_message(f"\nAll models saved to: models/")
    log_message(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log_message("="*60)


if __name__ == "__main__":
    main()

