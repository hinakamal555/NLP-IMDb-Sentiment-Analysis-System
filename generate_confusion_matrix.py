"""
Generate confusion matrix visualization for the best model (Logistic Regression).
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))

from data.download_data import prepare_dataset
from src.preprocessing import TextPreprocessor

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

def load_model_and_vectorizer():
    """Load the trained model and vectorizer."""
    models_dir = Path(__file__).parent / "models"
    
    # Load vectorizer
    with open(models_dir / "tfidf_vectorizer.pkl", 'rb') as f:
        vectorizer = pickle.load(f)
    
    # Load logistic regression model (best model)
    with open(models_dir / "logistic_regression.pkl", 'rb') as f:
        model = pickle.load(f)
    
    return model, vectorizer


def generate_confusion_matrix():
    """Generate and save confusion matrix for the best model."""
    print("Loading dataset...")
    (_, _), (test_texts, test_labels) = prepare_dataset()
    
    print("Loading model and vectorizer...")
    model, vectorizer = load_model_and_vectorizer()
    
    print("Preprocessing test texts...")
    preprocessor = TextPreprocessor(remove_stopwords=True, lemmatize=True)
    test_texts_processed = preprocessor.preprocess_batch(test_texts)
    
    print("Vectorizing...")
    X_test = vectorizer.transform(test_texts_processed)
    
    print("Making predictions...")
    y_pred = model.predict(X_test)
    
    print("Generating confusion matrix...")
    cm = confusion_matrix(test_labels, y_pred)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot confusion matrix
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=['Negative', 'Positive']
    )
    disp.plot(cmap='Blues', ax=ax, values_format='d')
    
    # Customize plot
    ax.set_title('Confusion Matrix - Logistic Regression\nIMDb Sentiment Analysis', 
                 fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Predicted Label', fontsize=12, fontweight='bold')
    ax.set_ylabel('True Label', fontsize=12, fontweight='bold')
    
    # Add accuracy text
    accuracy = (cm[0, 0] + cm[1, 1]) / cm.sum()
    plt.text(0.5, -0.15, f'Overall Accuracy: {accuracy:.2%} | Test Samples: {cm.sum():,}',
             ha='center', va='center', transform=ax.transAxes,
             fontsize=12, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Add detailed metrics
    true_neg, false_pos, false_neg, true_pos = cm.ravel()
    
    metrics_text = f"""
    True Negatives: {true_neg:,}  |  False Positives: {false_pos:,}
    False Negatives: {false_neg:,}  |  True Positives: {true_pos:,}
    """
    
    plt.text(0.5, -0.25, metrics_text,
             ha='center', va='center', transform=ax.transAxes,
             fontsize=10, family='monospace')
    
    plt.tight_layout()
    
    # Save figure
    output_path = Path(__file__).parent / "confusion_matrix.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nConfusion matrix saved to: {output_path}")
    
    # Display matrix values
    print("\nConfusion Matrix:")
    print("                 Predicted")
    print("               Neg     Pos")
    print(f"Actual Neg    {cm[0,0]:5d}   {cm[0,1]:5d}")
    print(f"       Pos    {cm[1,0]:5d}   {cm[1,1]:5d}")
    print(f"\nAccuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    return cm, accuracy


if __name__ == "__main__":
    try:
        cm, acc = generate_confusion_matrix()
        print("\n✓ Confusion matrix generated successfully!")
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()

