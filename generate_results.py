"""
Generate evaluation artifacts for reproducibility.
Saves confusion matrix, classification reports, and metrics to results/ folder.
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import (
    confusion_matrix, 
    ConfusionMatrixDisplay, 
    classification_report, 
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)
import sys
import csv

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))

from data.download_data import prepare_dataset
from src.preprocessing import TextPreprocessor

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Directories
MODELS_DIR = Path(__file__).parent / "models"
RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def load_model_and_vectorizer(model_name='logistic_regression'):
    """Load a trained model and vectorizer."""
    # Load vectorizer
    with open(MODELS_DIR / "tfidf_vectorizer.pkl", 'rb') as f:
        vectorizer = pickle.load(f)
    
    # Load model
    with open(MODELS_DIR / f"{model_name}.pkl", 'rb') as f:
        model = pickle.load(f)
    
    return model, vectorizer


def generate_confusion_matrix_image(y_true, y_pred, model_name, accuracy):
    """Generate and save confusion matrix visualization."""
    cm = confusion_matrix(y_true, y_pred)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot confusion matrix
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=['Negative', 'Positive']
    )
    disp.plot(cmap='Blues', ax=ax, values_format='d')
    
    # Customize plot
    ax.set_title(f'Confusion Matrix - {model_name.replace("_", " ").title()}\nIMDb Sentiment Analysis', 
                 fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Predicted Label', fontsize=12, fontweight='bold')
    ax.set_ylabel('True Label', fontsize=12, fontweight='bold')
    
    # Add accuracy text
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
    output_path = RESULTS_DIR / f"confusion_matrix_{model_name}.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Confusion matrix saved to: {output_path}")
    return cm


def save_classification_report_txt(y_true, y_pred, model_name):
    """Save classification report as text file."""
    report = classification_report(
        y_true, y_pred, 
        target_names=['Negative', 'Positive'],
        digits=4
    )
    
    # Calculate additional metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    
    output_path = RESULTS_DIR / f"classification_report_{model_name}.txt"
    
    with open(output_path, 'w') as f:
        f.write("="*70 + "\n")
        f.write(f"Classification Report - {model_name.replace('_', ' ').title()}\n")
        f.write("IMDb Sentiment Analysis System\n")
        f.write("="*70 + "\n\n")
        
        f.write("Model Performance Metrics:\n")
        f.write("-"*70 + "\n")
        f.write(f"Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)\n")
        f.write(f"Precision: {precision:.4f} ({precision*100:.2f}%)\n")
        f.write(f"Recall:    {recall:.4f} ({recall*100:.2f}%)\n")
        f.write(f"F1-Score:  {f1:.4f}\n\n")
        
        f.write("Detailed Classification Report:\n")
        f.write("-"*70 + "\n")
        f.write(report)
        f.write("\n")
        
        # Add confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        f.write("Confusion Matrix:\n")
        f.write("-"*70 + "\n")
        f.write("                    Predicted\n")
        f.write("                 Negative    Positive\n")
        f.write(f"Actual Negative    {cm[0,0]:6d}      {cm[0,1]:6d}\n")
        f.write(f"       Positive    {cm[1,0]:6d}      {cm[1,1]:6d}\n")
    
    print(f"✓ Classification report saved to: {output_path}")


def save_classification_report_csv(y_true, y_pred, model_name):
    """Save classification report as CSV file."""
    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    
    # Get per-class metrics
    report_dict = classification_report(
        y_true, y_pred,
        target_names=['Negative', 'Positive'],
        output_dict=True
    )
    
    output_path = RESULTS_DIR / f"classification_report_{model_name}.csv"
    
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        
        # Write header
        writer.writerow(['Metric', 'Value'])
        writer.writerow(['Model', model_name])
        writer.writerow([''])
        
        # Overall metrics
        writer.writerow(['Overall Metrics', ''])
        writer.writerow(['Accuracy', f'{accuracy:.4f}'])
        writer.writerow(['Precision (Weighted)', f'{precision:.4f}'])
        writer.writerow(['Recall (Weighted)', f'{recall:.4f}'])
        writer.writerow(['F1-Score (Weighted)', f'{f1:.4f}'])
        writer.writerow([''])
        
        # Per-class metrics
        writer.writerow(['Class', 'Precision', 'Recall', 'F1-Score', 'Support'])
        writer.writerow([
            'Negative',
            f"{report_dict['Negative']['precision']:.4f}",
            f"{report_dict['Negative']['recall']:.4f}",
            f"{report_dict['Negative']['f1-score']:.4f}",
            f"{report_dict['Negative']['support']:.0f}"
        ])
        writer.writerow([
            'Positive',
            f"{report_dict['Positive']['precision']:.4f}",
            f"{report_dict['Positive']['recall']:.4f}",
            f"{report_dict['Positive']['f1-score']:.4f}",
            f"{report_dict['Positive']['support']:.0f}"
        ])
        writer.writerow([''])
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        writer.writerow(['Confusion Matrix', ''])
        writer.writerow(['', 'Predicted Negative', 'Predicted Positive'])
        writer.writerow(['Actual Negative', cm[0, 0], cm[0, 1]])
        writer.writerow(['Actual Positive', cm[1, 0], cm[1, 1]])
    
    print(f"✓ CSV report saved to: {output_path}")


def evaluate_model(model_name='logistic_regression'):
    """Evaluate a model and generate all artifacts."""
    print(f"\n{'='*70}")
    print(f"Generating Results for: {model_name.replace('_', ' ').title()}")
    print('='*70)
    
    # Load data
    print("Loading test dataset...")
    (_, _), (test_texts, test_labels) = prepare_dataset()
    
    # Load model
    print("Loading model and vectorizer...")
    model, vectorizer = load_model_and_vectorizer(model_name)
    
    # Preprocess
    print("Preprocessing test texts...")
    preprocessor = TextPreprocessor(remove_stopwords=True, lemmatize=True)
    test_texts_processed = preprocessor.preprocess_batch(test_texts)
    
    # Vectorize
    print("Vectorizing...")
    X_test = vectorizer.transform(test_texts_processed)
    
    # Predict
    print("Making predictions...")
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    accuracy = accuracy_score(test_labels, y_pred)
    
    # Generate artifacts
    print("\nGenerating evaluation artifacts...")
    generate_confusion_matrix_image(test_labels, y_pred, model_name, accuracy)
    save_classification_report_txt(test_labels, y_pred, model_name)
    save_classification_report_csv(test_labels, y_pred, model_name)
    
    print(f"\n✓ All artifacts generated for {model_name}")
    print(f"  Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")


def generate_all_results():
    """Generate results for all trained models."""
    print("="*70)
    print("Generating Evaluation Artifacts for All Models")
    print("="*70)
    
    # Traditional models
    models = ['naive_bayes', 'logistic_regression', 'svm']
    
    for model_name in models:
        model_path = MODELS_DIR / f"{model_name}.pkl"
        if model_path.exists():
            try:
                evaluate_model(model_name)
            except Exception as e:
                print(f"✗ Error evaluating {model_name}: {e}")
        else:
            print(f"⚠ Model not found: {model_name}")
    
    print("\n" + "="*70)
    print("✓ All evaluation artifacts generated successfully!")
    print(f"✓ Results saved to: {RESULTS_DIR}")
    print("="*70)


if __name__ == "__main__":
    try:
        generate_all_results()
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()

