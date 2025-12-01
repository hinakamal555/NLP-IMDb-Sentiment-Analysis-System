"""
Make sentiment predictions on new movie reviews.
Supports single review or batch predictions.
"""

import sys
import pickle
import argparse
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from src.preprocessing import TextPreprocessor


MODELS_DIR = Path(__file__).parent.parent / "models"


class SentimentPredictor:
    """Sentiment prediction class."""
    
    def __init__(self, model_name='logistic_regression'):
        """
        Initialize predictor with a trained model.
        
        Args:
            model_name: Name of the model to use
        """
        self.model_name = model_name
        self.preprocessor = TextPreprocessor(remove_stopwords=True, lemmatize=True)
        
        # Load vectorizer
        vectorizer_path = MODELS_DIR / "tfidf_vectorizer.pkl"
        if not vectorizer_path.exists():
            raise FileNotFoundError(
                f"Vectorizer not found at {vectorizer_path}. "
                "Please train models first using src/train.py"
            )
        
        with open(vectorizer_path, 'rb') as f:
            self.vectorizer = pickle.load(f)
        
        # Load model
        model_path = MODELS_DIR / f"{model_name}.pkl"
        if not model_path.exists():
            raise FileNotFoundError(
                f"Model not found at {model_path}. "
                "Available models: naive_bayes, logistic_regression, svm"
            )
        
        with open(model_path, 'rb') as f:
            self.model = pickle.load(f)
        
        print(f"Loaded model: {model_name}")
    
    def predict(self, text):
        """
        Predict sentiment for a single review.
        
        Args:
            text: Review text
        
        Returns:
            Tuple of (prediction, confidence)
        """
        # Preprocess
        processed_text = self.preprocessor.preprocess(text)
        
        # Vectorize
        vector = self.vectorizer.transform([processed_text])
        
        # Predict
        prediction = self.model.predict(vector)[0]
        
        # Get confidence (if available)
        try:
            probabilities = self.model.predict_proba(vector)[0]
            confidence = probabilities[prediction]
        except AttributeError:
            # SVM doesn't have predict_proba
            confidence = None
        
        return prediction, confidence
    
    def predict_batch(self, texts):
        """
        Predict sentiment for multiple reviews.
        
        Args:
            texts: List of review texts
        
        Returns:
            List of (prediction, confidence) tuples
        """
        # Preprocess
        processed_texts = self.preprocessor.preprocess_batch(texts)
        
        # Vectorize
        vectors = self.vectorizer.transform(processed_texts)
        
        # Predict
        predictions = self.model.predict(vectors)
        
        # Get confidences
        try:
            probabilities = self.model.predict_proba(vectors)
            confidences = [probabilities[i][pred] for i, pred in enumerate(predictions)]
        except AttributeError:
            confidences = [None] * len(predictions)
        
        return list(zip(predictions, confidences))


def format_prediction(text, prediction, confidence):
    """Format prediction output."""
    sentiment = "POSITIVE" if prediction == 1 else "NEGATIVE"
    
    print(f"\nReview: {text[:100]}{'...' if len(text) > 100 else ''}")
    print(f"Sentiment: {sentiment}")
    if confidence is not None:
        print(f"Confidence: {confidence:.2%}")
    print("-" * 50)


def main():
    """Main prediction interface."""
    parser = argparse.ArgumentParser(description='Predict sentiment of movie reviews')
    parser.add_argument(
        '--model',
        default='logistic_regression',
        choices=['naive_bayes', 'logistic_regression', 'svm'],
        help='Model to use for prediction'
    )
    parser.add_argument(
        '--text',
        type=str,
        help='Single review text to predict'
    )
    parser.add_argument(
        '--file',
        type=str,
        help='File containing reviews (one per line)'
    )
    
    args = parser.parse_args()
    
    # Initialize predictor
    try:
        predictor = SentimentPredictor(model_name=args.model)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)
    
    print("="*50)
    print("MovieSentiment Prediction")
    print("="*50)
    
    # Predict from text
    if args.text:
        prediction, confidence = predictor.predict(args.text)
        format_prediction(args.text, prediction, confidence)
    
    # Predict from file
    elif args.file:
        file_path = Path(args.file)
        if not file_path.exists():
            print(f"File not found: {file_path}")
            sys.exit(1)
        
        with open(file_path, 'r', encoding='utf-8') as f:
            reviews = [line.strip() for line in f if line.strip()]
        
        print(f"\nProcessing {len(reviews)} reviews from {file_path}...\n")
        
        results = predictor.predict_batch(reviews)
        
        for text, (prediction, confidence) in zip(reviews, results):
            format_prediction(text, prediction, confidence)
    
    # Interactive mode
    else:
        print("\nInteractive Mode (press Ctrl+C to exit)")
        print("Enter a movie review to analyze:\n")
        
        try:
            while True:
                text = input("Review: ").strip()
                if not text:
                    continue
                
                prediction, confidence = predictor.predict(text)
                format_prediction(text, prediction, confidence)
                print()
        
        except KeyboardInterrupt:
            print("\n\nGoodbye!")


if __name__ == "__main__":
    main()

