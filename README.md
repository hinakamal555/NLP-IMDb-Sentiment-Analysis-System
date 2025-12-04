# MovieSentiment

Binary sentiment analysis system for movie reviews using NLP and machine learning.

**Best Model: Logistic Regression - 88.09% Accuracy**

## Dataset

Uses the Stanford IMDb movie review dataset (50,000 reviews).
- Training: 25,000 reviews (12,500 positive, 12,500 negative)
- Testing: 25,000 reviews (12,500 positive, 12,500 negative)

## Setup

1. **Create virtual environment (recommended):**
```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Download NLTK data:**
```bash
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet')"
```

4. **Download dataset:**
```bash
python data/download_data.py
```

## Usage

### Train Models

Train all models (Naive Bayes, Logistic Regression, SVM, BiLSTM):

```bash
python src/train.py
```

Models are saved to `models/` directory.

### Evaluate Models

Evaluate all trained models on test data:

```bash
python src/evaluate.py
```

Outputs: accuracy, precision, recall, F1 score.

### Make Predictions

**Single review (interactive):**
```bash
python src/predict.py --model logistic_regression
```

**Single review (command line):**
```bash
python src/predict.py --model logistic_regression --text "This movie was amazing!"
```

**Batch predictions (from file):**
```bash
python src/predict.py --model logistic_regression --file reviews.txt
```

Available models: `naive_bayes`, `logistic_regression`, `svm`

## Project Structure

```
NLP-IMDb-Sentiment-Analysis-System/
├── data/
│   ├── aclImdb/              # Dataset (50K reviews)
│   └── download_data.py       # Dataset downloader
├── src/
│   ├── preprocessing.py       # Text preprocessing pipeline
│   ├── train.py              # Model training (fixed seed: 42)
│   ├── evaluate.py           # Model evaluation
│   └── predict.py            # Prediction interface
├── models/                   # Trained models (committed for reproducibility)
│   ├── naive_bayes.pkl
│   ├── logistic_regression.pkl  ⭐ BEST
│   ├── svm.pkl
│   ├── bilstm.h5
│   ├── bilstm_tokenizer.pkl
│   └── tfidf_vectorizer.pkl
├── results/                  # Evaluation artifacts
│   ├── confusion_matrix_*.png
│   ├── classification_report_*.txt
│   └── classification_report_*.csv
├── logs/                     # Training & evaluation logs
│   ├── training_log_*.txt
│   └── evaluation_log_*.txt
├── requirements.txt          # Pinned dependencies
├── generate_results.py       # Generate evaluation artifacts
└── README.md                 # This file
```

## Results and Artifacts

All evaluation artifacts are available in the `results/` folder:

### Confusion Matrices
- `confusion_matrix_logistic_regression.png` (Best model)
- `confusion_matrix_naive_bayes.png`
- `confusion_matrix_svm.png`

### Classification Reports
Available in both TXT and CSV formats:
- `classification_report_logistic_regression.txt/.csv`
- `classification_report_naive_bayes.txt/.csv`
- `classification_report_svm.txt/.csv`

## Logs

All training and evaluation runs are automatically logged with timestamps:
- **Training logs**: `logs/training_log_YYYYMMDD_HHMMSS.txt`
  - Model parameters, training time, accuracies, epoch details, random seed
- **Evaluation logs**: `logs/evaluation_log_YYYYMMDD_HHMMSS.txt`
  - Accuracy, precision, recall, F1-score, classification reports

## How to Reproduce Results

This project is configured for **full reproducibility** with fixed random seeds and pinned dependencies.

2. **Train all models:**
```bash
python src/train.py
```
3. **Evaluate models:**
```bash
python src/evaluate.py
```

This generates detailed metrics for all models.

4. **Generate result artifacts:**
```bash
python generate_results.py
```

This creates:
- Confusion matrices (PNG images)
- Classification reports (TXT and CSV)
- Performance metrics

All outputs saved to `results/` folder.

### Expected Results

**Logistic Regression (Best Model):**
- Accuracy: 88.09%
- Precision: 87.76%
- Recall: 88.52%
- F1-Score: 0.8814

See `results/` folder for detailed reports and confusion matrices.

## Requirements

- Python 3.12.4 (exact version recommended)
- See `requirements.txt` for packages with pinned versions

