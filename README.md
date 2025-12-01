# MovieSentiment

Binary sentiment analysis system for movie reviews using NLP and machine learning.

## Dataset

Uses the Stanford IMDb movie review dataset (50,000 reviews).

## Setup

1. **Install dependencies:**
```bash
pip install -r requirements.txt
```

2. **Download dataset:**
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
├── data/
│   └── download_data.py    # Dataset downloader
├── src/
│   ├── preprocessing.py    # Text preprocessing
│   ├── train.py           # Model training
│   ├── evaluate.py        # Model evaluation
│   └── predict.py         # Prediction interface
├── models/                # Saved models (auto-generated)
├── logs/                  # Training & evaluation logs (auto-generated)
├── requirements.txt       # Dependencies
└── README.md             # This file
```

## Logs

All training and evaluation runs are automatically logged with timestamps:
- **Training logs**: `logs/training_log_YYYYMMDD_HHMMSS.txt`
  - Model parameters, training time, accuracies, epoch details
- **Evaluation logs**: `logs/evaluation_log_YYYYMMDD_HHMMSS.txt`
  - Accuracy, precision, recall, F1-score, classification reports

## Requirements

- Python 3.8+
- See `requirements.txt` for packages

