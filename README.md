# MovieSentiment

Binary sentiment analysis system for movie reviews using NLP and machine learning.

**Best Model: Logistic Regression - 88.09% Accuracy**

## Dataset

Uses the Stanford IMDb movie review dataset (50,000 reviews).
- Training: 25,000 reviews (12,500 positive, 12,500 negative)
- Testing: 25,000 reviews (12,500 positive, 12,500 negative)

## System Requirements

- **Python**: 3.12.4 (required)
- **OS**: Windows 10 / macOS / Linux
- **RAM**: 8GB minimum, 16GB recommended
- **Storage**: ~500MB for dataset, ~50MB for models

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
├── config.yaml               # Hyperparameters configuration
├── requirements.txt          # Pinned dependencies
├── generate_results.py       # Generate evaluation artifacts
├── generate_confusion_matrix.py  # Standalone confusion matrix generator
├── PROJECT_REPORT.md         # Detailed project report
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

## Environment Information

This project was developed and tested in the following environment:

**Software:**
- Python: 3.12.4
- OS: Windows 10
- Platform: win32

**Hardware (recommended minimum):**
- CPU: Any modern multi-core processor
- RAM: 8GB (16GB recommended for BiLSTM)
- Storage: 1GB free space
- GPU: Optional (CPU training works, but BiLSTM is slower)

**Key Dependencies:**
- TensorFlow 2.18.0
- scikit-learn 1.5.2
- NLTK 3.9.1
- NumPy 1.26.4
- Matplotlib 3.9.2

All versions pinned in `requirements.txt` for reproducibility.

## How to Reproduce Results

This project is configured for **full reproducibility** with fixed random seeds and pinned dependencies.

### Complete Reproduction Steps

1. **Setup environment** (see Setup section above)

2. **Train all models:**
```bash
python src/train.py
```

This trains 4 models:
- Naive Bayes (85.08% accuracy)
- Logistic Regression (88.09% accuracy) ⭐ **BEST**
- SVM (86.88% accuracy)
- BiLSTM (85.17% accuracy)

Training takes approximately 10-15 minutes (BiLSTM requires most time).

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

### Model Hyperparameters

All hyperparameters are documented in `config.yaml`. Key settings:

**Preprocessing:**
- TF-IDF with max 5,000 features
- N-gram range: (1, 2) - unigrams and bigrams
- Stopword removal + lemmatization

**Logistic Regression (Best Model):**
- C (inverse regularization): 1.0
- Penalty: L2
- Solver: lbfgs
- Max iterations: 1000
- Random state: 42

**SVM:**
- C: 1.0
- Kernel: Linear (LinearSVC)
- Penalty: L2
- Max iterations: 2000
- Random state: 42

**Naive Bayes:**
- Type: MultinomialNB
- Alpha (smoothing): 1.0 (Laplace smoothing)

**BiLSTM:**
- Vocabulary size: 10,000
- Sequence length: 200
- Embedding dimension: 128
- LSTM units: 64 → 32 (bidirectional)
- Dropout: 0.5
- Epochs: 3
- Batch size: 128
- Optimizer: Adam

### Reproducibility Notes

- **Random Seed**: 42 (fixed across all models)
- **Python Version**: 3.12.4
- **Package Versions**: Pinned in `requirements.txt`
- **Dataset**: Stanford IMDb (deterministic split)
- **Environment**: All settings in `config.yaml`

Running the same code with the same dependencies will produce **identical results** (±0.01% due to floating-point precision).

## Requirements

- Python 3.12.4 (exact version recommended)
- See `requirements.txt` for packages with pinned versions

