"""
Download and prepare the Stanford IMDb movie review dataset.
Dataset: https://ai.stanford.edu/~amaas/data/sentiment/
"""

import os
import tarfile
import urllib.request
from pathlib import Path

# Dataset URL
DATASET_URL = "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"
DATA_DIR = Path(__file__).parent
DOWNLOAD_PATH = DATA_DIR / "aclImdb_v1.tar.gz"
EXTRACT_DIR = DATA_DIR / "aclImdb"


def download_dataset():
    """Download the IMDb dataset if not already present."""
    if EXTRACT_DIR.exists():
        print(f"Dataset already exists at {EXTRACT_DIR}")
        return
    
    print(f"Downloading dataset from {DATASET_URL}...")
    os.makedirs(DATA_DIR, exist_ok=True)
    
    try:
        urllib.request.urlretrieve(DATASET_URL, DOWNLOAD_PATH)
        print(f"Download complete: {DOWNLOAD_PATH}")
    except Exception as e:
        print(f"Error downloading dataset: {e}")
        return
    
    # Extract the dataset
    print("Extracting dataset...")
    try:
        with tarfile.open(DOWNLOAD_PATH, 'r:gz') as tar:
            tar.extractall(path=DATA_DIR)
        print(f"Extraction complete: {EXTRACT_DIR}")
        
        # Remove the tar.gz file to save space
        os.remove(DOWNLOAD_PATH)
        print("Cleaned up download file.")
    except Exception as e:
        print(f"Error extracting dataset: {e}")


def load_reviews(data_type='train', sentiment='pos'):
    """
    Load reviews from the dataset.
    
    Args:
        data_type: 'train' or 'test'
        sentiment: 'pos' or 'neg'
    
    Returns:
        List of review texts
    """
    review_dir = EXTRACT_DIR / data_type / sentiment
    reviews = []
    
    if not review_dir.exists():
        print(f"Directory not found: {review_dir}")
        return reviews
    
    for file_path in review_dir.glob('*.txt'):
        with open(file_path, 'r', encoding='utf-8') as f:
            reviews.append(f.read())
    
    return reviews


def prepare_dataset():
    """Prepare train and test datasets with labels."""
    print("Loading training data...")
    train_pos = load_reviews('train', 'pos')
    train_neg = load_reviews('train', 'neg')
    
    print("Loading test data...")
    test_pos = load_reviews('test', 'pos')
    test_neg = load_reviews('test', 'neg')
    
    # Combine reviews and labels
    train_texts = train_pos + train_neg
    train_labels = [1] * len(train_pos) + [0] * len(train_neg)
    
    test_texts = test_pos + test_neg
    test_labels = [1] * len(test_pos) + [0] * len(test_neg)
    
    print(f"\nDataset Summary:")
    print(f"Train: {len(train_texts)} reviews ({len(train_pos)} pos, {len(train_neg)} neg)")
    print(f"Test: {len(test_texts)} reviews ({len(test_pos)} pos, {len(test_neg)} neg)")
    
    return (train_texts, train_labels), (test_texts, test_labels)


if __name__ == "__main__":
    print("=== IMDb Dataset Downloader ===\n")
    download_dataset()
    
    if EXTRACT_DIR.exists():
        print("\n=== Preparing Dataset ===\n")
        prepare_dataset()
        print("\nDataset ready for use!")
    else:
        print("\nFailed to prepare dataset. Please check the download.")

