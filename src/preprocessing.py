"""
Text preprocessing module for sentiment analysis.
Handles cleaning, tokenization, stopword removal, and lemmatization.
"""

import re
import string
from typing import List
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab', quiet=True)

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet', quiet=True)

try:
    nltk.data.find('corpora/omw-1.4')
except LookupError:
    nltk.download('omw-1.4', quiet=True)


class TextPreprocessor:
    """Text preprocessing pipeline for movie reviews."""
    
    def __init__(self, remove_stopwords=True, lemmatize=True):
        """
        Initialize preprocessor.
        
        Args:
            remove_stopwords: Whether to remove stopwords
            lemmatize: Whether to apply lemmatization
        """
        self.remove_stopwords = remove_stopwords
        self.lemmatize = lemmatize
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
    
    def clean_text(self, text: str) -> str:
        """
        Clean text by removing HTML, URLs, and special characters.
        
        Args:
            text: Raw text
        
        Returns:
            Cleaned text
        """
        # Remove HTML tags
        text = re.sub(r'<.*?>', '', text)
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+', '', text)
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove punctuation
        text = text.translate(str.maketrans('', '', string.punctuation))
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        return text
    
    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize text into words.
        
        Args:
            text: Cleaned text
        
        Returns:
            List of tokens
        """
        return word_tokenize(text)
    
    def remove_stop_words(self, tokens: List[str]) -> List[str]:
        """
        Remove stopwords from tokens.
        
        Args:
            tokens: List of tokens
        
        Returns:
            Filtered tokens
        """
        return [token for token in tokens if token not in self.stop_words]
    
    def lemmatize_tokens(self, tokens: List[str]) -> List[str]:
        """
        Lemmatize tokens to their base form.
        
        Args:
            tokens: List of tokens
        
        Returns:
            Lemmatized tokens
        """
        return [self.lemmatizer.lemmatize(token) for token in tokens]
    
    def preprocess(self, text: str) -> str:
        """
        Complete preprocessing pipeline.
        
        Args:
            text: Raw text
        
        Returns:
            Preprocessed text as string
        """
        # Clean text
        text = self.clean_text(text)
        
        # Tokenize
        tokens = self.tokenize(text)
        
        # Remove stopwords
        if self.remove_stopwords:
            tokens = self.remove_stop_words(tokens)
        
        # Lemmatize
        if self.lemmatize:
            tokens = self.lemmatize_tokens(tokens)
        
        # Join tokens back into string
        return ' '.join(tokens)
    
    def preprocess_batch(self, texts: List[str]) -> List[str]:
        """
        Preprocess a batch of texts.
        
        Args:
            texts: List of raw texts
        
        Returns:
            List of preprocessed texts
        """
        return [self.preprocess(text) for text in texts]


if __name__ == "__main__":
    # Example usage
    preprocessor = TextPreprocessor()
    
    sample_text = """
    <p>This movie was AMAZING! I loved every minute of it. 
    The acting was superb and the plot kept me engaged. 
    Check out http://example.com for more reviews!</p>
    """
    
    print("Original text:")
    print(sample_text)
    print("\nPreprocessed text:")
    print(preprocessor.preprocess(sample_text))

