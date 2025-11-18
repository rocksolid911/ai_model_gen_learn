"""
Text Preprocessing for News Classification
Includes traditional (TF-IDF) and neural (tokenization) approaches.
"""

import re
import string
from typing import List, Tuple, Optional
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle

try:
    import nltk
    from nltk.corpus import stopwords
    # Try to load stopwords, download if not available
    try:
        STOPWORDS = set(stopwords.words('english'))
    except LookupError:
        print("Downloading NLTK stopwords...")
        nltk.download('stopwords', quiet=True)
        STOPWORDS = set(stopwords.words('english'))
except ImportError:
    print("Warning: NLTK not available. Stopwords removal will be skipped.")
    STOPWORDS = set()


class TextPreprocessor:
    """
    Handles text cleaning and preprocessing for news articles.

    Why preprocess text:
    - Remove noise (HTML, special characters, extra spaces)
    - Normalize case (HELLO = hello)
    - Remove stopwords (common words like 'the', 'is', 'and')
    - Convert text to features that models can understand
    """

    def __init__(self, remove_stopwords: bool = True, lowercase: bool = True):
        """
        Initialize text preprocessor.

        Args:
            remove_stopwords: Whether to remove common stopwords
            lowercase: Whether to convert text to lowercase
        """
        self.remove_stopwords = remove_stopwords
        self.lowercase = lowercase
        self.stopwords = STOPWORDS if remove_stopwords else set()

    def clean_text(self, text: str) -> str:
        """
        Clean a single text string.

        Steps:
        1. Lowercase (normalize case)
        2. Remove URLs, emails, mentions
        3. Remove extra punctuation and spaces
        4. Remove stopwords

        Args:
            text: Raw text string

        Returns:
            Cleaned text string
        """
        if not isinstance(text, str):
            return ""

        # Convert to lowercase
        # Why: "Apple" and "apple" should be treated the same
        if self.lowercase:
            text = text.lower()

        # Remove URLs
        # Why: URLs don't help with political classification
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)

        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)

        # Remove mentions and hashtags (Twitter-style)
        text = re.sub(r'@\w+|#\w+', '', text)

        # Remove numbers (optional, can be useful in some contexts)
        # text = re.sub(r'\d+', '', text)

        # Remove extra punctuation but keep sentence structure
        # Why: "!!!" doesn't add much value, but periods help
        text = re.sub(r'[^\w\s\.]', ' ', text)

        # Remove extra whitespace
        # Why: Multiple spaces waste computation
        text = re.sub(r'\s+', ' ', text).strip()

        # Remove stopwords
        # Why: Words like "the", "is", "and" appear everywhere and add noise
        if self.remove_stopwords and self.stopwords:
            words = text.split()
            words = [w for w in words if w not in self.stopwords]
            text = ' '.join(words)

        return text

    def clean_texts(self, texts: List[str]) -> List[str]:
        """
        Clean multiple texts.

        Args:
            texts: List of raw text strings

        Returns:
            List of cleaned text strings
        """
        return [self.clean_text(text) for text in texts]


class TfidfFeatureExtractor:
    """
    Extract TF-IDF features for traditional ML models.

    TF-IDF (Term Frequency-Inverse Document Frequency):
    - TF: How often a word appears in a document
    - IDF: How unique/rare a word is across all documents
    - High TF-IDF = word is frequent in this document but rare overall
    - This highlights important, distinctive words
    """

    def __init__(
        self,
        max_features: int = 5000,
        ngram_range: Tuple[int, int] = (1, 2)
    ):
        """
        Initialize TF-IDF feature extractor.

        Args:
            max_features: Maximum number of features to keep
                         Why: Too many features = overfitting and slow training
            ngram_range: Range of n-grams to consider
                        (1,1) = single words only
                        (1,2) = single words + word pairs
                        Why: "not good" has different meaning than "good"
        """
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            strip_accents='unicode',
            lowercase=True,
            token_pattern=r'\w{2,}',  # Words with 2+ characters
            min_df=2,  # Word must appear in at least 2 documents
            max_df=0.9  # Ignore words in >90% of documents
        )

    def fit_transform(self, texts: List[str]) -> np.ndarray:
        """
        Fit vectorizer on training texts and transform them.

        Why fit on training only:
        - Prevents data leakage from test set
        - Model should only learn from training vocabulary

        Args:
            texts: List of text strings

        Returns:
            TF-IDF feature matrix (sparse)
        """
        # Fit learns the vocabulary and IDF weights
        # Transform converts texts to TF-IDF vectors
        features = self.vectorizer.fit_transform(texts)
        print(f"TF-IDF features shape: {features.shape}")
        print(f"Vocabulary size: {len(self.vectorizer.vocabulary_)}")
        return features

    def transform(self, texts: List[str]) -> np.ndarray:
        """
        Transform texts using fitted vectorizer.

        Use this for validation and test sets.

        Args:
            texts: List of text strings

        Returns:
            TF-IDF feature matrix
        """
        return self.vectorizer.transform(texts)

    def save(self, path: str):
        """Save fitted vectorizer to disk."""
        with open(path, 'wb') as f:
            pickle.dump(self.vectorizer, f)
        print(f"TF-IDF vectorizer saved to {path}")

    def load(self, path: str):
        """Load fitted vectorizer from disk."""
        with open(path, 'rb') as f:
            self.vectorizer = pickle.load(f)
        print(f"TF-IDF vectorizer loaded from {path}")


def main():
    """Demo text preprocessing."""
    # Sample texts
    texts = [
        "Breaking News!!! The president announced new policies today.",
        "BREAKING: Major development in the ongoing investigation!",
        "Opinion: This is clearly biased reporting with propaganda.",
    ]

    # Clean texts
    preprocessor = TextPreprocessor(remove_stopwords=True)
    cleaned = preprocessor.clean_texts(texts)

    print("Original texts:")
    for i, text in enumerate(texts):
        print(f"{i+1}. {text}")

    print("\nCleaned texts:")
    for i, text in enumerate(cleaned):
        print(f"{i+1}. {text}")

    # Extract TF-IDF features
    print("\n" + "="*50)
    print("TF-IDF Feature Extraction")
    print("="*50)

    tfidf_extractor = TfidfFeatureExtractor(max_features=100, ngram_range=(1, 2))
    features = tfidf_extractor.fit_transform(cleaned)

    print(f"\nFeature matrix shape: {features.shape}")
    print(f"Sample feature vector (first 10 values):")
    print(features[0].toarray()[0][:10])


if __name__ == "__main__":
    main()
