"""
News Dataset Loader
Handles loading, cleaning, and splitting news data for classification.
"""

import os
from typing import Tuple, Optional
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


class NewsDataset:
    """
    Load and manage news article dataset for classification.

    The dataset should have columns: id, title, content, label
    Labels: neutral, left, right, propaganda
    """

    def __init__(self, csv_path: str = "data/news.csv"):
        """
        Initialize the news dataset loader.

        Args:
            csv_path: Path to the CSV file containing news data
        """
        self.csv_path = csv_path
        self.data: Optional[pd.DataFrame] = None
        self.label_mapping = {
            'neutral': 0,
            'left': 1,
            'right': 2,
            'propaganda': 3
        }
        self.reverse_label_mapping = {v: k for k, v in self.label_mapping.items()}

    def load_data(self) -> pd.DataFrame:
        """
        Load news data from CSV file.

        Returns:
            DataFrame with news articles

        Raises:
            FileNotFoundError: If CSV file doesn't exist
        """
        if not os.path.exists(self.csv_path):
            raise FileNotFoundError(
                f"News dataset not found at {self.csv_path}. "
                "Please run 'python scripts/generate_sample_data.py' first."
            )

        # Load CSV using pandas
        self.data = pd.read_csv(self.csv_path)

        print(f"Loaded {len(self.data)} news articles")
        print(f"Columns: {list(self.data.columns)}")

        return self.data

    def clean_data(self) -> pd.DataFrame:
        """
        Clean the dataset by handling missing values and duplicates.

        Why we clean:
        - Missing values can cause errors during training
        - Duplicates can bias the model
        - Consistent data format improves model performance

        Returns:
            Cleaned DataFrame
        """
        if self.data is None:
            self.load_data()

        print("\nCleaning data...")
        initial_count = len(self.data)

        # Remove rows with missing critical columns
        # Why: We need both text and label to train
        self.data = self.data.dropna(subset=['title', 'content', 'label'])

        # Fill empty strings with placeholder
        self.data['title'] = self.data['title'].fillna('')
        self.data['content'] = self.data['content'].fillna('')

        # Combine title and content for full text
        # Why: Both contain useful information for classification
        self.data['full_text'] = self.data['title'] + ' ' + self.data['content']

        # Remove duplicates based on content
        # Why: Duplicates can cause data leakage and overfitting
        self.data = self.data.drop_duplicates(subset=['full_text'])

        # Filter valid labels only
        valid_labels = set(self.label_mapping.keys())
        self.data = self.data[self.data['label'].isin(valid_labels)]

        print(f"Removed {initial_count - len(self.data)} invalid/duplicate rows")
        print(f"Final dataset size: {len(self.data)}")

        # Show label distribution
        print("\nLabel distribution:")
        print(self.data['label'].value_counts())

        return self.data

    def prepare_splits(
        self,
        test_size: float = 0.2,
        val_size: float = 0.1,
        random_state: int = 42
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split data into train, validation, and test sets.

        Why we split:
        - Train: Used to learn patterns (typically 70-80%)
        - Validation: Used to tune hyperparameters (typically 10-15%)
        - Test: Final evaluation, never seen during training (typically 10-20%)

        This prevents overfitting and gives honest performance estimates.

        Args:
            test_size: Proportion of data for test set
            val_size: Proportion of remaining data for validation set
            random_state: Random seed for reproducibility

        Returns:
            Tuple of (train_df, val_df, test_df)
        """
        if self.data is None:
            self.load_data()
            self.clean_data()

        # First split: separate test set
        # Why: Test set should never be touched until final evaluation
        train_val_df, test_df = train_test_split(
            self.data,
            test_size=test_size,
            random_state=random_state,
            stratify=self.data['label']  # Maintain label distribution
        )

        # Second split: separate validation from training
        # Calculate validation size relative to train_val data
        val_relative_size = val_size / (1 - test_size)
        train_df, val_df = train_test_split(
            train_val_df,
            test_size=val_relative_size,
            random_state=random_state,
            stratify=train_val_df['label']
        )

        print(f"\nData splits:")
        print(f"Train: {len(train_df)} samples ({len(train_df)/len(self.data)*100:.1f}%)")
        print(f"Validation: {len(val_df)} samples ({len(val_df)/len(self.data)*100:.1f}%)")
        print(f"Test: {len(test_df)} samples ({len(test_df)/len(self.data)*100:.1f}%)")

        return train_df, val_df, test_df

    def get_texts_and_labels(
        self,
        df: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract texts and labels from DataFrame.

        Args:
            df: DataFrame with 'full_text' and 'label' columns

        Returns:
            Tuple of (texts, labels) as numpy arrays
        """
        texts = df['full_text'].values

        # Convert string labels to integers
        # Why: Neural networks need numeric inputs
        labels = df['label'].map(self.label_mapping).values

        return texts, labels

    def label_to_int(self, label: str) -> int:
        """Convert string label to integer."""
        return self.label_mapping.get(label, -1)

    def int_to_label(self, label_int: int) -> str:
        """Convert integer label to string."""
        return self.reverse_label_mapping.get(label_int, 'unknown')


def main():
    """Demo usage of NewsDataset."""
    # Initialize dataset
    dataset = NewsDataset()

    # Load and clean data
    data = dataset.load_data()
    dataset.clean_data()

    # Split into train/val/test
    train_df, val_df, test_df = dataset.prepare_splits()

    # Get texts and labels
    train_texts, train_labels = dataset.get_texts_and_labels(train_df)

    print(f"\nSample text: {train_texts[0][:200]}...")
    print(f"Sample label: {train_labels[0]} ({dataset.int_to_label(train_labels[0])})")


if __name__ == "__main__":
    main()
