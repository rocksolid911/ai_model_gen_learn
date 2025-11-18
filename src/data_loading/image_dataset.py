"""
Image Dataset Loader
Handles loading and organizing image data for classification.
"""

import os
from typing import List, Tuple, Optional
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split


class ImageDataset:
    """
    Load and manage image dataset for classification.

    Expected structure:
    data/images/
        class_1/
            img1.jpg
            img2.jpg
        class_2/
            img1.jpg
        ...
    """

    def __init__(self, images_dir: str = "data/images"):
        """
        Initialize the image dataset loader.

        Args:
            images_dir: Root directory containing class subdirectories
        """
        self.images_dir = Path(images_dir)
        self.df: Optional[pd.DataFrame] = None
        self.class_names: List[str] = []
        self.class_to_idx = {}
        self.idx_to_class = {}

    def load_data(self) -> pd.DataFrame:
        """
        Load image file paths and create a DataFrame index.

        Why use pandas for images:
        - Easy to manage file paths and labels together
        - Simple train/val/test splitting
        - Easy to shuffle and sample data

        Returns:
            DataFrame with columns: file_path, class_name, label

        Raises:
            FileNotFoundError: If images directory doesn't exist
        """
        if not self.images_dir.exists():
            raise FileNotFoundError(
                f"Images directory not found at {self.images_dir}. "
                "Please run 'python scripts/generate_sample_data.py' first."
            )

        # Find all class subdirectories
        class_dirs = [d for d in self.images_dir.iterdir() if d.is_dir()]

        if len(class_dirs) == 0:
            raise ValueError(
                f"No class subdirectories found in {self.images_dir}. "
                "Please organize images into class folders."
            )

        # Collect all image paths and labels
        data_records = []
        valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif'}

        for class_dir in sorted(class_dirs):
            class_name = class_dir.name

            # Find all image files in this class directory
            for img_path in class_dir.iterdir():
                if img_path.suffix.lower() in valid_extensions:
                    data_records.append({
                        'file_path': str(img_path),
                        'class_name': class_name
                    })

        # Create DataFrame using pandas
        self.df = pd.DataFrame(data_records)

        # Create label mappings
        # Why: Neural networks need integer labels, not strings
        self.class_names = sorted(self.df['class_name'].unique())
        self.class_to_idx = {name: idx for idx, name in enumerate(self.class_names)}
        self.idx_to_class = {idx: name for name, idx in self.class_to_idx.items()}

        # Add integer label column
        self.df['label'] = self.df['class_name'].map(self.class_to_idx)

        print(f"Loaded {len(self.df)} images")
        print(f"Found {len(self.class_names)} classes: {self.class_names}")
        print("\nClass distribution:")
        print(self.df['class_name'].value_counts())

        return self.df

    def prepare_splits(
        self,
        test_size: float = 0.2,
        val_size: float = 0.1,
        random_state: int = 42
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split data into train, validation, and test sets.

        Why we split:
        - Train: Learn visual patterns (70-80%)
        - Validation: Tune model architecture and hyperparameters (10-15%)
        - Test: Unbiased final evaluation (10-20%)

        Args:
            test_size: Proportion for test set
            val_size: Proportion of remaining data for validation
            random_state: Random seed for reproducibility

        Returns:
            Tuple of (train_df, val_df, test_df)
        """
        if self.df is None:
            self.load_data()

        # First split: separate test set
        train_val_df, test_df = train_test_split(
            self.df,
            test_size=test_size,
            random_state=random_state,
            stratify=self.df['label']  # Keep class proportions balanced
        )

        # Second split: separate validation from training
        val_relative_size = val_size / (1 - test_size)
        train_df, val_df = train_test_split(
            train_val_df,
            test_size=val_relative_size,
            random_state=random_state,
            stratify=train_val_df['label']
        )

        print(f"\nData splits:")
        print(f"Train: {len(train_df)} images ({len(train_df)/len(self.df)*100:.1f}%)")
        print(f"Validation: {len(val_df)} images ({len(val_df)/len(self.df)*100:.1f}%)")
        print(f"Test: {len(test_df)} images ({len(test_df)/len(self.df)*100:.1f}%)")

        return train_df, val_df, test_df

    def get_paths_and_labels(
        self,
        df: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract file paths and labels from DataFrame.

        Args:
            df: DataFrame with 'file_path' and 'label' columns

        Returns:
            Tuple of (file_paths, labels) as numpy arrays
        """
        file_paths = df['file_path'].values
        labels = df['label'].values

        return file_paths, labels

    def get_num_classes(self) -> int:
        """Get the number of classes in the dataset."""
        return len(self.class_names)

    def class_name_to_idx(self, class_name: str) -> int:
        """Convert class name to integer index."""
        return self.class_to_idx.get(class_name, -1)

    def idx_to_class_name(self, idx: int) -> str:
        """Convert integer index to class name."""
        return self.idx_to_class.get(idx, 'unknown')


def main():
    """Demo usage of ImageDataset."""
    # Initialize dataset
    dataset = ImageDataset()

    # Load data
    df = dataset.load_data()

    # Split into train/val/test
    train_df, val_df, test_df = dataset.prepare_splits()

    # Get paths and labels
    train_paths, train_labels = dataset.get_paths_and_labels(train_df)

    print(f"\nSample path: {train_paths[0]}")
    print(f"Sample label: {train_labels[0]} ({dataset.idx_to_class_name(train_labels[0])})")

    print(f"\nNumber of classes: {dataset.get_num_classes()}")


if __name__ == "__main__":
    main()
