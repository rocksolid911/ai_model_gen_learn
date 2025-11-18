"""
Training script for News Deep Learning Model (TensorFlow/Keras)
"""

import os
import sys
import matplotlib.pyplot as plt

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_loading.news_dataset import NewsDataset
from preprocessing.text_preprocessing import TextPreprocessor
from models.news_deep_tf import NewsDeepModel


def plot_training_history(history, save_path: str = None):
    """
    Plot training and validation metrics.

    Args:
        history: Training history from model.fit()
        save_path: Path to save plot (optional)
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Plot accuracy
    ax1.plot(history.history['accuracy'], label='Train Accuracy', marker='o')
    if 'val_accuracy' in history.history:
        ax1.plot(history.history['val_accuracy'], label='Val Accuracy', marker='o')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.set_title('Model Accuracy')
    ax1.legend()
    ax1.grid(True)

    # Plot loss
    ax2.plot(history.history['loss'], label='Train Loss', marker='o')
    if 'val_loss' in history.history:
        ax2.plot(history.history['val_loss'], label='Val Loss', marker='o')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.set_title('Model Loss')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Training plot saved to {save_path}")

    # Close plot to free memory
    plt.close()


def main():
    """Train deep learning news classifier."""
    print("="*70)
    print("Training News Deep Learning Model (TensorFlow/Keras)")
    print("="*70)

    # Configuration
    DATA_PATH = "data/news.csv"
    MODEL_SAVE_PATH = "models/news_deep_model.h5"
    TOKENIZER_SAVE_PATH = "models/news_tokenizer.pkl"
    PLOT_SAVE_PATH = "models/news_deep_training_history.png"

    # Hyperparameters
    VOCAB_SIZE = 10000
    EMBEDDING_DIM = 128
    MAX_LENGTH = 200
    ARCHITECTURE = 'cnn'  # 'cnn' or 'lstm'
    EPOCHS = 20
    BATCH_SIZE = 32

    # Check GPU availability
    import tensorflow as tf
    print("\n[GPU Check]")
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"✓ GPU available: {len(gpus)} device(s)")
        for gpu in gpus:
            print(f"  - {gpu}")
    else:
        print("✗ No GPU detected. Training will use CPU (slower).")

    # Step 1: Load and prepare data
    print("\n[Step 1/5] Loading data...")
    dataset = NewsDataset(csv_path=DATA_PATH)
    dataset.load_data()
    dataset.clean_data()

    # Split data
    train_df, val_df, test_df = dataset.prepare_splits(
        test_size=0.2,
        val_size=0.1,
        random_state=42
    )

    # Get texts and labels
    train_texts, train_labels = dataset.get_texts_and_labels(train_df)
    val_texts, val_labels = dataset.get_texts_and_labels(val_df)
    test_texts, test_labels = dataset.get_texts_and_labels(test_df)

    # Step 2: Preprocess text
    print("\n[Step 2/5] Preprocessing text...")
    preprocessor = TextPreprocessor(
        remove_stopwords=False,  # Keep all words for deep learning
        lowercase=True
    )

    # Clean texts (light preprocessing for deep learning)
    train_texts_clean = preprocessor.clean_texts(train_texts)
    val_texts_clean = preprocessor.clean_texts(val_texts)
    test_texts_clean = preprocessor.clean_texts(test_texts)

    print(f"Sample cleaned text: {train_texts_clean[0][:200]}...")

    # Step 3: Build model
    print("\n[Step 3/5] Building model...")
    model = NewsDeepModel(
        vocab_size=VOCAB_SIZE,
        embedding_dim=EMBEDDING_DIM,
        max_length=MAX_LENGTH,
        num_classes=4,
        architecture=ARCHITECTURE,
        random_state=42
    )

    model.build_model()

    # Step 4: Train model
    print("\n[Step 4/5] Training model...")
    print(f"Architecture: {ARCHITECTURE}")
    print(f"Epochs: {EPOCHS}, Batch size: {BATCH_SIZE}")
    print(f"Vocab size: {VOCAB_SIZE}, Embedding dim: {EMBEDDING_DIM}")

    history = model.train(
        train_texts=train_texts_clean,
        train_labels=train_labels,
        val_texts=val_texts_clean,
        val_labels=val_labels,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE
    )

    # Plot training history
    plot_training_history(history, save_path=PLOT_SAVE_PATH)

    # Step 5: Evaluate on test set
    print("\n[Step 5/5] Evaluating on test set...")

    # Make predictions
    test_preds = model.predict(test_texts_clean)

    # Calculate metrics
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
    import numpy as np

    test_accuracy = accuracy_score(test_labels, test_preds)
    print(f"\nTest Accuracy: {test_accuracy:.4f}")

    # Detailed classification report
    class_names = ['neutral', 'left', 'right', 'propaganda']
    print("\nClassification Report:")
    print(classification_report(
        test_labels,
        test_preds,
        target_names=class_names,
        digits=4
    ))

    # Confusion matrix
    print("\nConfusion Matrix:")
    cm = confusion_matrix(test_labels, test_preds)
    print(cm)

    # Save model
    print("\n[Saving Model]")
    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
    model.save(MODEL_SAVE_PATH, TOKENIZER_SAVE_PATH)

    # Print summary
    print("\n" + "="*70)
    print("Training Complete!")
    print("="*70)
    print(f"Model saved to: {MODEL_SAVE_PATH}")
    print(f"Tokenizer saved to: {TOKENIZER_SAVE_PATH}")
    print(f"Training plot saved to: {PLOT_SAVE_PATH}")
    print(f"\nFinal Results:")
    print(f"  Test Accuracy: {test_accuracy:.4f}")
    print(f"  Architecture: {ARCHITECTURE}")
    print(f"  Total Parameters: {model.model.count_params():,}")

    print("\nYou can now use this model for predictions with:")
    print("  python src/inference/predict_news.py --model deep --text 'your text'")


if __name__ == "__main__":
    main()
