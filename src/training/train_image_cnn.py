"""
Training script for Image CNN Model (TensorFlow/Keras)
"""

import os
import sys
import matplotlib.pyplot as plt
import numpy as np

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_loading.image_dataset import ImageDataset
from preprocessing.image_preprocessing import ImagePreprocessor, create_tf_dataset
from models.image_cnn_tf import ImageCNN


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

    plt.close()


def main():
    """Train image CNN classifier."""
    print("="*70)
    print("Training Image CNN Model (TensorFlow/Keras)")
    print("="*70)

    # Configuration
    IMAGES_DIR = "data/images"
    MODEL_SAVE_PATH = "models/image_cnn_model.h5"
    PLOT_SAVE_PATH = "models/image_cnn_training_history.png"

    # Hyperparameters
    IMAGE_SIZE = (128, 128)  # (height, width)
    NORMALIZATION = 'standard'  # 'standard', 'imagenet', or 'centered'
    ARCHITECTURE = 'standard'  # 'simple', 'standard', or 'deep'
    EPOCHS = 30
    BATCH_SIZE = 32

    # Check GPU availability
    import tensorflow as tf
    print("\n[GPU Check]")
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"✓ GPU available: {len(gpus)} device(s)")
        for gpu in gpus:
            print(f"  - {gpu}")
        # Enable memory growth to prevent OOM errors
        for gpu in gpus:
            try:
                tf.config.experimental.set_memory_growth(gpu, True)
            except:
                pass
    else:
        print("✗ No GPU detected. Training will use CPU (slower).")

    # Step 1: Load and prepare data
    print("\n[Step 1/5] Loading data...")
    dataset = ImageDataset(images_dir=IMAGES_DIR)
    dataset.load_data()

    # Split data
    train_df, val_df, test_df = dataset.prepare_splits(
        test_size=0.2,
        val_size=0.1,
        random_state=42
    )

    # Get paths and labels
    train_paths, train_labels = dataset.get_paths_and_labels(train_df)
    val_paths, val_labels = dataset.get_paths_and_labels(val_df)
    test_paths, test_labels = dataset.get_paths_and_labels(test_df)

    num_classes = dataset.get_num_classes()
    print(f"Number of classes: {num_classes}")
    print(f"Class names: {dataset.class_names}")

    # Step 2: Create preprocessor and datasets
    print("\n[Step 2/5] Creating data pipelines...")
    preprocessor = ImagePreprocessor(
        target_size=IMAGE_SIZE,
        normalization=NORMALIZATION
    )

    # Create TensorFlow datasets
    # Training set with augmentation
    train_dataset = create_tf_dataset(
        train_paths,
        train_labels,
        preprocessor,
        batch_size=BATCH_SIZE,
        shuffle=True,
        augment=False  # Augmentation can be added in preprocessing
    )

    # Validation set (no augmentation)
    val_dataset = create_tf_dataset(
        val_paths,
        val_labels,
        preprocessor,
        batch_size=BATCH_SIZE,
        shuffle=False,
        augment=False
    )

    # Test set (no augmentation)
    test_dataset = create_tf_dataset(
        test_paths,
        test_labels,
        preprocessor,
        batch_size=BATCH_SIZE,
        shuffle=False,
        augment=False
    )

    print(f"Training batches: {len(train_dataset)}")
    print(f"Validation batches: {len(val_dataset)}")
    print(f"Test batches: {len(test_dataset)}")

    # Step 3: Build model
    print("\n[Step 3/5] Building model...")
    model = ImageCNN(
        input_shape=(*IMAGE_SIZE, 3),
        num_classes=num_classes,
        architecture=ARCHITECTURE,
        random_state=42
    )

    model.build_model()

    # Step 4: Train model
    print("\n[Step 4/5] Training model...")
    print(f"Architecture: {ARCHITECTURE}")
    print(f"Image size: {IMAGE_SIZE}")
    print(f"Epochs: {EPOCHS}, Batch size: {BATCH_SIZE}")
    print(f"Normalization: {NORMALIZATION}")

    history = model.train(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        epochs=EPOCHS,
        verbose=1
    )

    # Plot training history
    plot_training_history(history, save_path=PLOT_SAVE_PATH)

    # Step 5: Evaluate on test set
    print("\n[Step 5/5] Evaluating on test set...")

    # Evaluate using Keras evaluate
    test_loss, test_accuracy = model.model.evaluate(test_dataset, verbose=0)
    print(f"\nTest Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")

    # Get detailed predictions
    print("\nGenerating detailed metrics...")
    test_preds_probs = model.model.predict(test_dataset, verbose=0)
    test_preds = np.argmax(test_preds_probs, axis=1)

    # Classification report
    from sklearn.metrics import classification_report, confusion_matrix

    print("\nClassification Report:")
    print(classification_report(
        test_labels,
        test_preds,
        target_names=dataset.class_names,
        digits=4
    ))

    # Confusion matrix
    print("\nConfusion Matrix:")
    cm = confusion_matrix(test_labels, test_preds)
    print(cm)

    # Per-class accuracy
    print("\nPer-class Accuracy:")
    for i, class_name in enumerate(dataset.class_names):
        class_mask = test_labels == i
        class_acc = (test_preds[class_mask] == test_labels[class_mask]).mean()
        print(f"  {class_name}: {class_acc:.4f}")

    # Save model
    print("\n[Saving Model]")
    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
    model.save(MODEL_SAVE_PATH)

    # Save class names for later use
    import json
    class_names_path = "models/image_class_names.json"
    with open(class_names_path, 'w') as f:
        json.dump(dataset.class_names, f)
    print(f"Class names saved to {class_names_path}")

    # Print summary
    print("\n" + "="*70)
    print("Training Complete!")
    print("="*70)
    print(f"Model saved to: {MODEL_SAVE_PATH}")
    print(f"Training plot saved to: {PLOT_SAVE_PATH}")
    print(f"\nFinal Results:")
    print(f"  Test Accuracy: {test_accuracy:.4f}")
    print(f"  Test Loss: {test_loss:.4f}")
    print(f"  Architecture: {ARCHITECTURE}")
    print(f"  Total Parameters: {model.model.count_params():,}")
    print(f"  Image Size: {IMAGE_SIZE}")

    print("\nYou can now use this model for predictions with:")
    print("  python src/inference/predict_image.py --image path/to/image.jpg")


if __name__ == "__main__":
    main()
