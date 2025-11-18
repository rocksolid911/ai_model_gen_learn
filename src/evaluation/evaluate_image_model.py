"""
Evaluation script for Image CNN Model
Provides detailed analysis of CNN performance on test set.
"""

import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    confusion_matrix, classification_report
)

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_loading.image_dataset import ImageDataset
from preprocessing.image_preprocessing import ImagePreprocessor, create_tf_dataset
from models.image_cnn_tf import ImageCNN


def plot_confusion_matrix(cm, class_names, save_path=None):
    """
    Plot confusion matrix heatmap.

    Args:
        cm: Confusion matrix
        class_names: List of class names
        save_path: Path to save plot
    """
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={'label': 'Count'}
    )
    plt.title('Confusion Matrix - Image CNN', fontsize=14, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix saved to {save_path}")

    plt.close()


def plot_per_class_metrics(precision, recall, f1, class_names, save_path=None):
    """
    Plot per-class metrics as bar chart.

    Args:
        precision: Per-class precision scores
        recall: Per-class recall scores
        f1: Per-class F1 scores
        class_names: List of class names
        save_path: Path to save plot
    """
    x = np.arange(len(class_names))
    width = 0.25

    fig, ax = plt.subplots(figsize=(12, 6))

    ax.bar(x - width, precision, width, label='Precision', alpha=0.8)
    ax.bar(x, recall, width, label='Recall', alpha=0.8)
    ax.bar(x + width, f1, width, label='F1-Score', alpha=0.8)

    ax.set_xlabel('Class', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Per-Class Performance Metrics', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(0, 1.1)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Per-class metrics plot saved to {save_path}")

    plt.close()


def visualize_sample_predictions(
    images,
    true_labels,
    pred_labels,
    pred_probs,
    class_names,
    num_samples=9,
    save_path=None
):
    """
    Visualize sample predictions with confidence scores.

    Args:
        images: Array of images
        true_labels: True labels
        pred_labels: Predicted labels
        pred_probs: Prediction probabilities
        class_names: List of class names
        num_samples: Number of samples to show
        save_path: Path to save plot
    """
    num_samples = min(num_samples, len(images))
    rows = int(np.sqrt(num_samples))
    cols = int(np.ceil(num_samples / rows))

    fig, axes = plt.subplots(rows, cols, figsize=(cols*3, rows*3))
    axes = axes.flatten() if num_samples > 1 else [axes]

    for i in range(num_samples):
        ax = axes[i]

        # Denormalize image for display (assuming [0, 1] normalization)
        img = images[i]
        if img.max() <= 1.0:
            img = (img * 255).astype(np.uint8)
        else:
            img = img.astype(np.uint8)

        ax.imshow(img)

        true_class = class_names[true_labels[i]]
        pred_class = class_names[pred_labels[i]]
        confidence = pred_probs[i][pred_labels[i]] * 100

        # Color: green if correct, red if wrong
        color = 'green' if true_labels[i] == pred_labels[i] else 'red'

        title = f"True: {true_class}\nPred: {pred_class}\nConf: {confidence:.1f}%"
        ax.set_title(title, fontsize=10, color=color, fontweight='bold')
        ax.axis('off')

    # Hide empty subplots
    for i in range(num_samples, len(axes)):
        axes[i].axis('off')

    plt.suptitle('Sample Predictions', fontsize=14, fontweight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Sample predictions saved to {save_path}")

    plt.close()


def main():
    """Evaluate image CNN model."""
    print("="*70)
    print("Evaluating Image CNN Model")
    print("="*70)

    # Configuration
    IMAGES_DIR = "data/images"
    MODEL_PATH = "models/image_cnn_model.h5"
    CLASS_NAMES_PATH = "models/image_class_names.json"
    IMAGE_SIZE = (128, 128)
    NORMALIZATION = 'standard'

    # Check if model exists
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model not found at {MODEL_PATH}")
        print("Please train the model first:")
        print("  python src/training/train_image_cnn.py")
        return

    # Load class names
    if os.path.exists(CLASS_NAMES_PATH):
        with open(CLASS_NAMES_PATH, 'r') as f:
            class_names = json.load(f)
    else:
        print("Warning: Class names file not found. Using default names.")
        class_names = None

    # Load data
    print("\n[Step 1/4] Loading test data...")
    dataset = ImageDataset(images_dir=IMAGES_DIR)
    dataset.load_data()

    if class_names is None:
        class_names = dataset.class_names

    train_df, val_df, test_df = dataset.prepare_splits(
        test_size=0.2,
        val_size=0.1,
        random_state=42
    )

    test_paths, test_labels = dataset.get_paths_and_labels(test_df)
    print(f"Test samples: {len(test_paths)}")
    print(f"Classes: {class_names}")

    # Create preprocessor and test dataset
    print("\n[Step 2/4] Preprocessing test data...")
    preprocessor = ImagePreprocessor(
        target_size=IMAGE_SIZE,
        normalization=NORMALIZATION
    )

    test_dataset = create_tf_dataset(
        test_paths,
        test_labels,
        preprocessor,
        batch_size=32,
        shuffle=False,
        augment=False
    )

    # Load model
    print("\n[Step 3/4] Loading model and making predictions...")
    model = ImageCNN(
        input_shape=(*IMAGE_SIZE, 3),
        num_classes=len(class_names)
    )
    model.load(MODEL_PATH)

    # Evaluate
    test_loss, test_accuracy = model.model.evaluate(test_dataset, verbose=0)
    print(f"\nTest Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")

    # Get predictions
    print("\nGenerating detailed predictions...")
    pred_probs = model.model.predict(test_dataset, verbose=0)
    pred_labels = np.argmax(pred_probs, axis=1)

    # Analyze results
    print("\n[Step 4/4] Analyzing results...")
    print("="*70)

    # Classification report
    print("\nClassification Report:")
    print(classification_report(
        test_labels,
        pred_labels,
        target_names=class_names,
        digits=4
    ))

    # Confusion matrix
    cm = confusion_matrix(test_labels, pred_labels)
    print("\nConfusion Matrix:")
    print(cm)

    plot_confusion_matrix(
        cm,
        class_names,
        save_path="models/image_cnn_confusion_matrix.png"
    )

    # Per-class metrics
    precision, recall, f1, support = precision_recall_fscore_support(
        test_labels,
        pred_labels,
        average=None
    )

    print("\n\nPer-Class Detailed Metrics:")
    print(f"{'Class':<20s} {'Precision':>12s} {'Recall':>12s} {'F1-Score':>12s} {'Support':>10s}")
    print("-" * 70)

    for i, class_name in enumerate(class_names):
        print(f"{class_name:<20s} {precision[i]:>12.4f} {recall[i]:>12.4f} "
              f"{f1[i]:>12.4f} {support[i]:>10d}")

    # Macro averages
    print("-" * 70)
    print(f"{'Macro Average':<20s} {precision.mean():>12.4f} {recall.mean():>12.4f} "
          f"{f1.mean():>12.4f} {support.sum():>10d}")

    # Plot per-class metrics
    plot_per_class_metrics(
        precision,
        recall,
        f1,
        class_names,
        save_path="models/image_cnn_per_class_metrics.png"
    )

    # Top-k accuracy
    print("\n\nTop-K Accuracy:")
    for k in [1, 3, 5]:
        if k > len(class_names):
            break
        top_k_preds = np.argsort(pred_probs, axis=1)[:, -k:]
        top_k_acc = np.mean([test_labels[i] in top_k_preds[i] for i in range(len(test_labels))])
        print(f"  Top-{k} Accuracy: {top_k_acc:.4f}")

    # Error analysis
    print("\n\nError Analysis:")
    print("-" * 70)

    errors = pred_labels != test_labels
    num_errors = errors.sum()
    error_rate = num_errors / len(test_labels)

    print(f"Total errors: {num_errors} / {len(test_labels)} ({error_rate*100:.2f}%)")

    # Most confused class pairs
    print("\nMost Confused Class Pairs:")
    confused_pairs = []
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            if i != j:
                count = cm[i, j]
                if count > 0:
                    confused_pairs.append((class_names[i], class_names[j], count))

    confused_pairs.sort(key=lambda x: x[2], reverse=True)
    for true_class, pred_class, count in confused_pairs[:5]:
        print(f"  {true_class} → {pred_class}: {count} times")

    # Visualize sample predictions
    print("\n\nVisualizing sample predictions...")

    # Load a few images for visualization
    sample_indices = np.random.choice(len(test_paths), size=min(9, len(test_paths)), replace=False)
    sample_images = []

    for idx in sample_indices:
        img = preprocessor.load_image(test_paths[idx])
        sample_images.append(img)

    sample_images = np.array(sample_images)
    sample_true_labels = test_labels[sample_indices]
    sample_pred_labels = pred_labels[sample_indices]
    sample_pred_probs = pred_probs[sample_indices]

    visualize_sample_predictions(
        sample_images,
        sample_true_labels,
        sample_pred_labels,
        sample_pred_probs,
        class_names,
        num_samples=9,
        save_path="models/image_cnn_sample_predictions.png"
    )

    # Summary
    print("\n" + "="*70)
    print("Evaluation Summary")
    print("="*70)
    print(f"Model: Image CNN")
    print(f"Test Accuracy: {test_accuracy:.4f}")
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Total Parameters: {model.model.count_params():,}")
    print(f"\nBest performing class: {class_names[np.argmax(f1)]} (F1: {f1.max():.4f})")
    print(f"Worst performing class: {class_names[np.argmin(f1)]} (F1: {f1.min():.4f})")

    print("\nGenerated plots:")
    print("  - models/image_cnn_confusion_matrix.png")
    print("  - models/image_cnn_per_class_metrics.png")
    print("  - models/image_cnn_sample_predictions.png")


if __name__ == "__main__":
    main()
