"""
Evaluation script for comparing News Classification Models
Compares baseline (sklearn) and deep learning (TensorFlow) models.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    confusion_matrix, classification_report
)

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_loading.news_dataset import NewsDataset
from preprocessing.text_preprocessing import TextPreprocessor, TfidfFeatureExtractor
from models.news_baseline_sklearn import NewsBaselineModel
from models.news_deep_tf import NewsDeepModel


def plot_confusion_matrix(cm, class_names, title, save_path=None):
    """
    Plot confusion matrix heatmap.

    Args:
        cm: Confusion matrix
        class_names: List of class names
        title: Plot title
        save_path: Path to save plot
    """
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names
    )
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix saved to {save_path}")

    plt.close()


def compare_models_plot(baseline_metrics, deep_metrics, class_names, save_path=None):
    """
    Create comparison plot of model metrics.

    Args:
        baseline_metrics: Dict with baseline model metrics
        deep_metrics: Dict with deep model metrics
        class_names: List of class names
        save_path: Path to save plot
    """
    metrics = ['precision', 'recall', 'f1']
    x = np.arange(len(class_names))
    width = 0.15

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for idx, metric in enumerate(metrics):
        ax = axes[idx]

        baseline_vals = baseline_metrics[metric]
        deep_vals = deep_metrics[metric]

        ax.bar(x - width/2, baseline_vals, width, label='Baseline (TF-IDF + Logistic)', alpha=0.8)
        ax.bar(x + width/2, deep_vals, width, label='Deep Learning (CNN/LSTM)', alpha=0.8)

        ax.set_xlabel('Class')
        ax.set_ylabel(metric.capitalize())
        ax.set_title(f'{metric.capitalize()} Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels(class_names, rotation=45)
        ax.legend()
        ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Comparison plot saved to {save_path}")

    plt.close()


def main():
    """Evaluate and compare news classification models."""
    print("="*70)
    print("Evaluating News Classification Models")
    print("="*70)

    # Configuration
    DATA_PATH = "data/news.csv"
    BASELINE_MODEL_PATH = "models/news_baseline_model.pkl"
    TFIDF_PATH = "models/news_tfidf_vectorizer.pkl"
    DEEP_MODEL_PATH = "models/news_deep_model.h5"
    TOKENIZER_PATH = "models/news_tokenizer.pkl"

    class_names = ['neutral', 'left', 'right', 'propaganda']

    # Check if models exist
    if not os.path.exists(BASELINE_MODEL_PATH):
        print(f"Error: Baseline model not found at {BASELINE_MODEL_PATH}")
        print("Please train the baseline model first:")
        print("  python src/training/train_news_baseline.py")
        return

    if not os.path.exists(DEEP_MODEL_PATH):
        print(f"Error: Deep model not found at {DEEP_MODEL_PATH}")
        print("Please train the deep model first:")
        print("  python src/training/train_news_tf.py")
        return

    # Load data
    print("\n[Step 1/4] Loading test data...")
    dataset = NewsDataset(csv_path=DATA_PATH)
    dataset.load_data()
    dataset.clean_data()

    train_df, val_df, test_df = dataset.prepare_splits(
        test_size=0.2,
        val_size=0.1,
        random_state=42
    )

    test_texts, test_labels = dataset.get_texts_and_labels(test_df)
    print(f"Test samples: {len(test_texts)}")

    # Evaluate Baseline Model
    print("\n[Step 2/4] Evaluating Baseline Model (TF-IDF + Logistic)...")
    print("-" * 70)

    preprocessor = TextPreprocessor(remove_stopwords=True, lowercase=True)
    test_texts_clean = preprocessor.clean_texts(test_texts)

    # Load TF-IDF vectorizer and transform
    tfidf = TfidfFeatureExtractor()
    tfidf.load(TFIDF_PATH)
    X_test = tfidf.transform(test_texts_clean)

    # Load and evaluate baseline model
    baseline_model = NewsBaselineModel(model_type='logistic')
    baseline_model.load(BASELINE_MODEL_PATH)

    baseline_preds = baseline_model.predict(X_test)
    baseline_probs = baseline_model.predict_proba(X_test)

    baseline_acc = accuracy_score(test_labels, baseline_preds)
    print(f"\nBaseline Test Accuracy: {baseline_acc:.4f}")

    print("\nBaseline Classification Report:")
    print(classification_report(
        test_labels,
        baseline_preds,
        target_names=class_names,
        digits=4
    ))

    baseline_cm = confusion_matrix(test_labels, baseline_preds)
    plot_confusion_matrix(
        baseline_cm,
        class_names,
        "Baseline Model - Confusion Matrix",
        save_path="models/baseline_confusion_matrix.png"
    )

    # Get per-class metrics
    baseline_precision, baseline_recall, baseline_f1, _ = precision_recall_fscore_support(
        test_labels, baseline_preds, average=None
    )

    # Evaluate Deep Learning Model
    print("\n[Step 3/4] Evaluating Deep Learning Model (TensorFlow)...")
    print("-" * 70)

    # Light preprocessing for deep learning
    preprocessor_dl = TextPreprocessor(remove_stopwords=False, lowercase=True)
    test_texts_clean_dl = preprocessor_dl.clean_texts(test_texts)

    # Load deep learning model
    deep_model = NewsDeepModel()
    deep_model.load(DEEP_MODEL_PATH, TOKENIZER_PATH)

    deep_preds = deep_model.predict(test_texts_clean_dl)
    deep_probs = deep_model.predict_proba(test_texts_clean_dl)

    deep_acc = accuracy_score(test_labels, deep_preds)
    print(f"\nDeep Learning Test Accuracy: {deep_acc:.4f}")

    print("\nDeep Learning Classification Report:")
    print(classification_report(
        test_labels,
        deep_preds,
        target_names=class_names,
        digits=4
    ))

    deep_cm = confusion_matrix(test_labels, deep_preds)
    plot_confusion_matrix(
        deep_cm,
        class_names,
        "Deep Learning Model - Confusion Matrix",
        save_path="models/deep_confusion_matrix.png"
    )

    # Get per-class metrics
    deep_precision, deep_recall, deep_f1, _ = precision_recall_fscore_support(
        test_labels, deep_preds, average=None
    )

    # Compare Models
    print("\n[Step 4/4] Model Comparison")
    print("="*70)

    comparison = []
    comparison.append(["Model", "Accuracy", "Avg Precision", "Avg Recall", "Avg F1"])
    comparison.append(["-" * 20] * 5)

    baseline_metrics = {
        'precision': baseline_precision,
        'recall': baseline_recall,
        'f1': baseline_f1
    }

    deep_metrics = {
        'precision': deep_precision,
        'recall': deep_recall,
        'f1': deep_f1
    }

    comparison.append([
        "Baseline (TF-IDF + Logistic)",
        f"{baseline_acc:.4f}",
        f"{baseline_precision.mean():.4f}",
        f"{baseline_recall.mean():.4f}",
        f"{baseline_f1.mean():.4f}"
    ])

    comparison.append([
        "Deep Learning (CNN/LSTM)",
        f"{deep_acc:.4f}",
        f"{deep_precision.mean():.4f}",
        f"{deep_recall.mean():.4f}",
        f"{deep_f1.mean():.4f}"
    ])

    # Print comparison table
    print("\nOverall Metrics:")
    for row in comparison:
        print(f"{row[0]:30s} {row[1]:12s} {row[2]:15s} {row[3]:12s} {row[4]:10s}")

    # Per-class comparison
    print("\n\nPer-Class Comparison:")
    print(f"{'Class':<15s} {'Baseline F1':>15s} {'Deep Learning F1':>20s} {'Difference':>15s}")
    print("-" * 70)

    for i, class_name in enumerate(class_names):
        diff = deep_f1[i] - baseline_f1[i]
        symbol = "↑" if diff > 0 else "↓" if diff < 0 else "="
        print(f"{class_name:<15s} {baseline_f1[i]:>15.4f} {deep_f1[i]:>20.4f} {diff:>14.4f} {symbol}")

    # Create comparison plot
    compare_models_plot(
        baseline_metrics,
        deep_metrics,
        class_names,
        save_path="models/model_comparison.png"
    )

    # Analyze errors
    print("\n\nError Analysis:")
    print("-" * 70)

    # Find samples where both models failed
    baseline_errors = baseline_preds != test_labels
    deep_errors = deep_preds != test_labels
    both_errors = baseline_errors & deep_errors

    print(f"Baseline errors: {baseline_errors.sum()} ({baseline_errors.mean()*100:.1f}%)")
    print(f"Deep learning errors: {deep_errors.sum()} ({deep_errors.mean()*100:.1f}%)")
    print(f"Both models failed: {both_errors.sum()} ({both_errors.mean()*100:.1f}%)")

    # Find hard examples (both models failed)
    if both_errors.sum() > 0:
        print("\nExample of difficult samples (both models failed):")
        hard_indices = np.where(both_errors)[0][:3]  # Show first 3
        for idx in hard_indices:
            print(f"\nSample {idx}:")
            print(f"  Text: {test_texts[idx][:150]}...")
            print(f"  True: {class_names[test_labels[idx]]}")
            print(f"  Baseline predicted: {class_names[baseline_preds[idx]]}")
            print(f"  Deep predicted: {class_names[deep_preds[idx]]}")

    # Summary
    print("\n" + "="*70)
    print("Evaluation Summary")
    print("="*70)

    if deep_acc > baseline_acc:
        improvement = (deep_acc - baseline_acc) * 100
        print(f"✓ Deep learning model outperforms baseline by {improvement:.2f}% accuracy")
    elif baseline_acc > deep_acc:
        difference = (baseline_acc - deep_acc) * 100
        print(f"✓ Baseline model outperforms deep learning by {difference:.2f}% accuracy")
    else:
        print("✓ Both models achieve similar accuracy")

    print(f"\nBest performing model: {'Deep Learning' if deep_acc > baseline_acc else 'Baseline'}")
    print(f"Best accuracy: {max(baseline_acc, deep_acc):.4f}")


if __name__ == "__main__":
    main()
