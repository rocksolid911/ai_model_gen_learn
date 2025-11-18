"""
Inference script for Image Classification
Predict image class using trained CNN model.
"""

import os
import sys
import argparse
import json
import numpy as np

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from preprocessing.image_preprocessing import ImagePreprocessor
from models.image_cnn_tf import ImageCNN


def predict_image(
    image_path: str,
    model_path: str,
    class_names: list,
    image_size: tuple = (128, 128),
    normalization: str = 'standard',
    top_k: int = 5
) -> tuple:
    """
    Predict image class.

    Args:
        image_path: Path to image file
        model_path: Path to trained model
        class_names: List of class names
        image_size: Target image size
        normalization: Normalization method
        top_k: Number of top predictions to return

    Returns:
        Tuple of (predictions, probabilities)
    """
    # Load and preprocess image
    preprocessor = ImagePreprocessor(
        target_size=image_size,
        normalization=normalization
    )

    image = preprocessor.load_image(image_path)
    image_batch = np.expand_dims(image, axis=0)  # Add batch dimension

    # Load model
    model = ImageCNN(
        input_shape=(*image_size, 3),
        num_classes=len(class_names)
    )
    model.load(model_path)

    # Make prediction
    probabilities = model.predict_proba(image_batch)[0]

    # Get top-k predictions
    top_k = min(top_k, len(class_names))
    top_indices = probabilities.argsort()[-top_k:][::-1]

    predictions = []
    for idx in top_indices:
        predictions.append({
            'class': class_names[idx],
            'class_index': int(idx),
            'probability': float(probabilities[idx])
        })

    return predictions, image


def visualize_prediction(image: np.ndarray, predictions: list, save_path: str = None):
    """
    Visualize image with predictions.

    Args:
        image: Image array
        predictions: List of prediction dictionaries
        save_path: Path to save visualization
    """
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Display image
    # Denormalize for display
    img_display = image
    if img_display.max() <= 1.0:
        img_display = (img_display * 255).astype(np.uint8)

    ax1.imshow(img_display)
    ax1.set_title('Input Image', fontsize=14, fontweight='bold')
    ax1.axis('off')

    # Display predictions as horizontal bar chart
    classes = [p['class'] for p in predictions]
    probs = [p['probability'] * 100 for p in predictions]

    colors = ['green' if i == 0 else 'skyblue' for i in range(len(classes))]

    ax2.barh(range(len(classes)), probs, color=colors, alpha=0.8)
    ax2.set_yticks(range(len(classes)))
    ax2.set_yticklabels(classes)
    ax2.set_xlabel('Probability (%)', fontsize=12)
    ax2.set_title('Top Predictions', fontsize=14, fontweight='bold')
    ax2.set_xlim(0, 100)
    ax2.grid(axis='x', alpha=0.3)

    # Add percentage labels
    for i, (class_name, prob) in enumerate(zip(classes, probs)):
        ax2.text(prob + 1, i, f'{prob:.1f}%', va='center', fontsize=10)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\nVisualization saved to {save_path}")

    plt.show()


def main():
    """Run image classification inference."""
    parser = argparse.ArgumentParser(
        description="Predict image class using trained CNN model"
    )

    parser.add_argument(
        '--image',
        type=str,
        required=True,
        help='Path to image file'
    )

    parser.add_argument(
        '--model',
        type=str,
        default='models/image_cnn_model.h5',
        help='Path to trained model (default: models/image_cnn_model.h5)'
    )

    parser.add_argument(
        '--classes',
        type=str,
        default='models/image_class_names.json',
        help='Path to class names JSON file'
    )

    parser.add_argument(
        '--top-k',
        type=int,
        default=5,
        help='Number of top predictions to show (default: 5)'
    )

    parser.add_argument(
        '--visualize',
        action='store_true',
        help='Show visualization of prediction'
    )

    parser.add_argument(
        '--save-viz',
        type=str,
        help='Path to save visualization'
    )

    args = parser.parse_args()

    # Check if image exists
    if not os.path.exists(args.image):
        print(f"Error: Image not found: {args.image}")
        return

    # Check if model exists
    if not os.path.exists(args.model):
        print(f"Error: Model not found: {args.model}")
        print("Please train the model first:")
        print("  python src/training/train_image_cnn.py")
        return

    # Load class names
    if os.path.exists(args.classes):
        with open(args.classes, 'r') as f:
            class_names = json.load(f)
    else:
        print(f"Warning: Class names file not found at {args.classes}")
        print("Using generic class names...")
        # Try to infer number of classes from model
        # For now, use placeholder names
        class_names = [f"class_{i}" for i in range(10)]  # Placeholder

    print("="*70)
    print("Image Classification - Prediction")
    print("="*70)
    print(f"\nImage: {args.image}")
    print(f"Model: {args.model}")

    # Make prediction
    try:
        print("\nProcessing image and making prediction...")

        predictions, image = predict_image(
            image_path=args.image,
            model_path=args.model,
            class_names=class_names,
            image_size=(128, 128),
            normalization='standard',
            top_k=args.top_k
        )

        # Display results
        print("\n" + "="*70)
        print("PREDICTION RESULTS")
        print("="*70)

        top_pred = predictions[0]
        print(f"\n🎯 Top Prediction: {top_pred['class'].upper()}")
        print(f"   Confidence: {top_pred['probability']*100:.2f}%")

        print(f"\n📊 Top-{len(predictions)} Predictions:")
        print("-" * 70)

        for i, pred in enumerate(predictions, 1):
            class_name = pred['class']
            prob = pred['probability'] * 100

            # Visual bar
            bar_length = int(prob / 2)  # Scale to 50 chars max
            bar = "█" * bar_length

            # Marker for top prediction
            marker = "→" if i == 1 else " "

            print(f"{marker} {i}. {class_name:20s} {prob:6.2f}% {bar}")

        print("\n" + "="*70)

        # Interpretation
        print("\n💡 Interpretation:")
        confidence = top_pred['probability'] * 100

        if confidence > 90:
            print(f"   Very high confidence - The image is almost certainly a {top_pred['class']}.")
        elif confidence > 75:
            print(f"   High confidence - The image is very likely a {top_pred['class']}.")
        elif confidence > 50:
            print(f"   Moderate confidence - The image is probably a {top_pred['class']}.")
        else:
            print(f"   Low confidence - The model is uncertain about this image.")
            print(f"   Consider the top predictions as possibilities.")

        # Visualize if requested
        if args.visualize or args.save_viz:
            print("\nGenerating visualization...")
            visualize_prediction(
                image,
                predictions,
                save_path=args.save_viz
            )

    except Exception as e:
        print(f"\nError during prediction: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
