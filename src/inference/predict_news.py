"""
Inference script for News Classification
Predict political leaning and propaganda in news articles.
"""

import os
import sys
import argparse

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from preprocessing.text_preprocessing import TextPreprocessor, TfidfFeatureExtractor
from models.news_baseline_sklearn import NewsBaselineModel
from models.news_deep_tf import NewsDeepModel


def predict_with_baseline(text: str, model_path: str, tfidf_path: str) -> tuple:
    """
    Make prediction using baseline model.

    Args:
        text: Input text
        model_path: Path to trained model
        tfidf_path: Path to TF-IDF vectorizer

    Returns:
        Tuple of (predicted_label, probabilities)
    """
    # Preprocess text
    preprocessor = TextPreprocessor(remove_stopwords=True, lowercase=True)
    text_clean = preprocessor.clean_text(text)

    # Load TF-IDF vectorizer and transform
    tfidf = TfidfFeatureExtractor()
    tfidf.load(tfidf_path)
    features = tfidf.transform([text_clean])

    # Load model and predict
    model = NewsBaselineModel(model_type='logistic')
    model.load(model_path)

    prediction = model.predict(features)[0]
    probabilities = model.predict_proba(features)[0]

    return prediction, probabilities


def predict_with_deep(text: str, model_path: str, tokenizer_path: str) -> tuple:
    """
    Make prediction using deep learning model.

    Args:
        text: Input text
        model_path: Path to trained model
        tokenizer_path: Path to tokenizer

    Returns:
        Tuple of (predicted_label, probabilities)
    """
    # Preprocess text (light preprocessing for deep learning)
    preprocessor = TextPreprocessor(remove_stopwords=False, lowercase=True)
    text_clean = preprocessor.clean_text(text)

    # Load model and tokenizer
    model = NewsDeepModel()
    model.load(model_path, tokenizer_path)

    # Make prediction
    prediction = model.predict([text_clean])[0]
    probabilities = model.predict_proba([text_clean])[0]

    return prediction, probabilities


def main():
    """Run news classification inference."""
    parser = argparse.ArgumentParser(
        description="Predict political leaning and propaganda in news articles"
    )

    parser.add_argument(
        '--text',
        type=str,
        help='News article text to classify'
    )

    parser.add_argument(
        '--file',
        type=str,
        help='Path to text file containing news article'
    )

    parser.add_argument(
        '--model',
        type=str,
        choices=['baseline', 'deep'],
        default='deep',
        help='Model to use: "baseline" (TF-IDF + Logistic) or "deep" (TensorFlow CNN/LSTM)'
    )

    args = parser.parse_args()

    # Get input text
    if args.text:
        text = args.text
    elif args.file:
        if not os.path.exists(args.file):
            print(f"Error: File not found: {args.file}")
            return
        with open(args.file, 'r', encoding='utf-8') as f:
            text = f.read()
    else:
        print("Error: Please provide either --text or --file argument")
        parser.print_help()
        return

    # Class names
    class_names = ['neutral', 'left', 'right', 'propaganda']

    print("="*70)
    print("News Classification - Prediction")
    print("="*70)
    print(f"\nModel: {args.model}")
    print(f"\nInput text ({len(text)} characters):")
    print("-" * 70)
    print(text[:500] + ("..." if len(text) > 500 else ""))
    print("-" * 70)

    # Make prediction based on model choice
    try:
        if args.model == 'baseline':
            model_path = "models/news_baseline_model.pkl"
            tfidf_path = "models/news_tfidf_vectorizer.pkl"

            if not os.path.exists(model_path):
                print(f"\nError: Baseline model not found at {model_path}")
                print("Please train the model first:")
                print("  python src/training/train_news_baseline.py")
                return

            print("\nRunning baseline model (TF-IDF + Logistic Regression)...")
            prediction, probabilities = predict_with_baseline(text, model_path, tfidf_path)

        else:  # deep
            model_path = "models/news_deep_model.h5"
            tokenizer_path = "models/news_tokenizer.pkl"

            if not os.path.exists(model_path):
                print(f"\nError: Deep learning model not found at {model_path}")
                print("Please train the model first:")
                print("  python src/training/train_news_tf.py")
                return

            print("\nRunning deep learning model (TensorFlow CNN/LSTM)...")
            prediction, probabilities = predict_with_deep(text, model_path, tokenizer_path)

        # Display results
        print("\n" + "="*70)
        print("PREDICTION RESULTS")
        print("="*70)

        predicted_class = class_names[prediction]
        confidence = probabilities[prediction] * 100

        print(f"\n🎯 Predicted Class: {predicted_class.upper()}")
        print(f"   Confidence: {confidence:.2f}%")

        print("\n📊 All Class Probabilities:")
        print("-" * 70)

        # Sort by probability
        sorted_indices = probabilities.argsort()[::-1]

        for idx in sorted_indices:
            class_name = class_names[idx]
            prob = probabilities[idx] * 100

            # Visual bar
            bar_length = int(prob / 2)  # Scale to 50 chars max
            bar = "█" * bar_length

            # Highlight predicted class
            marker = "→" if idx == prediction else " "

            print(f"{marker} {class_name:12s} {prob:6.2f}% {bar}")

        print("\n" + "="*70)

        # Interpretation
        print("\n💡 Interpretation:")
        if confidence > 80:
            print(f"   High confidence - The article is strongly classified as {predicted_class}.")
        elif confidence > 60:
            print(f"   Moderate confidence - The article leans toward {predicted_class}.")
        else:
            print(f"   Low confidence - The classification is uncertain.")
            print(f"   Consider the top {min(2, len(class_names))} classes.")

        if predicted_class == 'propaganda':
            print("\n⚠️  Warning: This article may contain propaganda or biased language.")
        elif predicted_class == 'neutral':
            print("\n✓  This article appears to be relatively neutral in tone.")

    except Exception as e:
        print(f"\nError during prediction: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
