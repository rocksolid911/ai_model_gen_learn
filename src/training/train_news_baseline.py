"""
Training script for News Baseline Model (Sklearn + TF-IDF)
"""

import os
import sys

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_loading.news_dataset import NewsDataset
from preprocessing.text_preprocessing import TextPreprocessor, TfidfFeatureExtractor
from models.news_baseline_sklearn import NewsBaselineModel


def main():
    """Train baseline news classifier."""
    print("="*70)
    print("Training News Baseline Model (Sklearn + TF-IDF)")
    print("="*70)

    # Configuration
    DATA_PATH = "data/news.csv"
    MODEL_SAVE_PATH = "models/news_baseline_model.pkl"
    TFIDF_SAVE_PATH = "models/news_tfidf_vectorizer.pkl"

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
        remove_stopwords=True,
        lowercase=True
    )

    # Clean all texts
    train_texts_clean = preprocessor.clean_texts(train_texts)
    val_texts_clean = preprocessor.clean_texts(val_texts)
    test_texts_clean = preprocessor.clean_texts(test_texts)

    print(f"Sample cleaned text: {train_texts_clean[0][:200]}...")

    # Step 3: Extract TF-IDF features
    print("\n[Step 3/5] Extracting TF-IDF features...")
    tfidf_extractor = TfidfFeatureExtractor(
        max_features=5000,
        ngram_range=(1, 2)  # Unigrams and bigrams
    )

    # Fit on training data and transform
    X_train = tfidf_extractor.fit_transform(train_texts_clean)
    X_val = tfidf_extractor.transform(val_texts_clean)
    X_test = tfidf_extractor.transform(test_texts_clean)

    print(f"Training features shape: {X_train.shape}")

    # Save TF-IDF vectorizer
    os.makedirs(os.path.dirname(TFIDF_SAVE_PATH), exist_ok=True)
    tfidf_extractor.save(TFIDF_SAVE_PATH)

    # Step 4: Train model
    print("\n[Step 4/5] Training model...")
    model = NewsBaselineModel(
        model_type='logistic',  # Can also use 'svm'
        max_iter=1000,
        random_state=42
    )

    model.train(
        X_train, train_labels,
        X_val, val_labels
    )

    # Step 5: Evaluate on test set
    print("\n[Step 5/5] Evaluating on test set...")
    class_names = ['neutral', 'left', 'right', 'propaganda']

    results = model.evaluate(
        X_test,
        test_labels,
        class_names=class_names
    )

    # Save trained model
    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
    model.save(MODEL_SAVE_PATH)

    # Print feature importance (for logistic regression)
    if model.model_type == 'logistic':
        print("\n" + "="*70)
        print("Top Features per Class")
        print("="*70)

        # Get feature names
        feature_names = tfidf_extractor.vectorizer.get_feature_names_out()
        importance = model.get_feature_importance(feature_names, top_n=10)

        for class_idx, features in importance.items():
            print(f"\nClass: {class_names[class_idx]}")
            print("-" * 40)
            for feat, coef in features[:10]:
                print(f"  {feat:30s} {coef:+.4f}")

    print("\n" + "="*70)
    print("Training Complete!")
    print("="*70)
    print(f"Model saved to: {MODEL_SAVE_PATH}")
    print(f"Vectorizer saved to: {TFIDF_SAVE_PATH}")
    print(f"Test Accuracy: {results['accuracy']:.4f}")
    print("\nYou can now use this model for predictions with:")
    print("  python src/inference/predict_news.py --model baseline --text 'your text'")


if __name__ == "__main__":
    main()
