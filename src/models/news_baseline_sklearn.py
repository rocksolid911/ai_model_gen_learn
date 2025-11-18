"""
News Baseline Model using Scikit-learn
Traditional ML approach with TF-IDF features and Logistic Regression.
"""

from typing import Optional
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report
import pickle


class NewsBaselineModel:
    """
    Baseline classifier for news articles using traditional ML.

    Why start with a baseline:
    - Fast to train (seconds vs minutes/hours)
    - Interpretable (can see which features matter)
    - Good starting point to compare deep learning improvements
    - Often surprisingly effective for text classification
    """

    def __init__(
        self,
        model_type: str = 'logistic',
        max_iter: int = 1000,
        random_state: int = 42
    ):
        """
        Initialize baseline model.

        Args:
            model_type: Type of classifier ('logistic' or 'svm')
            max_iter: Maximum iterations for training
            random_state: Random seed for reproducibility
        """
        self.model_type = model_type
        self.random_state = random_state

        if model_type == 'logistic':
            # Logistic Regression
            # Why: Simple, fast, works well with TF-IDF
            # Multi-class: Uses one-vs-rest strategy
            self.model = LogisticRegression(
                max_iter=max_iter,
                random_state=random_state,
                multi_class='multinomial',  # True multi-class classification
                solver='lbfgs',  # Optimization algorithm
                C=1.0,  # Regularization (smaller = more regularization)
                verbose=1
            )
        elif model_type == 'svm':
            # Linear SVM
            # Why: Often better than logistic regression for text
            # Good at finding decision boundaries
            self.model = LinearSVC(
                max_iter=max_iter,
                random_state=random_state,
                C=1.0,
                verbose=1
            )
        else:
            raise ValueError(f"Unknown model_type: {model_type}")

        print(f"Initialized {model_type} baseline model")

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None
    ):
        """
        Train the baseline model.

        Args:
            X_train: Training features (TF-IDF vectors)
            y_train: Training labels
            X_val: Validation features (optional)
            y_val: Validation labels (optional)
        """
        print(f"\nTraining {self.model_type} model...")
        print(f"Training samples: {X_train.shape[0]}")
        print(f"Features: {X_train.shape[1]}")

        # Fit the model
        # This learns the optimal weights for each feature
        self.model.fit(X_train, y_train)

        # Evaluate on training set
        train_preds = self.model.predict(X_train)
        train_acc = accuracy_score(y_train, train_preds)
        print(f"Training accuracy: {train_acc:.4f}")

        # Evaluate on validation set if provided
        if X_val is not None and y_val is not None:
            val_preds = self.model.predict(X_val)
            val_acc = accuracy_score(y_val, val_preds)
            print(f"Validation accuracy: {val_acc:.4f}")

            # Check for overfitting
            # If train accuracy >> val accuracy, model is overfitting
            if train_acc - val_acc > 0.1:
                print("Warning: Possible overfitting (train acc >> val acc)")

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions.

        Args:
            X: Feature vectors

        Returns:
            Predicted class labels
        """
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Get prediction probabilities.

        Args:
            X: Feature vectors

        Returns:
            Probability for each class
        """
        if hasattr(self.model, 'predict_proba'):
            return self.model.predict_proba(X)
        else:
            # SVM doesn't have predict_proba by default
            # Use decision function as proxy
            decision = self.model.decision_function(X)
            # Apply softmax to convert to probabilities
            exp_scores = np.exp(decision - np.max(decision, axis=1, keepdims=True))
            probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
            return probs

    def evaluate(
        self,
        X: np.ndarray,
        y: np.ndarray,
        class_names: list
    ) -> dict:
        """
        Evaluate model performance.

        Args:
            X: Feature vectors
            y: True labels
            class_names: List of class names

        Returns:
            Dictionary with metrics
        """
        predictions = self.predict(X)
        accuracy = accuracy_score(y, predictions)

        print(f"\nEvaluation Results:")
        print(f"Accuracy: {accuracy:.4f}")
        print("\nClassification Report:")
        print(classification_report(
            y,
            predictions,
            target_names=class_names,
            digits=4
        ))

        return {
            'accuracy': accuracy,
            'predictions': predictions
        }

    def save(self, path: str):
        """
        Save trained model to disk.

        Args:
            path: File path to save model
        """
        with open(path, 'wb') as f:
            pickle.dump(self.model, f)
        print(f"Model saved to {path}")

    def load(self, path: str):
        """
        Load trained model from disk.

        Args:
            path: File path to load model from
        """
        with open(path, 'rb') as f:
            self.model = pickle.load(f)
        print(f"Model loaded from {path}")

    def get_feature_importance(self, feature_names: list, top_n: int = 20):
        """
        Get most important features for each class.

        This shows which words are most indicative of each class.

        Args:
            feature_names: List of feature names (words/ngrams)
            top_n: Number of top features to return per class

        Returns:
            Dictionary mapping class to top features
        """
        if self.model_type == 'logistic':
            # Get coefficients for each class
            # Higher absolute value = more important
            coef = self.model.coef_

            importance_dict = {}
            for class_idx, class_coef in enumerate(coef):
                # Get top positive and negative features
                top_indices = np.argsort(np.abs(class_coef))[-top_n:][::-1]
                top_features = [(feature_names[i], class_coef[i]) for i in top_indices]
                importance_dict[class_idx] = top_features

            return importance_dict
        else:
            print("Feature importance only available for logistic regression")
            return {}


def main():
    """Demo baseline model."""
    print("News Baseline Model Demo")
    print("="*50)

    # Create dummy data
    np.random.seed(42)
    X_train = np.random.randn(100, 1000)  # 100 samples, 1000 features
    y_train = np.random.randint(0, 4, 100)  # 4 classes
    X_val = np.random.randn(20, 1000)
    y_val = np.random.randint(0, 4, 20)

    # Initialize and train model
    model = NewsBaselineModel(model_type='logistic')
    model.train(X_train, y_train, X_val, y_val)

    # Make predictions
    predictions = model.predict(X_val)
    print(f"\nSample predictions: {predictions[:5]}")

    # Get probabilities
    probs = model.predict_proba(X_val)
    print(f"Sample probabilities:\n{probs[0]}")


if __name__ == "__main__":
    main()
