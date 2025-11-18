"""
News Deep Learning Model using TensorFlow/Keras
Neural network approach with embeddings and convolutional/recurrent layers.
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
from typing import Tuple, Optional


class NewsDeepModel:
    """
    Deep learning classifier for news articles.

    Architecture options:
    1. Embedding + Conv1D + GlobalMaxPooling (faster, good for short texts)
    2. Embedding + LSTM/GRU (slower, captures sequential patterns)

    Why deep learning:
    - Learns feature representations automatically
    - Can capture complex patterns and word relationships
    - Often better performance with enough data
    - Can use pre-trained embeddings (word2vec, GloVe)
    """

    def __init__(
        self,
        vocab_size: int = 10000,
        embedding_dim: int = 128,
        max_length: int = 200,
        num_classes: int = 4,
        architecture: str = 'cnn',
        random_state: int = 42
    ):
        """
        Initialize deep learning model.

        Args:
            vocab_size: Maximum number of words to keep
            embedding_dim: Dimension of word embeddings
                          Why: Converts words to dense vectors
                          Larger = more expressive but slower
            max_length: Maximum sequence length (truncate/pad to this)
            num_classes: Number of output classes
            architecture: 'cnn' or 'lstm'
            random_state: Random seed
        """
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.max_length = max_length
        self.num_classes = num_classes
        self.architecture = architecture
        self.random_state = random_state

        # Tokenizer converts text to sequences of integers
        self.tokenizer: Optional[Tokenizer] = None
        self.model: Optional[keras.Model] = None

        # Set random seeds for reproducibility
        tf.random.set_seed(random_state)
        np.random.seed(random_state)

        print(f"Initialized {architecture} model")
        print(f"Vocab size: {vocab_size}, Embedding dim: {embedding_dim}")

    def build_model(self):
        """
        Build the neural network architecture.

        Returns:
            Compiled Keras model
        """
        print(f"\nBuilding {self.architecture} model...")

        # Input layer: sequences of word indices
        input_layer = layers.Input(shape=(self.max_length,), name='text_input')

        # Embedding layer: word index -> dense vector
        # Why: Neural networks need continuous inputs, not discrete word IDs
        # Trainable: Yes, will learn optimal embeddings for this task
        embedding = layers.Embedding(
            input_dim=self.vocab_size,
            output_dim=self.embedding_dim,
            input_length=self.max_length,
            name='embedding'
        )(input_layer)

        if self.architecture == 'cnn':
            # CNN Architecture
            # Why Conv1D: Captures local patterns (n-grams) automatically
            # Multiple filter sizes capture different n-gram lengths

            # First convolutional block
            conv1 = layers.Conv1D(
                filters=128,
                kernel_size=5,  # Look at 5 words at a time
                activation='relu',
                name='conv1'
            )(embedding)
            pool1 = layers.GlobalMaxPooling1D(name='pool1')(conv1)
            # Why GlobalMaxPooling: Takes the most important feature
            # from each filter across the entire sequence

            # Second convolutional block (different kernel size)
            conv2 = layers.Conv1D(
                filters=128,
                kernel_size=3,  # Look at 3 words at a time
                activation='relu',
                name='conv2'
            )(embedding)
            pool2 = layers.GlobalMaxPooling1D(name='pool2')(conv2)

            # Concatenate features from different kernel sizes
            concatenated = layers.Concatenate(name='concat')([pool1, pool2])

            # Dropout for regularization
            # Why: Randomly drops connections to prevent overfitting
            x = layers.Dropout(0.5, name='dropout')(concatenated)

        elif self.architecture == 'lstm':
            # LSTM Architecture
            # Why LSTM: Captures long-range dependencies in text
            # Bidirectional: Reads text forward and backward

            # Bidirectional LSTM layer
            # Why bidirectional: Context from both directions helps
            x = layers.Bidirectional(
                layers.LSTM(
                    64,
                    return_sequences=False,  # Only final state
                    dropout=0.2,  # Recurrent dropout
                    recurrent_dropout=0.2
                ),
                name='bi_lstm'
            )(embedding)

            # Additional dropout
            x = layers.Dropout(0.5, name='dropout')(x)

        else:
            raise ValueError(f"Unknown architecture: {self.architecture}")

        # Dense layer for learning high-level features
        x = layers.Dense(64, activation='relu', name='dense1')(x)
        x = layers.Dropout(0.3, name='dropout2')(x)

        # Output layer: softmax for multi-class classification
        # Why softmax: Converts logits to probability distribution
        # Sum of probabilities = 1.0
        output = layers.Dense(
            self.num_classes,
            activation='softmax',
            name='output'
        )(x)

        # Create model
        model = keras.Model(inputs=input_layer, outputs=output, name='news_classifier')

        # Compile model
        # Loss: categorical_crossentropy for multi-class classification
        # Optimizer: Adam (adaptive learning rate, usually works well)
        # Metrics: Track accuracy during training
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

        # Print model summary
        model.summary()

        self.model = model
        return model

    def prepare_tokenizer(self, texts: list):
        """
        Fit tokenizer on training texts.

        Tokenizer learns vocabulary from training data.

        Args:
            texts: List of text strings
        """
        print("\nFitting tokenizer...")

        self.tokenizer = Tokenizer(
            num_words=self.vocab_size,
            oov_token='<OOV>',  # Out-of-vocabulary token
            lower=True
        )

        # Fit on texts to build vocabulary
        # Most frequent words get lower indices
        self.tokenizer.fit_on_texts(texts)

        print(f"Vocabulary size: {len(self.tokenizer.word_index)}")
        print(f"Using top {self.vocab_size} words")

    def texts_to_sequences(self, texts: list) -> np.ndarray:
        """
        Convert texts to padded sequences.

        Args:
            texts: List of text strings

        Returns:
            Padded sequences as numpy array
        """
        # Convert texts to sequences of integers
        sequences = self.tokenizer.texts_to_sequences(texts)

        # Pad sequences to same length
        # Why: Neural networks need fixed-size inputs for batching
        # Truncate long sequences, pad short ones
        padded = pad_sequences(
            sequences,
            maxlen=self.max_length,
            padding='post',  # Pad at the end
            truncating='post'  # Truncate at the end
        )

        return padded

    def train(
        self,
        train_texts: list,
        train_labels: np.ndarray,
        val_texts: Optional[list] = None,
        val_labels: Optional[np.ndarray] = None,
        epochs: int = 10,
        batch_size: int = 32
    ):
        """
        Train the deep learning model.

        Args:
            train_texts: Training text strings
            train_labels: Training labels
            val_texts: Validation texts (optional)
            val_labels: Validation labels (optional)
            epochs: Number of training epochs
            batch_size: Batch size for training
        """
        # Prepare tokenizer if not already done
        if self.tokenizer is None:
            self.prepare_tokenizer(train_texts)

        # Convert texts to sequences
        X_train = self.texts_to_sequences(train_texts)
        y_train = train_labels

        validation_data = None
        if val_texts is not None and val_labels is not None:
            X_val = self.texts_to_sequences(val_texts)
            y_val = val_labels
            validation_data = (X_val, y_val)

        # Build model if not already done
        if self.model is None:
            self.build_model()

        print(f"\nTraining model...")
        print(f"Training samples: {len(X_train)}")
        print(f"Batch size: {batch_size}, Epochs: {epochs}")

        # Callbacks for better training
        callbacks = [
            # Early stopping: Stop if validation loss doesn't improve
            # Why: Prevents overfitting and saves time
            keras.callbacks.EarlyStopping(
                monitor='val_loss' if validation_data else 'loss',
                patience=3,  # Stop after 3 epochs without improvement
                restore_best_weights=True,
                verbose=1
            ),
            # Reduce learning rate when stuck
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss' if validation_data else 'loss',
                factor=0.5,  # Reduce by half
                patience=2,
                verbose=1,
                min_lr=1e-6
            )
        ]

        # Train the model
        history = self.model.fit(
            X_train,
            y_train,
            validation_data=validation_data,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )

        return history

    def predict(self, texts: list) -> np.ndarray:
        """
        Predict class labels for texts.

        Args:
            texts: List of text strings

        Returns:
            Predicted class indices
        """
        X = self.texts_to_sequences(texts)
        predictions = self.model.predict(X, verbose=0)
        return np.argmax(predictions, axis=1)

    def predict_proba(self, texts: list) -> np.ndarray:
        """
        Get prediction probabilities.

        Args:
            texts: List of text strings

        Returns:
            Probability distribution over classes
        """
        X = self.texts_to_sequences(texts)
        return self.model.predict(X, verbose=0)

    def save(self, model_path: str, tokenizer_path: str):
        """
        Save model and tokenizer.

        Args:
            model_path: Path to save Keras model
            tokenizer_path: Path to save tokenizer
        """
        # Save Keras model
        self.model.save(model_path)
        print(f"Model saved to {model_path}")

        # Save tokenizer
        with open(tokenizer_path, 'wb') as f:
            pickle.dump(self.tokenizer, f)
        print(f"Tokenizer saved to {tokenizer_path}")

    def load(self, model_path: str, tokenizer_path: str):
        """
        Load model and tokenizer.

        Args:
            model_path: Path to Keras model
            tokenizer_path: Path to tokenizer
        """
        # Load Keras model
        self.model = keras.models.load_model(model_path)
        print(f"Model loaded from {model_path}")

        # Load tokenizer
        with open(tokenizer_path, 'rb') as f:
            self.tokenizer = pickle.load(f)
        print(f"Tokenizer loaded from {tokenizer_path}")


def main():
    """Demo deep learning model."""
    print("News Deep Learning Model Demo")
    print("="*50)

    # Sample texts
    texts = [
        "breaking news about politics and government",
        "sports team wins championship game",
        "technology company releases new product",
        "propaganda fake news biased reporting"
    ] * 25  # Repeat to have more samples

    labels = np.array([0, 1, 2, 3] * 25)

    # Initialize model
    model = NewsDeepModel(
        vocab_size=1000,
        embedding_dim=64,
        max_length=50,
        num_classes=4,
        architecture='cnn'
    )

    # Build model
    model.build_model()

    # Prepare tokenizer
    model.prepare_tokenizer(texts)

    # Convert to sequences
    sequences = model.texts_to_sequences(texts[:5])
    print(f"\nSample sequences:\n{sequences}")


if __name__ == "__main__":
    main()
