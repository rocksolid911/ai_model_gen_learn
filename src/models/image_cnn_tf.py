"""
Image CNN Model using TensorFlow/Keras
Convolutional Neural Network built from scratch for image classification.
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from typing import Optional, Tuple


class ImageCNN:
    """
    Convolutional Neural Network for image classification.

    Built from scratch (no pre-trained weights).

    CNN Architecture:
    - Convolutional layers: Extract visual features (edges, patterns, objects)
    - MaxPooling layers: Reduce spatial dimensions, increase receptive field
    - Dense layers: Combine features for classification
    - Dropout: Prevent overfitting

    Why CNNs for images:
    - Spatial hierarchy: Learn from pixels -> edges -> patterns -> objects
    - Parameter sharing: Same filter used across entire image
    - Translation invariant: Recognizes features regardless of position
    """

    def __init__(
        self,
        input_shape: Tuple[int, int, int] = (128, 128, 3),
        num_classes: int = 5,
        architecture: str = 'standard',
        random_state: int = 42
    ):
        """
        Initialize CNN model.

        Args:
            input_shape: (height, width, channels)
            num_classes: Number of output classes
            architecture: 'simple', 'standard', or 'deep'
            random_state: Random seed
        """
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.architecture = architecture
        self.random_state = random_state
        self.model: Optional[keras.Model] = None

        # Set random seeds
        tf.random.set_seed(random_state)
        np.random.seed(random_state)

        print(f"Initialized {architecture} CNN")
        print(f"Input shape: {input_shape}, Classes: {num_classes}")

    def build_model(self):
        """
        Build CNN architecture from scratch.

        Returns:
            Compiled Keras model
        """
        print(f"\nBuilding {self.architecture} CNN architecture...")

        if self.architecture == 'simple':
            # Simple CNN: Good for quick prototyping
            model = self._build_simple_cnn()

        elif self.architecture == 'standard':
            # Standard CNN: Balanced performance and speed
            model = self._build_standard_cnn()

        elif self.architecture == 'deep':
            # Deep CNN: More capacity, needs more data
            model = self._build_deep_cnn()

        else:
            raise ValueError(f"Unknown architecture: {self.architecture}")

        # Compile model
        # Loss: sparse_categorical_crossentropy for integer labels
        # Optimizer: Adam with default learning rate
        # Metrics: Track accuracy
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

        model.summary()
        self.model = model
        return model

    def _build_simple_cnn(self) -> keras.Model:
        """
        Build a simple CNN with 2 conv blocks.

        Fast to train, good for small datasets or quick experiments.
        """
        model = keras.Sequential([
            # Input layer
            layers.Input(shape=self.input_shape),

            # Conv Block 1
            # Why 32 filters: Start with small number, increase in later layers
            layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),  # Normalize activations for stable training
            layers.MaxPooling2D((2, 2)),  # Reduce spatial size by 2

            # Conv Block 2
            layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),

            # Flatten and classify
            layers.Flatten(),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.5),  # Dropout to prevent overfitting
            layers.Dense(self.num_classes, activation='softmax')
        ], name='simple_cnn')

        return model

    def _build_standard_cnn(self) -> keras.Model:
        """
        Build a standard CNN with 3-4 conv blocks.

        Good balance of performance and training time.
        """
        model = keras.Sequential([
            # Input layer
            layers.Input(shape=self.input_shape),

            # Conv Block 1: Extract low-level features (edges, colors)
            layers.Conv2D(32, (3, 3), activation='relu', padding='same', name='conv1_1'),
            layers.Conv2D(32, (3, 3), activation='relu', padding='same', name='conv1_2'),
            layers.BatchNormalization(name='bn1'),
            layers.MaxPooling2D((2, 2), name='pool1'),
            layers.Dropout(0.25, name='dropout1'),

            # Conv Block 2: Extract mid-level features (textures, patterns)
            layers.Conv2D(64, (3, 3), activation='relu', padding='same', name='conv2_1'),
            layers.Conv2D(64, (3, 3), activation='relu', padding='same', name='conv2_2'),
            layers.BatchNormalization(name='bn2'),
            layers.MaxPooling2D((2, 2), name='pool2'),
            layers.Dropout(0.25, name='dropout2'),

            # Conv Block 3: Extract high-level features (parts, objects)
            layers.Conv2D(128, (3, 3), activation='relu', padding='same', name='conv3_1'),
            layers.Conv2D(128, (3, 3), activation='relu', padding='same', name='conv3_2'),
            layers.BatchNormalization(name='bn3'),
            layers.MaxPooling2D((2, 2), name='pool3'),
            layers.Dropout(0.25, name='dropout3'),

            # Global Average Pooling: Alternative to Flatten
            # Why: Reduces parameters, acts as regularization
            layers.GlobalAveragePooling2D(name='gap'),

            # Dense layers for classification
            layers.Dense(256, activation='relu', name='dense1'),
            layers.BatchNormalization(name='bn_dense'),
            layers.Dropout(0.5, name='dropout_dense'),

            layers.Dense(128, activation='relu', name='dense2'),
            layers.Dropout(0.5, name='dropout_dense2'),

            # Output layer
            layers.Dense(self.num_classes, activation='softmax', name='output')
        ], name='standard_cnn')

        return model

    def _build_deep_cnn(self) -> keras.Model:
        """
        Build a deeper CNN with 5 conv blocks.

        More capacity but needs more data and training time.
        Inspired by VGG architecture.
        """
        model = keras.Sequential([
            # Input
            layers.Input(shape=self.input_shape),

            # Block 1
            layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.2),

            # Block 2
            layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
            layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.2),

            # Block 3
            layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
            layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.3),

            # Block 4
            layers.Conv2D(512, (3, 3), activation='relu', padding='same'),
            layers.Conv2D(512, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.3),

            # Global pooling and classification
            layers.GlobalAveragePooling2D(),
            layers.Dense(512, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(256, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(self.num_classes, activation='softmax')
        ], name='deep_cnn')

        return model

    def train(
        self,
        train_dataset: tf.data.Dataset,
        val_dataset: Optional[tf.data.Dataset] = None,
        epochs: int = 30,
        verbose: int = 1
    ):
        """
        Train the CNN model.

        Args:
            train_dataset: Training dataset (tf.data.Dataset)
            val_dataset: Validation dataset (optional)
            epochs: Number of training epochs
            verbose: Verbosity level

        Returns:
            Training history
        """
        if self.model is None:
            self.build_model()

        print(f"\nTraining CNN for {epochs} epochs...")

        # Callbacks for better training
        callbacks = [
            # Early stopping
            keras.callbacks.EarlyStopping(
                monitor='val_loss' if val_dataset else 'loss',
                patience=5,
                restore_best_weights=True,
                verbose=1
            ),
            # Reduce learning rate on plateau
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss' if val_dataset else 'loss',
                factor=0.5,
                patience=3,
                verbose=1,
                min_lr=1e-7
            ),
            # Model checkpointing (optional)
            # keras.callbacks.ModelCheckpoint(
            #     'best_model.h5',
            #     monitor='val_accuracy',
            #     save_best_only=True,
            #     verbose=1
            # )
        ]

        # Train the model
        history = self.model.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=epochs,
            callbacks=callbacks,
            verbose=verbose
        )

        return history

    def train_with_arrays(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        epochs: int = 30,
        batch_size: int = 32
    ):
        """
        Train with numpy arrays (alternative to tf.data.Dataset).

        Args:
            X_train: Training images
            y_train: Training labels
            X_val: Validation images
            y_val: Validation labels
            epochs: Number of epochs
            batch_size: Batch size
        """
        if self.model is None:
            self.build_model()

        validation_data = None
        if X_val is not None and y_val is not None:
            validation_data = (X_val, y_val)

        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_loss' if validation_data else 'loss',
                patience=5,
                restore_best_weights=True
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss' if validation_data else 'loss',
                factor=0.5,
                patience=3,
                min_lr=1e-7
            )
        ]

        history = self.model.fit(
            X_train,
            y_train,
            validation_data=validation_data,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks
        )

        return history

    def predict(self, images: np.ndarray) -> np.ndarray:
        """
        Predict class labels.

        Args:
            images: Image array

        Returns:
            Predicted class indices
        """
        predictions = self.model.predict(images, verbose=0)
        return np.argmax(predictions, axis=1)

    def predict_proba(self, images: np.ndarray) -> np.ndarray:
        """
        Get prediction probabilities.

        Args:
            images: Image array

        Returns:
            Probability distribution over classes
        """
        return self.model.predict(images, verbose=0)

    def save(self, path: str):
        """
        Save trained model.

        Args:
            path: Path to save model
        """
        self.model.save(path)
        print(f"Model saved to {path}")

    def load(self, path: str):
        """
        Load trained model.

        Args:
            path: Path to model file
        """
        self.model = keras.models.load_model(path)
        print(f"Model loaded from {path}")


def main():
    """Demo CNN model."""
    print("Image CNN Model Demo")
    print("="*50)

    # Initialize model
    cnn = ImageCNN(
        input_shape=(128, 128, 3),
        num_classes=5,
        architecture='standard'
    )

    # Build model
    cnn.build_model()

    # Test with dummy data
    dummy_images = np.random.rand(10, 128, 128, 3).astype(np.float32)
    predictions = cnn.predict(dummy_images)

    print(f"\nSample predictions: {predictions}")


if __name__ == "__main__":
    main()
