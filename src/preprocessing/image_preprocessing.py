"""
Image Preprocessing for CNN Classification
Handles loading, resizing, normalization, and augmentation.
"""

from typing import Tuple, Optional
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator


class ImagePreprocessor:
    """
    Handles image loading and preprocessing.

    Why preprocess images:
    - Resize to consistent dimensions (neural networks need fixed input size)
    - Normalize pixel values (helps neural networks learn faster)
    - Data augmentation (creates variations to prevent overfitting)
    """

    def __init__(
        self,
        target_size: Tuple[int, int] = (128, 128),
        normalization: str = 'standard'
    ):
        """
        Initialize image preprocessor.

        Args:
            target_size: (height, width) to resize images
                        Why: CNNs need consistent input dimensions
                        Larger = more detail but slower training
            normalization: How to normalize pixel values
                          'standard': [0, 1] range
                          'imagenet': Mean subtraction used by ImageNet models
                          'centered': [-1, 1] range
        """
        self.target_size = target_size
        self.normalization = normalization

    def load_image(self, image_path: str) -> np.ndarray:
        """
        Load and preprocess a single image.

        Args:
            image_path: Path to image file

        Returns:
            Preprocessed image as numpy array
        """
        # Load image using PIL
        # Why PIL: Handles many formats, reliable resizing
        img = Image.open(image_path)

        # Convert to RGB if needed
        # Why: Some images are grayscale or RGBA, we need consistent format
        if img.mode != 'RGB':
            img = img.convert('RGB')

        # Resize to target size
        # Why: All images must have same dimensions for batching
        img = img.resize(self.target_size, Image.BILINEAR)

        # Convert to numpy array
        img_array = np.array(img)

        # Normalize pixel values
        img_array = self.normalize(img_array)

        return img_array

    def normalize(self, image: np.ndarray) -> np.ndarray:
        """
        Normalize pixel values.

        Why normalize:
        - Raw pixels are 0-255, which are large numbers for neural networks
        - Normalized values help with gradient descent convergence
        - Makes training faster and more stable

        Args:
            image: Image array with values in [0, 255]

        Returns:
            Normalized image array
        """
        image = image.astype(np.float32)

        if self.normalization == 'standard':
            # Scale to [0, 1]
            image = image / 255.0

        elif self.normalization == 'imagenet':
            # ImageNet mean subtraction
            # These are the mean RGB values across ImageNet
            mean = np.array([123.675, 116.28, 103.53])
            std = np.array([58.395, 57.12, 57.375])
            image = (image - mean) / std

        elif self.normalization == 'centered':
            # Scale to [-1, 1]
            image = (image / 127.5) - 1.0

        return image

    def denormalize(self, image: np.ndarray) -> np.ndarray:
        """
        Reverse normalization for visualization.

        Args:
            image: Normalized image array

        Returns:
            Image array in [0, 255] range
        """
        if self.normalization == 'standard':
            image = image * 255.0

        elif self.normalization == 'imagenet':
            mean = np.array([123.675, 116.28, 103.53])
            std = np.array([58.395, 57.12, 57.375])
            image = (image * std) + mean

        elif self.normalization == 'centered':
            image = (image + 1.0) * 127.5

        return np.clip(image, 0, 255).astype(np.uint8)

    def load_images_batch(self, image_paths: list) -> np.ndarray:
        """
        Load multiple images at once.

        Args:
            image_paths: List of image file paths

        Returns:
            Batch of images as 4D array (batch_size, height, width, channels)
        """
        images = [self.load_image(path) for path in image_paths]
        return np.array(images)


class ImageAugmentor:
    """
    Create augmented versions of images for training.

    Data Augmentation: Creating variations of training images
    Why augment:
    - Increases effective dataset size
    - Prevents overfitting (model sees different versions each epoch)
    - Makes model robust to variations (rotation, flip, zoom, etc.)
    - Especially important when you have limited training data
    """

    def __init__(
        self,
        rotation_range: int = 20,
        width_shift_range: float = 0.2,
        height_shift_range: float = 0.2,
        horizontal_flip: bool = True,
        zoom_range: float = 0.2,
        fill_mode: str = 'nearest'
    ):
        """
        Initialize image augmentor.

        Args:
            rotation_range: Degrees of random rotation
            width_shift_range: Fraction of width to shift horizontally
            height_shift_range: Fraction of height to shift vertically
            horizontal_flip: Whether to randomly flip images horizontally
            zoom_range: Range for random zoom
            fill_mode: How to fill new pixels after transformations
        """
        self.augmentor = ImageDataGenerator(
            rotation_range=rotation_range,
            width_shift_range=width_shift_range,
            height_shift_range=height_shift_range,
            horizontal_flip=horizontal_flip,
            zoom_range=zoom_range,
            fill_mode=fill_mode
        )

    def flow_from_arrays(
        self,
        images: np.ndarray,
        labels: np.ndarray,
        batch_size: int = 32
    ):
        """
        Create an augmented data generator from arrays.

        Why use generator:
        - Memory efficient (doesn't load all images at once)
        - Creates augmentations on-the-fly during training
        - Each epoch sees different augmented versions

        Args:
            images: Array of images
            labels: Array of labels
            batch_size: Number of images per batch

        Returns:
            Generator yielding (images, labels) batches
        """
        return self.augmentor.flow(
            images,
            labels,
            batch_size=batch_size,
            shuffle=True
        )


def create_tf_dataset(
    image_paths: np.ndarray,
    labels: np.ndarray,
    preprocessor: ImagePreprocessor,
    batch_size: int = 32,
    shuffle: bool = True,
    augment: bool = False
) -> tf.data.Dataset:
    """
    Create a TensorFlow dataset for efficient training.

    Why use tf.data:
    - Optimized for performance (prefetching, parallel processing)
    - GPU-friendly data pipeline
    - Memory efficient

    Args:
        image_paths: Array of image file paths
        labels: Array of labels
        preprocessor: ImagePreprocessor instance
        batch_size: Number of images per batch
        shuffle: Whether to shuffle data
        augment: Whether to apply augmentation

    Returns:
        TensorFlow Dataset
    """
    def load_and_preprocess(path, label):
        """Load and preprocess single image."""
        # Read file
        image = tf.io.read_file(path)
        # Decode image
        image = tf.image.decode_jpeg(image, channels=3)
        # Resize
        image = tf.image.resize(image, preprocessor.target_size)
        # Normalize
        if preprocessor.normalization == 'standard':
            image = image / 255.0
        elif preprocessor.normalization == 'centered':
            image = (image / 127.5) - 1.0

        return image, label

    # Create dataset from file paths and labels
    dataset = tf.data.Dataset.from_tensor_slices((image_paths, labels))

    if shuffle:
        # Shuffle with buffer
        # Why: Prevents model from learning order patterns
        dataset = dataset.shuffle(buffer_size=len(image_paths))

    # Load and preprocess images
    dataset = dataset.map(
        load_and_preprocess,
        num_parallel_calls=tf.data.AUTOTUNE  # Parallel processing for speed
    )

    # Batch the data
    dataset = dataset.batch(batch_size)

    # Prefetch batches for performance
    # Why: While GPU trains on one batch, CPU prepares the next
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset


def main():
    """Demo image preprocessing."""
    print("Image Preprocessing Demo")
    print("="*50)

    # Create preprocessor
    preprocessor = ImagePreprocessor(
        target_size=(128, 128),
        normalization='standard'
    )

    print(f"Target size: {preprocessor.target_size}")
    print(f"Normalization: {preprocessor.normalization}")

    # Note: This would work with actual images
    # from PIL import Image
    # sample_image = Image.new('RGB', (256, 256), color='red')
    # img_array = preprocessor.load_image('sample.jpg')
    # print(f"Loaded image shape: {img_array.shape}")
    # print(f"Value range: [{img_array.min():.2f}, {img_array.max():.2f}]")

    print("\nAugmentation settings:")
    augmentor = ImageAugmentor(
        rotation_range=20,
        horizontal_flip=True,
        zoom_range=0.2
    )
    print("- Rotation: ±20 degrees")
    print("- Horizontal flip: Yes")
    print("- Zoom: ±20%")
    print("\nAugmentation increases dataset diversity and prevents overfitting!")


if __name__ == "__main__":
    main()
