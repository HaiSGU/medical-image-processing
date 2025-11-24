"""
Brain Tumor Classification Module

Classifies brain MRI scans using deep learning:
- Pre-trained CNN models (ResNet50, VGG16, EfficientNet)
- Transfer learning for medical images
- Multi-class classification (Glioma, Meningioma, Pituitary, Normal)
- Grad-CAM visualization

Author: HaiSGU
Date: 2025-11-23
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
import warnings

# TensorFlow/Keras imports
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras.applications import ResNet50, VGG16, EfficientNetB0
    from tensorflow.keras.preprocessing import image as keras_image
    from tensorflow.keras import layers, models

    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning("TensorFlow not available. Classification features will be limited.")

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suppress TF warnings
warnings.filterwarnings("ignore")
if TENSORFLOW_AVAILABLE:
    tf.get_logger().setLevel("ERROR")


class BrainTumorClassifier:
    """
    Classify brain tumors using deep learning.

    Supports multi-class classification:
    - Glioma: Aggressive brain tumor
    - Meningioma: Usually benign tumor
    - Pituitary: Pituitary gland tumor
    - Normal: No tumor detected

    Examples:
        >>> from src.computer_vision.classification import BrainTumorClassifier
        >>> import numpy as np
        >>>
        >>> # Create classifier
        >>> classifier = BrainTumorClassifier(model_name='resnet50')
        >>>
        >>> # Load and classify image
        >>> image = np.random.rand(224, 224, 3)
        >>> prediction = classifier.predict(image)
        >>>
        >>> print(f"Predicted class: {prediction['class']}")
        >>> print(f"Confidence: {prediction['confidence']:.2%}")
        >>> print(f"All probabilities: {prediction['probabilities']}")
    """

    # Class names
    CLASSES = ["glioma", "meningioma", "normal", "pituitary"]

    # Model input sizes
    MODEL_SIZES = {
        "resnet50": (224, 224),
        "vgg16": (224, 224),
        "efficientnet": (224, 224),
    }

    def __init__(
        self,
        model_name: str = "resnet50",
        num_classes: int = 4,
        weights: str = "imagenet",
    ):
        """
        Initialize brain tumor classifier.

        Args:
            model_name: 'resnet50', 'vgg16', or 'efficientnet'
            num_classes: Number of output classes
            weights: 'imagenet' for pre-trained weights, or None

        Raises:
            ImportError: If TensorFlow is not installed
            ValueError: If model_name is not recognized
        """
        if not TENSORFLOW_AVAILABLE:
            raise ImportError(
                "TensorFlow is required for classification. "
                "Install it with: pip install tensorflow>=2.13.0"
            )

        self.model_name = model_name.lower()
        self.num_classes = num_classes
        self.input_size = self.MODEL_SIZES.get(self.model_name, (224, 224))

        # Build model
        self.model = self._build_model(weights)

        logger.info(f"BrainTumorClassifier initialized: {model_name}")
        logger.info(f"Input size: {self.input_size}")
        logger.info(f"Number of classes: {num_classes}")

    def _build_model(self, weights: str) -> keras.Model:
        """
        Build CNN model with transfer learning.

        Args:
            weights: Pre-trained weights ('imagenet' or None)

        Returns:
            Compiled Keras model
        """
        # Load base model
        if self.model_name == "resnet50":
            base_model = ResNet50(
                weights=weights, include_top=False, input_shape=(*self.input_size, 3)
            )
        elif self.model_name == "vgg16":
            base_model = VGG16(
                weights=weights, include_top=False, input_shape=(*self.input_size, 3)
            )
        elif self.model_name == "efficientnet":
            base_model = EfficientNetB0(
                weights=weights, include_top=False, input_shape=(*self.input_size, 3)
            )
        else:
            raise ValueError(
                f"Unknown model: {self.model_name}. "
                f"Choose from: {list(self.MODEL_SIZES.keys())}"
            )

        # Freeze base model layers (for transfer learning)
        base_model.trainable = False

        # Add classification head
        model = models.Sequential(
            [
                base_model,
                layers.GlobalAveragePooling2D(),
                layers.Dense(256, activation="relu"),
                layers.Dropout(0.5),
                layers.Dense(self.num_classes, activation="softmax"),
            ]
        )

        # Compile model
        model.compile(
            optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
        )

        logger.info("Model built successfully")
        return model

    def preprocess_image(self, image: np.ndarray, normalize: bool = True) -> np.ndarray:
        """
        Preprocess image for model input.

        Args:
            image: Input image (2D grayscale or 3D RGB)
            normalize: Apply ImageNet normalization

        Returns:
            Preprocessed image ready for model (1, H, W, 3)
        """
        # Convert grayscale to RGB if needed
        if image.ndim == 2:
            image = np.stack([image] * 3, axis=-1)
        elif image.shape[-1] == 1:
            image = np.concatenate([image] * 3, axis=-1)

        # Resize to model input size
        if image.shape[:2] != self.input_size:
            import cv2

            image = cv2.resize(image, self.input_size)

        # Normalize to 0-255
        if image.max() <= 1.0:
            image = (image * 255).astype(np.uint8)

        # Add batch dimension
        image = np.expand_dims(image, axis=0)

        # Apply model-specific preprocessing
        if normalize:
            if self.model_name == "resnet50":
                from tensorflow.keras.applications.resnet50 import preprocess_input
            elif self.model_name == "vgg16":
                from tensorflow.keras.applications.vgg16 import preprocess_input
            elif self.model_name == "efficientnet":
                from tensorflow.keras.applications.efficientnet import preprocess_input

            image = preprocess_input(image)

        return image

    def predict(self, image: np.ndarray, top_k: int = 3) -> Dict:
        """
        Predict brain tumor class.

        Args:
            image: Input brain MRI scan
            top_k: Return top K predictions

        Returns:
            Dictionary with:
            - class: Predicted class name
            - confidence: Confidence score (0-1)
            - probabilities: Dict of all class probabilities
            - top_k: List of (class, probability) for top K predictions

        Example:
            >>> result = classifier.predict(brain_scan)
            >>> print(f"Prediction: {result['class']} ({result['confidence']:.1%})")
            >>> print("Top 3 predictions:")
            >>> for cls, prob in result['top_k']:
            ...     print(f"  {cls}: {prob:.1%}")
        """
        try:
            # Preprocess image
            processed = self.preprocess_image(image)

            # Get predictions
            predictions = self.model.predict(processed, verbose=0)[0]

            # Get top class
            top_idx = np.argmax(predictions)
            top_class = (
                self.CLASSES[top_idx]
                if top_idx < len(self.CLASSES)
                else f"class_{top_idx}"
            )
            confidence = float(predictions[top_idx])

            # All probabilities
            probabilities = {
                self.CLASSES[i] if i < len(self.CLASSES) else f"class_{i}": float(
                    predictions[i]
                )
                for i in range(len(predictions))
            }

            # Top K predictions
            top_k_indices = np.argsort(predictions)[-top_k:][::-1]
            top_k_preds = [
                (
                    self.CLASSES[i] if i < len(self.CLASSES) else f"class_{i}",
                    float(predictions[i]),
                )
                for i in top_k_indices
            ]

            result = {
                "class": top_class,
                "confidence": confidence,
                "probabilities": probabilities,
                "top_k": top_k_preds,
            }

            logger.info(f"✅ Prediction: {top_class} ({confidence:.2%})")
            return result

        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            return {
                "class": "error",
                "confidence": 0.0,
                "probabilities": {},
                "top_k": [],
                "error": str(e),
            }

    def predict_batch(self, images: List[np.ndarray]) -> List[Dict]:
        """
        Predict multiple images at once.

        Args:
            images: List of brain MRI scans

        Returns:
            List of prediction dictionaries

        Example:
            >>> brain_scans = [image1, image2, image3]
            >>> results = classifier.predict_batch(brain_scans)
            >>> for i, result in enumerate(results):
            ...     print(f"Image {i}: {result['class']} ({result['confidence']:.1%})")
        """
        return [self.predict(img) for img in images]

    def get_gradcam(
        self, image: np.ndarray, layer_name: Optional[str] = None
    ) -> np.ndarray:
        """
        Generate Grad-CAM heatmap to visualize important regions.

        Grad-CAM shows which parts of the image influenced the decision.

        Args:
            image: Input brain scan
            layer_name: Name of layer to visualize (None for last conv layer)

        Returns:
            Heatmap array (0-1)

        Example:
            >>> heatmap = classifier.get_gradcam(brain_scan)
            >>> plt.imshow(brain_scan, cmap='gray')
            >>> plt.imshow(heatmap, alpha=0.5, cmap='jet')
            >>> plt.show()
        """
        try:
            # Preprocess image
            processed = self.preprocess_image(image)

            # Find last convolutional layer if not specified
            if layer_name is None:
                for layer in reversed(self.model.layers[0].layers):
                    if "conv" in layer.name.lower():
                        layer_name = layer.name
                        break

            if layer_name is None:
                logger.warning("No convolutional layer found")
                return np.zeros(image.shape[:2])

            # Create Grad-CAM model
            grad_model = models.Model(
                inputs=[self.model.inputs],
                outputs=[self.model.get_layer(layer_name).output, self.model.output],
            )

            # Compute gradients
            with tf.GradientTape() as tape:
                conv_outputs, predictions = grad_model(processed)
                loss = predictions[:, np.argmax(predictions[0])]

            # Get gradients
            grads = tape.gradient(loss, conv_outputs)

            # Pool gradients
            pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

            # Weight feature maps
            conv_outputs = conv_outputs[0]
            heatmap = tf.reduce_sum(tf.multiply(pooled_grads, conv_outputs), axis=-1)

            # Normalize heatmap
            heatmap = np.maximum(heatmap, 0)
            heatmap /= np.max(heatmap) + 1e-10

            # Resize to original size
            import cv2

            heatmap = cv2.resize(heatmap.numpy(), image.shape[:2][::-1])

            logger.info(f"✅ Generated Grad-CAM using layer: {layer_name}")
            return heatmap

        except Exception as e:
            logger.error(f"Grad-CAM failed: {e}")
            return np.zeros(image.shape[:2])

    def train(
        self,
        train_images: np.ndarray,
        train_labels: np.ndarray,
        validation_data: Optional[Tuple] = None,
        epochs: int = 10,
        batch_size: int = 32,
    ) -> keras.callbacks.History:
        """
        Fine-tune model on medical imaging data.

        Args:
            train_images: Training images (N, H, W, C)
            train_labels: One-hot encoded labels (N, num_classes)
            validation_data: Optional (val_images, val_labels)
            epochs: Number of training epochs
            batch_size: Batch size

        Returns:
            Training history

        Example:
            >>> # Prepare data
            >>> train_images = np.array([...])  # Shape: (100, 224, 224, 3)
            >>> train_labels = keras.utils.to_categorical(labels, 4)
            >>>
            >>> # Train model
            >>> history = classifier.train(
            ...     train_images,
            ...     train_labels,
            ...     epochs=20,
            ...     batch_size=16
            ... )
            >>>
            >>> # Plot training history
            >>> import matplotlib.pyplot as plt
            >>> plt.plot(history.history['accuracy'])
            >>> plt.plot(history.history['val_accuracy'])
            >>> plt.title('Model Accuracy')
            >>> plt.legend(['train', 'val'])
            >>> plt.show()
        """
        logger.info(f"Training model for {epochs} epochs...")

        history = self.model.fit(
            train_images,
            train_labels,
            validation_data=validation_data,
            epochs=epochs,
            batch_size=batch_size,
            verbose=1,
        )

        logger.info("✅ Training complete")
        return history

    def save_model(self, filepath: str):
        """
        Save model to file.

        Args:
            filepath: Path to save model (.h5 or .keras)

        Example:
            >>> classifier.save_model('brain_tumor_classifier.h5')
        """
        self.model.save(filepath)
        logger.info(f"✅ Model saved to: {filepath}")

    def load_model(self, filepath: str):
        """
        Load model from file.

        Args:
            filepath: Path to model file

        Example:
            >>> classifier.load_model('pretrained_model.h5')
        """
        self.model = keras.models.load_model(filepath)
        logger.info(f"✅ Model loaded from: {filepath}")


# Convenience functions
def classify_brain_tumor(image: np.ndarray, model_name: str = "resnet50") -> Dict:
    """
    Quick brain tumor classification.

    Args:
        image: Brain MRI scan
        model_name: Model to use ('resnet50', 'vgg16', 'efficientnet')

    Returns:
        Prediction dictionary

    Example:
        >>> result = classify_brain_tumor(brain_scan)
        >>> print(f"Diagnosis: {result['class']}")
        >>> print(f"Confidence: {result['confidence']:.1%}")
    """
    classifier = BrainTumorClassifier(model_name=model_name)
    return classifier.predict(image)


def get_available_models() -> List[str]:
    """
    Get list of available classification models.

    Returns:
        List of model names
    """
    return list(BrainTumorClassifier.MODEL_SIZES.keys())
