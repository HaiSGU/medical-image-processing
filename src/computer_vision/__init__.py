"""
Computer Vision Module for Medical Image Processing

This module provides advanced computer vision capabilities including:
- Classification: Brain tumor classification using deep learning
- Detection: Lesion and tumor detection
- Feature Extraction: Texture, shape, and intensity features

Author: HaiSGU
Date: 2025-11-23
"""

from .feature_extraction import FeatureExtractor
from .detection import LesionDetector

# Classification requires TensorFlow (optional)
try:
    from .classification import BrainTumorClassifier

    __all__ = ["FeatureExtractor", "LesionDetector", "BrainTumorClassifier"]
except ImportError:
    __all__ = ["FeatureExtractor", "LesionDetector"]

__version__ = "1.0.0"
