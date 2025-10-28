"""
Medical Image Segmentation Package

Implements segmentation algorithms for medical images:
- Brain segmentation (threshold, region growing, morphology)
- Evaluation metrics (Dice, IoU, Hausdorff)
"""

from .brain_segmentation import BrainSegmentation

__all__ = ["BrainSegmentation"]
