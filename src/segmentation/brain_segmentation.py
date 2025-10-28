"""
Brain Segmentation Module

Implements various brain segmentation algorithms:
- Threshold-based segmentation (Otsu, manual)
- Region growing
- Morphological operations
- Connected component analysis
- Evaluation metrics (Dice, IoU, Hausdorff)

GIẢI THÍCH:
-----------
Segmentation = Chia ảnh thành các vùng có ý nghĩa (ROI - Region of Interest)

Ví dụ brain MRI:
- Gray matter (chất xám)
- White matter (chất trắng)
- CSF (cerebrospinal fluid - dịch não tủy)
- Ventricles (tâm thất)

Ứng dụng:
- Đo thể tích não (brain volume)
- Phát hiện khối u (tumor detection)
- Theo dõi bệnh (disease progression)

Author: HaiSGU
Date: 2025-10-28
"""

import numpy as np
import logging
from typing import Tuple, Optional, List, Union
import matplotlib.pyplot as plt
from scipy import ndimage
from skimage import filters, morphology, measure
from skimage.segmentation import random_walker

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BrainSegmentation:
    """
    Brain segmentation using various algorithms.

    Phương pháp:
    1. Threshold-based: Otsu, manual threshold
    2. Region growing: Seed-based
    3. Morphological: Erosion, dilation, opening, closing
    4. Connected components: Label regions

    Attributes:
        image: 2D/3D numpy array (brain MRI)

    Examples:
        >>> # Load brain MRI
        >>> image = np.load('brain.npy')
        >>>
        >>> # Create segmentator
        >>> seg = BrainSegmentation(image)
        >>>
        >>> # Otsu thresholding
        >>> mask = seg.threshold_otsu()
        >>>
        >>> # Region growing
        >>> mask = seg.region_growing(seed=(128, 128))
        >>>
        >>> # Evaluate
        >>> dice = seg.dice_coefficient(ground_truth, prediction)
    """

    def __init__(self, image: np.ndarray):
        """
        Initialize Brain Segmentation.

        Args:
            image: 2D or 3D numpy array (brain MRI)
                  - 2D: (height, width)
                  - 3D: (slices, height, width)

        Example:
            >>> image = np.load('brain_mri.npy')
            >>> seg = BrainSegmentation(image)
        """
        self.image = image.astype(np.float32)
        self.ndim = image.ndim

        logger.info(f"BrainSegmentation initialized:")
        logger.info(f"  Image shape: {image.shape}")
        logger.info(f"  Image range: [{image.min():.2f}, {image.max():.2f}]")
        logger.info(f"  Dimensions: {self.ndim}D")

    def threshold_manual(self, threshold: float, mode: str = "greater") -> np.ndarray:
        """
        Manual thresholding segmentation.

        GIẢI THÍCH:
        -----------
        Phân loại dựa trên intensity value:

        mode='greater':
            if intensity > threshold: label = 1 (foreground)
            else: label = 0 (background)

        mode='less':
            if intensity < threshold: label = 1

        mode='range':
            if lower < intensity < upper: label = 1

        Args:
            threshold: Threshold value (hoặc tuple (lower, upper) cho range)
            mode: 'greater', 'less', hoặc 'range'

        Returns:
            Binary mask (0 and 1)

        Example:
            >>> mask = seg.threshold_manual(100, mode='greater')
        """
        logger.info(f"Manual thresholding: threshold={threshold}, mode={mode}")

        if mode == "greater":
            mask = self.image > threshold
        elif mode == "less":
            mask = self.image < threshold
        elif mode == "range":
            if not isinstance(threshold, (list, tuple)):
                raise ValueError("For 'range' mode, provide (lower, upper)")
            lower, upper = threshold
            mask = (self.image > lower) & (self.image < upper)
        else:
            raise ValueError(f"Unknown mode: {mode}")

        mask = mask.astype(np.uint8)

        logger.info(
            f"  Foreground pixels: {mask.sum()} "
            f"({mask.sum() / mask.size * 100:.1f}%)"
        )

        return mask

    def threshold_otsu(self) -> np.ndarray:
        """
        Otsu's automatic thresholding.

        GIẢI THÍCH:
        -----------
        Otsu's method tự động tìm threshold tốt nhất bằng cách:
        - Minimize within-class variance (phương sai trong class)
        - Maximize between-class variance (phương sai giữa các class)

        Histogram:
           │  Background  │  Foreground
           │    ▁▂▃▄▃▂▁   │  ▁▂▃▄▃▂▁
           └──────────────┼────────────→
                      Otsu threshold

        Returns:
            Binary mask (0 and 1)

        Example:
            >>> mask = seg.threshold_otsu()
        """
        logger.info("Otsu thresholding")

        # Calculate Otsu threshold
        threshold = filters.threshold_otsu(self.image)

        logger.info(f"  Otsu threshold: {threshold:.2f}")

        # Apply threshold
        mask = self.image > threshold
        mask = mask.astype(np.uint8)

        logger.info(
            f"  Foreground pixels: {mask.sum()} "
            f"({mask.sum() / mask.size * 100:.1f}%)"
        )

        return mask

    def region_growing(
        self, seed: Union[Tuple, List[Tuple]], tolerance: float = 10.0
    ) -> np.ndarray:
        """
        Region growing segmentation.

        GIẢI THÍCH:
        -----------
        Bắt đầu từ seed point, mở rộng dần vùng:

        1. Bắt đầu với seed point
        2. Kiểm tra neighbors (8-connectivity)
        3. Nếu |intensity - seed_intensity| < tolerance:
              Thêm vào region
        4. Lặp lại cho tất cả pixels

        Args:
            seed: Seed point coordinates
                 - 2D: (y, x)
                 - 3D: (z, y, x)
                 - Multiple seeds: [(y1, x1), (y2, x2), ...]
            tolerance: Intensity difference tolerance

        Returns:
            Binary mask (0 and 1)

        Example:
            >>> # Single seed
            >>> mask = seg.region_growing(seed=(128, 128), tolerance=10)
            >>>
            >>> # Multiple seeds
            >>> mask = seg.region_growing(seed=[(100, 100), (150, 150)])
        """
        logger.info(f"Region growing: seed={seed}, tolerance={tolerance}")

        # Handle multiple seeds
        if isinstance(seed, list):
            seeds = seed
        else:
            seeds = [seed]

        # Create markers for random walker
        markers = np.zeros(self.image.shape, dtype=np.int32)

        for i, s in enumerate(seeds, start=1):
            markers[s] = i

        # Random walker segmentation (advanced region growing)
        try:
            labels = random_walker(self.image, markers, beta=tolerance, mode="bf")
            mask = (labels > 0).astype(np.uint8)
        except:
            # Fallback to simple flood fill
            logger.warning("Random walker failed, using flood fill")
            mask = np.zeros(self.image.shape, dtype=np.uint8)

            for s in seeds:
                seed_value = self.image[s]
                lower = seed_value - tolerance
                upper = seed_value + tolerance

                # Flood fill
                filled = ndimage.binary_fill_holes(
                    (self.image >= lower) & (self.image <= upper)
                )
                mask = np.maximum(mask, filled.astype(np.uint8))

        logger.info(
            f"  Segmented pixels: {mask.sum()} "
            f"({mask.sum() / mask.size * 100:.1f}%)"
        )

        return mask

    def morphological_opening(
        self, mask: np.ndarray, kernel_size: int = 3
    ) -> np.ndarray:
        """
        Morphological opening: Erosion + Dilation.

        GIẢI THÍCH:
        -----------
        Opening = Erosion → Dilation

        Tác dụng:
        - Remove small objects (noise)
        - Smooth boundaries
        - Preserve large objects

        Args:
            mask: Binary mask
            kernel_size: Size of structuring element

        Returns:
            Processed mask

        Example:
            >>> opened = seg.morphological_opening(mask, kernel_size=3)
        """
        logger.info(f"Morphological opening: kernel_size={kernel_size}")

        if self.ndim == 2:
            kernel = morphology.disk(kernel_size)
        else:
            kernel = morphology.ball(kernel_size)

        opened = morphology.binary_opening(mask, kernel)

        return opened.astype(np.uint8)

    def morphological_closing(
        self, mask: np.ndarray, kernel_size: int = 3
    ) -> np.ndarray:
        """
        Morphological closing: Dilation + Erosion.

        GIẢI THÍCH:
        -----------
        Closing = Dilation → Erosion

        Tác dụng:
        - Fill small holes
        - Connect nearby objects
        - Smooth boundaries

        Args:
            mask: Binary mask
            kernel_size: Size of structuring element

        Returns:
            Processed mask

        Example:
            >>> closed = seg.morphological_closing(mask, kernel_size=3)
        """
        logger.info(f"Morphological closing: kernel_size={kernel_size}")

        if self.ndim == 2:
            kernel = morphology.disk(kernel_size)
        else:
            kernel = morphology.ball(kernel_size)

        closed = morphology.binary_closing(mask, kernel)

        return closed.astype(np.uint8)

    def remove_small_objects(self, mask: np.ndarray, min_size: int = 100) -> np.ndarray:
        """
        Remove small connected components.

        GIẢI THÍCH:
        -----------
        Loại bỏ các objects nhỏ (noise):
        - Label connected components
        - Remove components < min_size

        Args:
            mask: Binary mask
            min_size: Minimum object size (pixels)

        Returns:
            Cleaned mask

        Example:
            >>> clean = seg.remove_small_objects(mask, min_size=100)
        """
        logger.info(f"Removing small objects: min_size={min_size}")

        cleaned = morphology.remove_small_objects(mask.astype(bool), min_size=min_size)

        removed = mask.sum() - cleaned.sum()
        logger.info(f"  Removed {removed} pixels")

        return cleaned.astype(np.uint8)

    def get_largest_component(self, mask: np.ndarray) -> np.ndarray:
        """
        Keep only the largest connected component.

        GIẢI THÍCH:
        -----------
        Chỉ giữ lại component lớn nhất (thường là brain):
        - Label all components
        - Find largest
        - Keep only that component

        Args:
            mask: Binary mask

        Returns:
            Mask with only largest component

        Example:
            >>> largest = seg.get_largest_component(mask)
        """
        logger.info("Extracting largest connected component")

        # Label components
        labeled = measure.label(mask, connectivity=2)

        if labeled.max() == 0:
            logger.warning("  No components found")
            return mask

        # Find largest
        props = measure.regionprops(labeled)
        largest = max(props, key=lambda x: x.area)

        # Keep only largest
        largest_mask = (labeled == largest.label).astype(np.uint8)

        logger.info(
            f"  Largest component: {largest.area} pixels "
            f"({largest.area / mask.size * 100:.1f}%)"
        )

        return largest_mask

    @staticmethod
    def dice_coefficient(ground_truth: np.ndarray, prediction: np.ndarray) -> float:
        """
        Calculate Dice coefficient (F1 score).

        GIẢI THÍCH:
        -----------
        Dice coefficient đo độ overlap giữa 2 segmentation:

                2 × |A ∩ B|
        Dice = ─────────────
                 |A| + |B|

        Range: 0 (no overlap) to 1 (perfect overlap)

        Dice > 0.7: Good
        Dice > 0.8: Very good
        Dice > 0.9: Excellent

        Args:
            ground_truth: Ground truth mask
            prediction: Predicted mask

        Returns:
            Dice coefficient (0-1)

        Example:
            >>> dice = BrainSegmentation.dice_coefficient(gt, pred)
            >>> print(f"Dice: {dice:.3f}")
        """
        gt = ground_truth.astype(bool)
        pred = prediction.astype(bool)

        intersection = np.logical_and(gt, pred).sum()

        if gt.sum() + pred.sum() == 0:
            return 1.0  # Both empty

        dice = 2.0 * intersection / (gt.sum() + pred.sum())

        return float(dice)

    @staticmethod
    def iou_score(ground_truth: np.ndarray, prediction: np.ndarray) -> float:
        """
        Calculate IoU (Intersection over Union).

        GIẢI THÍCH:
        -----------
        IoU (Jaccard index):

               |A ∩ B|
        IoU = ─────────
              |A ∪ B|

        Range: 0 to 1

        IoU vs Dice:
        - IoU more strict than Dice
        - IoU = Dice / (2 - Dice)

        Args:
            ground_truth: Ground truth mask
            prediction: Predicted mask

        Returns:
            IoU score (0-1)

        Example:
            >>> iou = BrainSegmentation.iou_score(gt, pred)
        """
        gt = ground_truth.astype(bool)
        pred = prediction.astype(bool)

        intersection = np.logical_and(gt, pred).sum()
        union = np.logical_or(gt, pred).sum()

        if union == 0:
            return 1.0  # Both empty

        iou = intersection / union

        return float(iou)

    @staticmethod
    def hausdorff_distance(ground_truth: np.ndarray, prediction: np.ndarray) -> float:
        """
        Calculate Hausdorff distance.

        GIẢI THÍCH:
        -----------
        Hausdorff distance = Khoảng cách lớn nhất giữa 2 boundaries

        Smaller = Better (0 = perfect match)

        Args:
            ground_truth: Ground truth mask
            prediction: Predicted mask

        Returns:
            Hausdorff distance (pixels)

        Example:
            >>> hd = BrainSegmentation.hausdorff_distance(gt, pred)
        """
        from scipy.spatial.distance import directed_hausdorff

        # Get boundaries
        gt_boundary = ndimage.binary_erosion(ground_truth) ^ ground_truth
        pred_boundary = ndimage.binary_erosion(prediction) ^ prediction

        # Get coordinates
        gt_points = np.argwhere(gt_boundary)
        pred_points = np.argwhere(pred_boundary)

        if len(gt_points) == 0 or len(pred_points) == 0:
            return float("inf")

        # Calculate Hausdorff
        hd_1 = directed_hausdorff(gt_points, pred_points)[0]
        hd_2 = directed_hausdorff(pred_points, gt_points)[0]

        hausdorff = max(hd_1, hd_2)

        return float(hausdorff)

    def visualize_segmentation(
        self,
        mask: np.ndarray,
        slice_idx: Optional[int] = None,
        overlay: bool = True,
        figsize: Tuple[int, int] = (15, 5),
        save_path: Optional[str] = None,
    ) -> None:
        """
        Visualize segmentation results.

        Args:
            mask: Segmentation mask
            slice_idx: Slice index for 3D (None = middle)
            overlay: Show overlay on original image
            figsize: Figure size
            save_path: Path to save figure

        Example:
            >>> seg.visualize_segmentation(mask, overlay=True)
        """
        # Get 2D slice for 3D images
        if self.ndim == 3:
            if slice_idx is None:
                slice_idx = self.image.shape[0] // 2
            image_2d = self.image[slice_idx]
            mask_2d = mask[slice_idx]
        else:
            image_2d = self.image
            mask_2d = mask

        fig, axes = plt.subplots(1, 3, figsize=figsize)

        # Original image
        axes[0].imshow(image_2d, cmap="gray")
        axes[0].set_title("Original Image")
        axes[0].axis("off")

        # Segmentation mask
        axes[1].imshow(mask_2d, cmap="gray")
        axes[1].set_title("Segmentation Mask")
        axes[1].axis("off")

        # Overlay
        if overlay:
            axes[2].imshow(image_2d, cmap="gray")
            axes[2].imshow(mask_2d, cmap="Reds", alpha=0.5)
            axes[2].set_title("Overlay")
        else:
            # Masked image
            masked = image_2d * mask_2d
            axes[2].imshow(masked, cmap="gray")
            axes[2].set_title("Masked Image")
        axes[2].axis("off")

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            logger.info(f"Saved visualization to: {save_path}")

        plt.show()
