"""
Lesion and Tumor Detection Module

Detects and localizes abnormalities in medical images using:
- Threshold-based detection
- Blob detection
- Morphological operations
- Connected component analysis

Author: HaiSGU
Date: 2025-11-23
"""

import numpy as np
from typing import List, Tuple, Dict, Optional
import logging
from skimage import morphology, measure
from skimage.feature import blob_dog, blob_log, blob_doh
from scipy import ndimage
import cv2

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Detection:
    """
    Represents a detected region/lesion.

    Attributes:
        bbox: Bounding box (min_row, min_col, max_row, max_col)
        centroid: Center point (row, col)
        area: Number of pixels
        confidence: Detection confidence score (0-1)
        properties: Additional region properties
    """

    def __init__(
        self,
        bbox: Tuple[int, int, int, int],
        centroid: Tuple[float, float],
        area: float,
        confidence: float = 1.0,
        properties: Optional[Dict] = None,
    ):
        self.bbox = bbox
        self.centroid = centroid
        self.area = area
        self.confidence = confidence
        self.properties = properties or {}

    def __repr__(self):
        return f"Detection(area={self.area:.0f}, centroid={self.centroid}, conf={self.confidence:.2f})"


class LesionDetector:
    """
    Detect lesions, tumors, and abnormalities in medical images.

    Uses traditional computer vision methods:
    - Intensity thresholding
    - Blob detection (LoG, DoG, DoH)
    - Morphological operations
    - Connected component analysis

    Examples:
        >>> from src.computer_vision.detection import LesionDetector
        >>> import numpy as np
        >>>
        >>> # Create detector
        >>> detector = LesionDetector()
        >>>
        >>> # Threshold-based detection
        >>> image = np.random.rand(256, 256)
        >>> detections = detector.detect_by_threshold(image, threshold=0.7)
        >>>
        >>> # Blob detection
        >>> blobs = detector.detect_blobs(image, method='log')
        >>>
        >>> # Get all detections
        >>> all_detections = detector.detect_lesions(image, threshold=0.5)
        >>> print(f"Found {len(all_detections)} lesions")
    """

    def __init__(self, min_area: int = 50, max_area: int = 10000):
        """
        Initialize lesion detector.

        Args:
            min_area: Minimum area (pixels) to consider as detection
            max_area: Maximum area (pixels) to consider as detection
        """
        self.min_area = min_area
        self.max_area = max_area
        logger.info(f"LesionDetector initialized (area range: {min_area}-{max_area})")

    def detect_by_threshold(
        self,
        image: np.ndarray,
        threshold: Optional[float] = None,
        adaptive: bool = False,
        percentile: float = 95,
    ) -> List[Detection]:
        """
        Detect lesions using intensity thresholding.

        Phát hiện vùng sáng (high intensity) như khối u hoặc tổn thương.

        Args:
            image: 2D grayscale image (normalized 0-1 or 0-255)
            threshold: Manual threshold value. If None, auto-compute
            adaptive: Use adaptive thresholding
            percentile: Percentile for auto-threshold (if threshold=None)

        Returns:
            List of Detection objects

        Example:
            >>> # Auto threshold at 95th percentile
            >>> detections = detector.detect_by_threshold(image)
            >>>
            >>> # Manual threshold
            >>> detections = detector.detect_by_threshold(image, threshold=0.8)
            >>>
            >>> for det in detections:
            ...     print(f"Lesion at {det.centroid}, area={det.area}")
        """
        try:
            # Normalize image to 0-1
            image = self._normalize_image(image)

            # Determine threshold
            if threshold is None:
                if adaptive:
                    # Otsu's method
                    from skimage.filters import threshold_otsu

                    threshold = threshold_otsu(image)
                else:
                    # Percentile-based
                    threshold = np.percentile(image, percentile)

            logger.info(f"Using threshold: {threshold:.3f}")

            # Threshold image
            binary = image > threshold

            # Morphological cleanup
            binary = morphology.remove_small_objects(binary, min_size=self.min_area)
            binary = morphology.remove_small_holes(
                binary, area_threshold=self.min_area // 2
            )

            # Find connected components
            labeled = measure.label(binary)
            regions = measure.regionprops(labeled)

            # Convert regions to Detection objects
            detections = []
            for region in regions:
                if self.min_area <= region.area <= self.max_area:
                    bbox = region.bbox  # (min_row, min_col, max_row, max_col)
                    centroid = region.centroid  # (row, col)

                    # Calculate confidence based on mean intensity
                    mask = labeled == region.label
                    mean_intensity = np.mean(image[mask])
                    confidence = float(min(mean_intensity / threshold, 1.0))

                    detection = Detection(
                        bbox=bbox,
                        centroid=centroid,
                        area=region.area,
                        confidence=confidence,
                        properties={
                            "mean_intensity": mean_intensity,
                            "eccentricity": region.eccentricity,
                            "solidity": region.solidity,
                        },
                    )
                    detections.append(detection)

            logger.info(f"✅ Detected {len(detections)} lesions by threshold")
            return detections

        except Exception as e:
            logger.error(f"Threshold detection failed: {e}")
            return []

    def detect_blobs(
        self,
        image: np.ndarray,
        method: str = "log",
        min_sigma: float = 1,
        max_sigma: float = 30,
        num_sigma: int = 10,
        threshold: float = 0.1,
    ) -> List[Detection]:
        """
        Detect blob-like structures (tumors, nodules).

        Sử dụng Laplacian of Gaussian (LoG) hoặc các phương pháp khác
        để tìm các vùng tròn, oval giống khối u.

        Args:
            image: 2D grayscale image
            method: 'log' (Laplacian of Gaussian), 'dog' (Difference of Gaussian),
                   or 'doh' (Determinant of Hessian)
            min_sigma: Minimum blob size
            max_sigma: Maximum blob size
            num_sigma: Number of intermediate values for sigma
            threshold: Detection threshold (lower = more sensitive)

        Returns:
            List of Detection objects

        Example:
            >>> # Detect medium-sized blobs
            >>> blobs = detector.detect_blobs(
            ...     image,
            ...     method='log',
            ...     min_sigma=5,
            ...     max_sigma=20,
            ...     threshold=0.05
            ... )
            >>> print(f"Found {len(blobs)} blob-like lesions")
        """
        try:
            # Normalize image
            image = self._normalize_image(image)

            # Select blob detection method
            if method == "log":
                blobs = blob_log(
                    image,
                    min_sigma=min_sigma,
                    max_sigma=max_sigma,
                    num_sigma=num_sigma,
                    threshold=threshold,
                )
            elif method == "dog":
                blobs = blob_dog(
                    image, min_sigma=min_sigma, max_sigma=max_sigma, threshold=threshold
                )
            elif method == "doh":
                blobs = blob_doh(
                    image,
                    min_sigma=min_sigma,
                    max_sigma=max_sigma,
                    num_sigma=num_sigma,
                    threshold=threshold,
                )
            else:
                raise ValueError(f"Unknown method: {method}")

            # Convert blobs to Detection objects
            detections = []
            for blob in blobs:
                row, col, sigma = blob

                # Estimate radius
                radius = sigma * np.sqrt(2)  # For LoG
                area = np.pi * radius**2

                # Create bounding box
                min_row = int(max(0, row - radius))
                max_row = int(min(image.shape[0], row + radius))
                min_col = int(max(0, col - radius))
                max_col = int(min(image.shape[1], col + radius))
                bbox = (min_row, min_col, max_row, max_col)

                # Calculate confidence (based on local intensity)
                y, x = int(row), int(col)
                r = int(radius)
                roi = image[
                    max(0, y - r) : min(image.shape[0], y + r),
                    max(0, x - r) : min(image.shape[1], x + r),
                ]
                confidence = float(np.mean(roi))

                detection = Detection(
                    bbox=bbox,
                    centroid=(row, col),
                    area=area,
                    confidence=confidence,
                    properties={"sigma": sigma, "radius": radius, "method": method},
                )
                detections.append(detection)

            logger.info(f"✅ Detected {len(detections)} blobs using {method}")
            return detections

        except Exception as e:
            logger.error(f"Blob detection failed: {e}")
            return []

    def detect_lesions(
        self,
        image: np.ndarray,
        threshold: Optional[float] = None,
        use_blobs: bool = True,
        combine: bool = True,
    ) -> List[Detection]:
        """
        Comprehensive lesion detection combining multiple methods.

        Args:
            image: 2D grayscale image
            threshold: Threshold for intensity-based detection
            use_blobs: Also use blob detection
            combine: Merge overlapping detections

        Returns:
            List of all detected lesions

        Example:
            >>> # Complete lesion detection
            >>> all_lesions = detector.detect_lesions(
            ...     image,
            ...     threshold=0.7,
            ...     use_blobs=True,
            ...     combine=True
            ... )
            >>>
            >>> # Sort by confidence
            >>> all_lesions.sort(key=lambda d: d.confidence, reverse=True)
            >>> print(f"Top detection: {all_lesions[0]}")
        """
        all_detections = []

        # Threshold-based detection
        thresh_detections = self.detect_by_threshold(image, threshold=threshold)
        all_detections.extend(thresh_detections)

        # Blob-based detection
        if use_blobs:
            blob_detections = self.detect_blobs(image, method="log")
            all_detections.extend(blob_detections)

        # Remove duplicates if requested
        if combine and len(all_detections) > 1:
            all_detections = self._remove_duplicates(all_detections)

        logger.info(f"✅ Total detections: {len(all_detections)}")
        return all_detections

    def visualize_detections(
        self,
        image: np.ndarray,
        detections: List[Detection],
        color: Tuple[int, int, int] = (255, 0, 0),
        thickness: int = 2,
    ) -> np.ndarray:
        """
        Draw bounding boxes on image.

        Args:
            image: Original image
            detections: List of Detection objects
            color: Box color (R, G, B)
            thickness: Line thickness

        Returns:
            Image with bounding boxes drawn

        Example:
            >>> detections = detector.detect_lesions(image)
            >>> vis_image = detector.visualize_detections(image, detections)
            >>> plt.imshow(vis_image)
            >>> plt.show()
        """
        # Convert to RGB if grayscale
        if image.ndim == 2:
            vis_image = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_GRAY2RGB)
        else:
            vis_image = image.copy()

        # Draw each detection
        for det in detections:
            min_row, min_col, max_row, max_col = det.bbox

            # Draw rectangle
            cv2.rectangle(
                vis_image, (min_col, min_row), (max_col, max_row), color, thickness
            )

            # Draw centroid
            cy, cx = det.centroid
            cv2.circle(vis_image, (int(cx), int(cy)), 3, color, -1)

            # Draw confidence
            label = f"{det.confidence:.2f}"
            cv2.putText(
                vis_image,
                label,
                (min_col, min_row - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1,
            )

        return vis_image

    # ==================== Helper Methods ====================

    def _normalize_image(self, image: np.ndarray) -> np.ndarray:
        """Normalize image to 0-1 range."""
        image = image.astype(np.float32)
        if image.max() > 1.0:
            image = image / 255.0
        return image

    def _remove_duplicates(
        self, detections: List[Detection], iou_threshold: float = 0.3
    ) -> List[Detection]:
        """
        Remove duplicate detections using Non-Maximum Suppression.

        Args:
            detections: List of detections
            iou_threshold: IoU threshold for considering duplicates

        Returns:
            Filtered list of detections
        """
        if len(detections) == 0:
            return []

        # Sort by confidence (descending)
        detections = sorted(detections, key=lambda d: d.confidence, reverse=True)

        keep = []
        while len(detections) > 0:
            # Keep highest confidence detection
            current = detections.pop(0)
            keep.append(current)

            # Remove overlapping detections
            detections = [
                d
                for d in detections
                if self._compute_iou(current.bbox, d.bbox) < iou_threshold
            ]

        logger.debug(f"Kept {len(keep)} detections after NMS")
        return keep

    def _compute_iou(
        self, bbox1: Tuple[int, int, int, int], bbox2: Tuple[int, int, int, int]
    ) -> float:
        """Compute Intersection over Union between two bounding boxes."""
        min_r1, min_c1, max_r1, max_c1 = bbox1
        min_r2, min_c2, max_r2, max_c2 = bbox2

        # Intersection
        inter_min_r = max(min_r1, min_r2)
        inter_min_c = max(min_c1, min_c2)
        inter_max_r = min(max_r1, max_r2)
        inter_max_c = min(max_c1, max_c2)

        if inter_max_r <= inter_min_r or inter_max_c <= inter_min_c:
            return 0.0

        inter_area = (inter_max_r - inter_min_r) * (inter_max_c - inter_min_c)

        # Union
        area1 = (max_r1 - min_r1) * (max_c1 - min_c1)
        area2 = (max_r2 - min_r2) * (max_c2 - min_c2)
        union_area = area1 + area2 - inter_area

        return inter_area / union_area if union_area > 0 else 0.0


# Convenience function
def detect_lesions(
    image: np.ndarray,
    threshold: Optional[float] = None,
    min_area: int = 50,
    max_area: int = 10000,
) -> List[Detection]:
    """
    Convenience function for quick lesion detection.

    Args:
        image: 2D grayscale image
        threshold: Detection threshold (None for auto)
        min_area: Minimum lesion area
        max_area: Maximum lesion area

    Returns:
        List of detected lesions

    Example:
        >>> lesions = detect_lesions(brain_scan, threshold=0.75)
        >>> print(f"Found {len(lesions)} potential lesions")
    """
    detector = LesionDetector(min_area=min_area, max_area=max_area)
    return detector.detect_lesions(image, threshold=threshold)
