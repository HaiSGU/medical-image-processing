"""
Feature Extraction Module

Extracts quantitative features from medical images for analysis:
- Texture features: GLCM, LBP, Haralick
- Shape features: Area, perimeter, circularity, moments
- Intensity features: Mean, std, histogram statistics

Author: HaiSGU
Date: 2025-11-23
"""

import numpy as np
from typing import Dict, Optional, Tuple, List
import logging
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern
from skimage.measure import label, regionprops
from scipy.stats import skew, kurtosis
import warnings

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suppress warnings
warnings.filterwarnings("ignore")


class FeatureExtractor:
    """
    Extract quantitative features from medical images.

    Features include texture (GLCM, LBP), shape, and intensity statistics.
    Useful for radiomics analysis and machine learning.

    Examples:
        >>> from src.computer_vision.feature_extraction import FeatureExtractor
        >>> import numpy as np
        >>>
        >>> # Create feature extractor
        >>> extractor = FeatureExtractor()
        >>>
        >>> # Extract texture features from grayscale image
        >>> image = np.random.rand(128, 128)
        >>> texture_features = extractor.extract_texture_features(image)
        >>>
        >>> # Extract shape features from binary mask
        >>> mask = image > 0.5
        >>> shape_features = extractor.extract_shape_features(mask)
        >>>
        >>> # Extract all features
        >>> all_features = extractor.extract_all_features(image, mask)
    """

    def __init__(self):
        """Initialize feature extractor."""
        self.feature_names = []
        logger.info("FeatureExtractor initialized")

    # ==================== Texture Features ====================

    def extract_glcm_features(
        self,
        image: np.ndarray,
        distances: List[int] = [1, 2, 3],
        angles: List[float] = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
        levels: int = 256,
        symmetric: bool = True,
        normed: bool = True,
    ) -> Dict[str, float]:
        """
        Extract Gray Level Co-occurrence Matrix (GLCM) texture features.

        GLCM đo lường mối quan hệ giữa các pixels lân cận để phân tích texture.

        Args:
            image: 2D grayscale image (0-1 or 0-255)
            distances: List of pixel pair distance offsets
            angles: List of pixel pair angles (in radians)
            levels: Number of gray levels (reduce for faster computation)
            symmetric: If True, GLCM is symmetric
            normed: If True, normalize GLCM

        Returns:
            Dictionary with GLCM features:
            - contrast: Intensity contrast between pixel and neighbor
            - dissimilarity: Similar to contrast but linear
            - homogeneity: Closeness of distribution to diagonal
            - energy: Sum of squared elements (uniformity)
            - correlation: Pixel pair correlation
            - ASM: Angular Second Moment (energy squared)

        Example:
            >>> features = extractor.extract_glcm_features(image)
            >>> print(f"Contrast: {features['glcm_contrast']:.3f}")
            >>> print(f"Homogeneity: {features['glcm_homogeneity']:.3f}")
        """
        try:
            # Normalize to 0-255 and convert to uint8
            if image.max() <= 1.0:
                image = (image * 255).astype(np.uint8)
            else:
                image = image.astype(np.uint8)

            # Reduce levels if too high (for efficiency)
            if levels > 64:
                image = (image // (256 // levels)).astype(np.uint8)

            # Compute GLCM
            glcm = graycomatrix(
                image,
                distances=distances,
                angles=angles,
                levels=levels,
                symmetric=symmetric,
                normed=normed,
            )

            # Extract properties
            features = {}
            properties = [
                "contrast",
                "dissimilarity",
                "homogeneity",
                "energy",
                "correlation",
                "ASM",
            ]

            for prop in properties:
                values = graycoprops(glcm, prop)
                # Average over all distances and angles
                features[f"glcm_{prop}"] = float(np.mean(values))

            logger.debug(f"Extracted {len(features)} GLCM features")
            return features

        except Exception as e:
            logger.error(f"GLCM feature extraction failed: {e}")
            return {}

    def extract_lbp_features(
        self,
        image: np.ndarray,
        radius: int = 3,
        n_points: int = 24,
        method: str = "uniform",
    ) -> Dict[str, float]:
        """
        Extract Local Binary Pattern (LBP) texture features.

        LBP mô tả texture bằng cách so sánh pixel với các pixels xung quanh.

        Args:
            image: 2D grayscale image
            radius: Radius of circle around center pixel
            n_points: Number of circularly symmetric points
            method: 'uniform', 'ror', 'var', or 'default'

        Returns:
            Dictionary with LBP histogram statistics:
            - Mean, std, energy of LBP histogram
            - Entropy (texture randomness)

        Example:
            >>> lbp_features = extractor.extract_lbp_features(image)
            >>> print(f"LBP Mean: {lbp_features['lbp_mean']:.3f}")
        """
        try:
            # Ensure 2D
            if image.ndim > 2:
                image = image[:, :, 0] if image.shape[2] == 3 else image.mean(axis=2)

            # Compute LBP
            lbp = local_binary_pattern(image, n_points, radius, method=method)

            # Compute histogram
            hist, _ = np.histogram(
                lbp.ravel(), bins=n_points + 2, range=(0, n_points + 2), density=True
            )

            # Extract features from histogram
            features = {
                "lbp_mean": float(np.mean(lbp)),
                "lbp_std": float(np.std(lbp)),
                "lbp_energy": float(np.sum(hist**2)),
                "lbp_entropy": float(-np.sum(hist * np.log2(hist + 1e-10))),
            }

            logger.debug(f"Extracted {len(features)} LBP features")
            return features

        except Exception as e:
            logger.error(f"LBP feature extraction failed: {e}")
            return {}

    def extract_texture_features(
        self, image: np.ndarray, use_glcm: bool = True, use_lbp: bool = True
    ) -> Dict[str, float]:
        """
        Extract all texture features.

        Args:
            image: 2D grayscale image
            use_glcm: Extract GLCM features
            use_lbp: Extract LBP features

        Returns:
            Combined dictionary of all texture features
        """
        features = {}

        if use_glcm:
            features.update(self.extract_glcm_features(image))

        if use_lbp:
            features.update(self.extract_lbp_features(image))

        return features

    # ==================== Shape Features ====================

    def extract_shape_features(
        self, binary_image: np.ndarray, largest_only: bool = True
    ) -> Dict[str, float]:
        """
        Extract shape features from binary image or mask.

        Args:
            binary_image: 2D binary image (0 or 1)
            largest_only: If True, only analyze largest connected component

        Returns:
            Dictionary with shape features:
            - area: Number of pixels
            - perimeter: Perimeter length
            - circularity: 4π * area / perimeter²
            - eccentricity: Ratio of focal distance to major axis
            - solidity: Area / convex hull area
            - extent: Area / bounding box area
            - major_axis, minor_axis: Ellipse axes lengths

        Example:
            >>> mask = image > 0.5  # Threshold to create binary mask
            >>> shape = extractor.extract_shape_features(mask)
            >>> print(f"Area: {shape['area']:.0f} pixels")
            >>> print(f"Circularity: {shape['circularity']:.3f}")
        """
        try:
            # Ensure binary
            binary_image = (binary_image > 0).astype(np.uint8)

            # Label connected components
            labeled = label(binary_image)
            regions = regionprops(labeled)

            if len(regions) == 0:
                logger.warning("No regions found in binary image")
                return {}

            # Select region to analyze
            if largest_only:
                region = max(regions, key=lambda r: r.area)
            else:
                region = regions[0]

            # Extract shape features
            features = {
                "area": float(region.area),
                "perimeter": float(region.perimeter),
                "eccentricity": float(region.eccentricity),
                "solidity": float(region.solidity),
                "extent": float(region.extent),
                "major_axis_length": float(region.major_axis_length),
                "minor_axis_length": float(region.minor_axis_length),
            }

            # Calculate circularity
            if region.perimeter > 0:
                features["circularity"] = float(
                    4 * np.pi * region.area / (region.perimeter**2)
                )
            else:
                features["circularity"] = 0.0

            # Moments
            features["centroid_x"] = float(region.centroid[1])
            features["centroid_y"] = float(region.centroid[0])

            logger.debug(f"Extracted {len(features)} shape features")
            return features

        except Exception as e:
            logger.error(f"Shape feature extraction failed: {e}")
            return {}

    # ==================== Intensity Features ====================

    def extract_intensity_features(
        self, image: np.ndarray, mask: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """
        Extract intensity-based features.

        Args:
            image: 2D grayscale image
            mask: Optional binary mask to restrict analysis

        Returns:
            Dictionary with intensity features:
            - mean, median, std: Basic statistics
            - min, max, range: Value range
            - percentile_25, 75: Quartiles
            - skewness: Asymmetry of distribution
            - kurtosis: Tailedness of distribution
            - energy: Sum of squared intensities

        Example:
            >>> intensity = extractor.extract_intensity_features(image)
            >>> print(f"Mean intensity: {intensity['mean']:.3f}")
            >>> print(f"Std: {intensity['std']:.3f}")
        """
        try:
            # Apply mask if provided
            if mask is not None:
                values = image[mask > 0]
            else:
                values = image.ravel()

            if len(values) == 0:
                logger.warning("No pixels to analyze")
                return {}

            # Basic statistics
            features = {
                "mean": float(np.mean(values)),
                "median": float(np.median(values)),
                "std": float(np.std(values)),
                "min": float(np.min(values)),
                "max": float(np.max(values)),
                "range": float(np.ptp(values)),
                "percentile_25": float(np.percentile(values, 25)),
                "percentile_75": float(np.percentile(values, 75)),
            }

            # Higher order statistics
            features["skewness"] = float(skew(values))
            features["kurtosis"] = float(kurtosis(values))

            # Energy
            features["energy"] = float(np.sum(values**2))

            # Coefficient of variation
            if features["mean"] != 0:
                features["cv"] = features["std"] / features["mean"]
            else:
                features["cv"] = 0.0

            logger.debug(f"Extracted {len(features)} intensity features")
            return features

        except Exception as e:
            logger.error(f"Intensity feature extraction failed: {e}")
            return {}

    # ==================== Combined Extraction ====================

    def extract_all_features(
        self,
        image: np.ndarray,
        mask: Optional[np.ndarray] = None,
        extract_texture: bool = True,
        extract_shape: bool = True,
        extract_intensity: bool = True,
    ) -> Dict[str, float]:
        """
        Extract all available features from image.

        Args:
            image: 2D grayscale image
            mask: Optional binary mask for shape features
            extract_texture: Extract texture features (GLCM, LBP)
            extract_shape: Extract shape features (requires mask)
            extract_intensity: Extract intensity features

        Returns:
            Combined dictionary with all features

        Example:
            >>> # Extract all features
            >>> all_features = extractor.extract_all_features(
            ...     image,
            ...     mask=binary_mask,
            ...     extract_texture=True,
            ...     extract_shape=True,
            ...     extract_intensity=True
            ... )
            >>> print(f"Total features: {len(all_features)}")
            >>> print(f"Feature names: {list(all_features.keys())[:5]}")
        """
        all_features = {}

        # Texture features
        if extract_texture:
            logger.info("Extracting texture features...")
            texture = self.extract_texture_features(image)
            all_features.update(texture)

        # Shape features
        if extract_shape and mask is not None:
            logger.info("Extracting shape features...")
            shape = self.extract_shape_features(mask)
            all_features.update(shape)

        # Intensity features
        if extract_intensity:
            logger.info("Extracting intensity features...")
            intensity = self.extract_intensity_features(image, mask)
            all_features.update(intensity)

        logger.info(f"✅ Extracted {len(all_features)} total features")
        return all_features

    def get_feature_vector(
        self, features: Dict[str, float]
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Convert feature dictionary to numpy array for ML.

        Args:
            features: Dictionary of features

        Returns:
            Tuple of (feature_vector, feature_names)

        Example:
            >>> features = extractor.extract_all_features(image)
            >>> vector, names = extractor.get_feature_vector(features)
            >>> print(f"Feature vector shape: {vector.shape}")
            >>> print(f"First 5 features: {names[:5]}")
        """
        feature_names = sorted(features.keys())
        feature_vector = np.array([features[name] for name in feature_names])

        return feature_vector, feature_names


# Convenience function
def extract_features(
    image: np.ndarray,
    mask: Optional[np.ndarray] = None,
    feature_types: List[str] = ["texture", "intensity", "shape"],
) -> Dict[str, float]:
    """
    Convenience function to extract features.

    Args:
        image: 2D grayscale image
        mask: Optional binary mask
        feature_types: List of feature types to extract
                      Options: 'texture', 'intensity', 'shape'

    Returns:
        Dictionary of features

    Example:
        >>> features = extract_features(image, mask, ['texture', 'intensity'])
        >>> print(f"Extracted {len(features)} features")
    """
    extractor = FeatureExtractor()

    return extractor.extract_all_features(
        image,
        mask,
        extract_texture="texture" in feature_types,
        extract_shape="shape" in feature_types and mask is not None,
        extract_intensity="intensity" in feature_types,
    )
