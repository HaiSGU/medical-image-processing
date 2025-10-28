"""
Image Transformations Module

Implements preprocessing transformations for medical images:
- Intensity normalization (min-max, z-score, percentile)
- Noise reduction (Gaussian, median, bilateral, NLM)
- Resampling and resizing
- Histogram operations

GIẢI THÍCH:
-----------
Preprocessing = Chuẩn bị ảnh trước khi analysis

Mục đích:
- Reduce noise (giảm nhiễu)
- Standardize intensities (chuẩn hóa cường độ)
- Improve quality (cải thiện chất lượng)

Author: HaiSGU
Date: 2025-10-28
"""

import numpy as np
import logging
from typing import Tuple, Optional, Union
import matplotlib.pyplot as plt
from scipy import ndimage
from skimage import filters, exposure, restoration, transform

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ImageTransforms:
    """
    Image preprocessing transformations.

    Các phương pháp:
    1. Normalization: min-max, z-score, percentile
    2. Denoising: Gaussian, median, bilateral, NLM
    3. Histogram: equalization, matching, CLAHE
    4. Geometric: rotate, resize, flip

    Examples:
        >>> image = np.load('brain.npy')
        >>> transforms = ImageTransforms(image)
        >>>
        >>> # Normalize
        >>> normalized = transforms.normalize_minmax()
        >>>
        >>> # Denoise
        >>> denoised = transforms.denoise_gaussian(sigma=1.0)
        >>>
        >>> # Histogram equalization
        >>> enhanced = transforms.histogram_equalization()
    """

    def __init__(self, image: np.ndarray):
        """
        Initialize Image Transforms.

        Args:
            image: Input image (2D or 3D numpy array)

        Example:
            >>> image = np.load('mri.npy')
            >>> transforms = ImageTransforms(image)
        """
        self.image = image.astype(np.float32)
        self.original_dtype = image.dtype

        logger.info(f"ImageTransforms initialized:")
        logger.info(f"  Image shape: {image.shape}")
        logger.info(f"  Image range: [{image.min():.2f}, {image.max():.2f}]")
        logger.info(f"  Data type: {image.dtype}")

    # ==================== NORMALIZATION ====================

    def normalize_minmax(
        self, new_min: float = 0.0, new_max: float = 1.0
    ) -> np.ndarray:
        """
        Min-Max normalization.

        GIẢI THÍCH:
        -----------
        Đưa intensities về range [new_min, new_max]:

                   x - min(x)
        x_norm = ───────────── × (new_max - new_min) + new_min
                 max(x) - min(x)

        Args:
            new_min: New minimum value
            new_max: New maximum value

        Returns:
            Normalized image

        Example:
            >>> # Normalize to [0, 1]
            >>> norm = transforms.normalize_minmax(0, 1)
            >>>
            >>> # Normalize to [0, 255]
            >>> norm = transforms.normalize_minmax(0, 255)
        """
        logger.info(f"Min-Max normalization to [{new_min}, {new_max}]")

        old_min = self.image.min()
        old_max = self.image.max()

        if old_max - old_min == 0:
            logger.warning("Image is constant, returning zeros")
            return np.zeros_like(self.image)

        normalized = (self.image - old_min) / (old_max - old_min)
        normalized = normalized * (new_max - new_min) + new_min

        logger.info(f"  Original range: [{old_min:.2f}, {old_max:.2f}]")
        logger.info(f"  New range: [{normalized.min():.2f}, {normalized.max():.2f}]")

        return normalized

    def normalize_zscore(self) -> np.ndarray:
        """
        Z-score normalization (standardization).

        GIẢI THÍCH:
        -----------
        Đưa về mean=0, std=1:

                 x - mean(x)
        x_std = ─────────────
                   std(x)

        Tốt cho machine learning (neural networks)!

        Returns:
            Standardized image

        Example:
            >>> std = transforms.normalize_zscore()
            >>> print(f"Mean: {std.mean():.6f}")  # ~0
            >>> print(f"Std: {std.std():.6f}")    # ~1
        """
        logger.info("Z-score normalization")

        mean = self.image.mean()
        std = self.image.std()

        if std == 0:
            logger.warning("Image has zero std, returning zeros")
            return np.zeros_like(self.image)

        standardized = (self.image - mean) / std

        logger.info(f"  Original mean: {mean:.2f}, std: {std:.2f}")
        logger.info(
            f"  New mean: {standardized.mean():.6f}, std: {standardized.std():.6f}"
        )

        return standardized

    def normalize_percentile(
        self, lower: float = 1.0, upper: float = 99.0
    ) -> np.ndarray:
        """
        Percentile normalization (robust to outliers).

        GIẢI THÍCH:
        -----------
        1. Clip to percentiles (remove outliers)
        2. Min-max normalize

        Robust hơn min-max thông thường!

        Args:
            lower: Lower percentile (0-100)
            upper: Upper percentile (0-100)

        Returns:
            Normalized image

        Example:
            >>> # Clip to 1st-99th percentile
            >>> norm = transforms.normalize_percentile(1, 99)
        """
        logger.info(f"Percentile normalization [{lower}%, {upper}%]")

        p_low = np.percentile(self.image, lower)
        p_high = np.percentile(self.image, upper)

        logger.info(f"  {lower}th percentile: {p_low:.2f}")
        logger.info(f"  {upper}th percentile: {p_high:.2f}")

        # Clip
        clipped = np.clip(self.image, p_low, p_high)

        # Normalize
        if p_high - p_low == 0:
            return np.zeros_like(self.image)

        normalized = (clipped - p_low) / (p_high - p_low)

        return normalized

    # ==================== DENOISING ====================

    def denoise_gaussian(self, sigma: float = 1.0) -> np.ndarray:
        """
        Gaussian filtering.

        GIẢI THÍCH:
        -----------
        Convolution với Gaussian kernel:

        Kernel ~ exp(-(x² + y²) / (2σ²))

        Ưu điểm: Đơn giản, nhanh
        Nhược điểm: Blur edges

        Args:
            sigma: Standard deviation of Gaussian kernel
                  - Small sigma (0.5-1): Less smoothing
                  - Large sigma (2-5): More smoothing

        Returns:
            Denoised image

        Example:
            >>> denoised = transforms.denoise_gaussian(sigma=1.5)
        """
        logger.info(f"Gaussian denoising: sigma={sigma}")

        denoised = ndimage.gaussian_filter(self.image, sigma=sigma)

        return denoised

    def denoise_median(self, size: int = 3) -> np.ndarray:
        """
        Median filtering.

        GIẢI THÍCH:
        -----------
        Thay mỗi pixel bằng median của neighbors:

        Neighborhood → Sort → Take median

        Ưu điểm: Preserves edges, tốt cho salt & pepper noise
        Nhược điểm: Chậm hơn Gaussian

        Args:
            size: Neighborhood size (odd number)

        Returns:
            Denoised image

        Example:
            >>> denoised = transforms.denoise_median(size=5)
        """
        logger.info(f"Median filtering: size={size}")

        denoised = ndimage.median_filter(self.image, size=size)

        return denoised

    def denoise_bilateral(
        self, sigma_spatial: float = 15, sigma_color: float = 0.05
    ) -> np.ndarray:
        """
        Bilateral filtering.

        GIẢI THÍCH:
        -----------
        Weighted average dựa trên:
        1. Spatial distance (như Gaussian)
        2. Intensity similarity

        Result: Smooth nhưng preserve edges!

        Args:
            sigma_spatial: Spatial sigma (kernel size)
            sigma_color: Intensity sigma (similarity)

        Returns:
            Denoised image

        Example:
            >>> denoised = transforms.denoise_bilateral(
            ...     sigma_spatial=15, sigma_color=0.05
            ... )
        """
        logger.info(
            f"Bilateral filtering: spatial={sigma_spatial}, color={sigma_color}"
        )

        # Normalize for bilateral (works best on [0,1])
        image_norm = (self.image - self.image.min()) / (
            self.image.max() - self.image.min()
        )

        denoised = restoration.denoise_bilateral(
            image_norm,
            sigma_spatial=sigma_spatial,
            sigma_color=sigma_color,
            channel_axis=None,
        )

        # Denormalize
        denoised = denoised * (self.image.max() - self.image.min()) + self.image.min()

        return denoised

    def denoise_nlm(
        self, patch_size: int = 7, patch_distance: int = 11, h: float = 0.1
    ) -> np.ndarray:
        """
        Non-Local Means (NLM) denoising.

        GIẢI THÍCH:
        -----------
        So sánh patches (không chỉ pixels):

        Tìm similar patches trong search window
        Weighted average based on patch similarity

        Tốt nhất cho MRI! (Medical images có self-similarity)

        Args:
            patch_size: Size of patches (odd number)
            patch_distance: Search window size
            h: Filter strength (higher = more smoothing)

        Returns:
            Denoised image

        Example:
            >>> denoised = transforms.denoise_nlm()
        """
        logger.info(f"NLM denoising: patch_size={patch_size}, h={h}")

        # Normalize
        image_norm = (self.image - self.image.min()) / (
            self.image.max() - self.image.min()
        )

        denoised = restoration.denoise_nl_means(
            image_norm,
            patch_size=patch_size,
            patch_distance=patch_distance,
            h=h,
            channel_axis=None,
            fast_mode=True,
        )

        # Denormalize
        denoised = denoised * (self.image.max() - self.image.min()) + self.image.min()

        return denoised

    # ==================== HISTOGRAM OPERATIONS ====================

    def histogram_equalization(self) -> np.ndarray:
        """
        Histogram equalization.

        GIẢI THÍCH:
        -----------
        Spread histogram uniformly:

        Before:    │    ▄▄▄
                   │  ▄▄███▄▄
                   └────────────

        After:     │▄  ▄  ▄  ▄
                   │█  █  █  █
                   └────────────

        Tăng contrast!

        Returns:
            Equalized image

        Example:
            >>> equalized = transforms.histogram_equalization()
        """
        logger.info("Histogram equalization")

        equalized = exposure.equalize_hist(self.image)

        return equalized

    def adaptive_histogram_equalization(self, clip_limit: float = 0.03) -> np.ndarray:
        """
        CLAHE (Contrast Limited Adaptive Histogram Equalization).

        GIẢI THÍCH:
        -----------
        Histogram equalization locally (từng vùng):

        Better than global histogram equalization!
        Prevent over-amplification of noise

        Args:
            clip_limit: Contrast limit (0-1)
                       Higher = more contrast

        Returns:
            Enhanced image

        Example:
            >>> enhanced = transforms.adaptive_histogram_equalization()
        """
        logger.info(f"CLAHE: clip_limit={clip_limit}")

        # Normalize to [0, 1]
        image_norm = (self.image - self.image.min()) / (
            self.image.max() - self.image.min()
        )

        enhanced = exposure.equalize_adapthist(image_norm, clip_limit=clip_limit)

        # Denormalize
        enhanced = enhanced * (self.image.max() - self.image.min()) + self.image.min()

        return enhanced

    # ==================== GEOMETRIC TRANSFORMS ====================

    def rotate(self, angle: float) -> np.ndarray:
        """
        Rotate image.

        Args:
            angle: Rotation angle in degrees

        Returns:
            Rotated image

        Example:
            >>> rotated = transforms.rotate(45)
        """
        logger.info(f"Rotating by {angle} degrees")

        rotated = ndimage.rotate(self.image, angle, reshape=False)

        return rotated

    def resize(self, output_shape: Tuple[int, ...], order: int = 1) -> np.ndarray:
        """
        Resize image.

        Args:
            output_shape: Target shape
            order: Interpolation order
                  0 = nearest neighbor
                  1 = bilinear (default)
                  3 = bicubic

        Returns:
            Resized image

        Example:
            >>> resized = transforms.resize((128, 128))
        """
        logger.info(f"Resizing from {self.image.shape} to {output_shape}")

        resized = transform.resize(
            self.image,
            output_shape,
            order=order,
            preserve_range=True,
            anti_aliasing=True,
        )

        return resized

    def flip(self, axis: int = 0) -> np.ndarray:
        """
        Flip image along axis.

        Args:
            axis: Axis to flip (0=vertical, 1=horizontal)

        Returns:
            Flipped image

        Example:
            >>> flipped = transforms.flip(axis=1)  # Horizontal flip
        """
        logger.info(f"Flipping along axis {axis}")

        flipped = np.flip(self.image, axis=axis)

        return flipped

    # ==================== VISUALIZATION ====================

    def compare_denoising(
        self,
        methods: Optional[list] = None,
        figsize: Tuple[int, int] = (20, 5),
        save_path: Optional[str] = None,
    ) -> None:
        """
        Compare different denoising methods.

        Args:
            methods: List of methods to compare
            figsize: Figure size
            save_path: Path to save figure

        Example:
            >>> transforms.compare_denoising()
        """
        if methods is None:
            methods = [
                ("Original", self.image),
                ("Gaussian", self.denoise_gaussian(1.0)),
                ("Median", self.denoise_median(3)),
                ("Bilateral", self.denoise_bilateral()),
            ]

        fig, axes = plt.subplots(1, len(methods), figsize=figsize)

        for ax, (name, img) in zip(axes, methods):
            if self.image.ndim == 3:
                # Show middle slice for 3D
                img_2d = img[img.shape[0] // 2]
            else:
                img_2d = img

            ax.imshow(img_2d, cmap="gray")
            ax.set_title(name)
            ax.axis("off")

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            logger.info(f"Saved to: {save_path}")

        plt.show()
