"""
CT Reconstruction Module

Implements CT image reconstruction from sinogram data using:
- Filtered Back Projection (FBP)
- Simultaneous Algebraic Reconstruction Technique (SART)

Author: HaiSGU
Date: 2025-10-27
"""

import numpy as np
import logging
from typing import Tuple, Optional, Dict
from skimage.transform import radon, iradon, iradon_sart
from scipy import signal
import matplotlib.pyplot as plt

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CTReconstructor:
    """
    CT Image Reconstruction from Sinogram.

    CT reconstruction tái tạo ảnh 2D từ sinogram (projection data).

    Attributes:
        sinogram: 2D array (num_angles, num_detectors)
        theta: Array of projection angles in degrees

    Examples:
        >>> # Load sinogram
        >>> sinogram = np.load('sinogram.npy')
        >>>
        >>> # Create reconstructor
        >>> reconstructor = CTReconstructor(sinogram)
        >>>
        >>> # FBP reconstruction
        >>> image_fbp = reconstructor.reconstruct_fbp(filter_name='ramp')
        >>>
        >>> # SART reconstruction
        >>> image_sart = reconstructor.reconstruct_sart(iterations=5)
        >>>
        >>> # Compare methods
        >>> comparison = reconstructor.compare_methods()
    """

    def __init__(self, sinogram: np.ndarray, theta: Optional[np.ndarray] = None):
        """
        Initialize CT Reconstructor.

        Args:
            sinogram: 2D array of projection data
                     Shape: (num_angles, num_detectors)
                     - Mỗi hàng = một góc chiếu
                     - Mỗi cột = một detector position

            theta: Array of projection angles in degrees
                   Nếu None, tự động tạo từ 0° đến 180°

        Example:
            >>> sinogram = np.random.rand(180, 256)  # 180 angles, 256 detectors
            >>> reconstructor = CTReconstructor(sinogram)
        """
        self.sinogram = sinogram

        # Validate sinogram shape
        if sinogram.ndim != 2:
            raise ValueError(f"Sinogram must be 2D, got shape {sinogram.shape}")

        num_angles, num_detectors = sinogram.shape

        # Set projection angles
        if theta is None:
            # Mặc định: từ 0° đến 180° (đủ để tái tạo ảnh)
            self.theta = np.linspace(0.0, 180.0, num_angles, endpoint=False)
        else:
            if len(theta) != num_angles:
                raise ValueError(
                    f"theta length {len(theta)} != num_angles {num_angles}"
                )
            self.theta = theta

        logger.info(f"CTReconstructor initialized:")
        logger.info(f"  Sinogram shape: {sinogram.shape}")
        logger.info(
            f"  Angles: {len(self.theta)} from {self.theta[0]:.1f}° to {self.theta[-1]:.1f}°"
        )
        logger.info(f"  Detectors: {num_detectors}")

    def reconstruct_fbp(
        self,
        filter_name: str = "ramp",
        interpolation: str = "linear",
        circle: bool = True,
    ) -> np.ndarray:
        """
        Filtered Back Projection (FBP) reconstruction.

        FBP là phương pháp phổ biến nhất trong CT reconstruction:
        1. Filter sinogram (loại bỏ artifacts)
        2. Back-project (chiếu ngược) lên image space

        Args:
            filter_name: Loại filter để áp dụng
                        - 'ramp': Sharp, high frequency (mặc định)
                        - 'shepp-logan': Smooth hơn
                        - 'cosine': Medium smoothing
                        - 'hamming': Smooth, low noise
                        - 'hann': Very smooth

            interpolation: Phương pháp nội suy
                          - 'linear': Nhanh, đủ tốt
                          - 'cubic': Chậm hơn, mượt hơn

            circle: Nếu True, mask vùng ngoài hình tròn
                    (vì CT scanner quay tròn)

        Returns:
            Reconstructed image (2D array)

        Example:
            >>> # Sharp reconstruction
            >>> image = reconstructor.reconstruct_fbp(filter_name='ramp')
            >>>
            >>> # Smooth reconstruction (ít noise hơn)
            >>> image = reconstructor.reconstruct_fbp(filter_name='hamming')
        """
        logger.info(f"Running FBP reconstruction with filter: {filter_name}")

        try:
            # iradon expects (num_detectors, num_angles)
            # but our sinogram is (num_angles, num_detectors), so transpose
            sinogram_for_iradon = self.sinogram.T

            # Use scikit-image's iradon (inverse radon transform)
            reconstruction = iradon(
                sinogram_for_iradon,
                theta=self.theta,
                filter_name=filter_name,
                interpolation=interpolation,
                circle=circle,
            )

            logger.info(f"✅ FBP reconstruction complete: {reconstruction.shape}")
            return reconstruction

        except Exception as e:
            logger.error(f"❌ FBP reconstruction failed: {e}")
            raise

    def reconstruct_sart(
        self,
        iterations: int = 1,
        relaxation: float = 0.15,
        image_size: Optional[int] = None,
    ) -> np.ndarray:
        """
        SART (Simultaneous Algebraic Reconstruction Technique).

        SART là iterative method:
        1. Bắt đầu với ảnh ban đầu (thường là zeros)
        2. Lặp lại nhiều lần:
           - Tính projection từ ảnh hiện tại
           - So sánh với sinogram thực
           - Update ảnh dựa trên error
        3. Sau nhiều iterations → ảnh càng chính xác

        Args:
            iterations: Số lần lặp
                       - Nhiều hơn = chính xác hơn nhưng chậm hơn
                       - Thường dùng 1-10 iterations

            relaxation: Relaxation factor (0 < r < 1)
                       - Nhỏ hơn = hội tụ chậm nhưng ổn định
                       - Lớn hơn = hội tụ nhanh nhưng có thể diverge

            image_size: Kích thước ảnh output
                       Nếu None, tự động từ num_detectors

        Returns:
            Reconstructed image (2D array)

        Example:
            >>> # Quick reconstruction (1 iteration)
            >>> image = reconstructor.reconstruct_sart(iterations=1)
            >>>
            >>> # Better quality (more iterations)
            >>> image = reconstructor.reconstruct_sart(iterations=10)
        """
        logger.info(f"Running SART reconstruction: {iterations} iterations")

        if image_size is None:
            # Auto-determine image size from sinogram
            image_size = self.sinogram.shape[1]

        try:
            # iradon_sart expects (num_detectors, num_angles)
            sinogram_for_sart = self.sinogram.T

            # Check for valid sinogram values
            if np.any(np.isnan(sinogram_for_sart)) or np.any(
                np.isinf(sinogram_for_sart)
            ):
                raise ValueError("Sinogram contains NaN or Inf values")

            # Initialize with a better starting image (small positive values)
            # instead of zeros to avoid numerical issues
            initial_image = np.ones((image_size, image_size)) * 0.01

            # Use scikit-image's iradon_sart with better parameters
            reconstruction = iradon_sart(
                sinogram_for_sart,
                theta=self.theta,
                image=initial_image,  # Start with small positive values
                relaxation=relaxation,
                clip=(0, None),  # Non-negative constraint
            )

            # Check for NaN in first iteration
            if np.any(np.isnan(reconstruction)):
                logger.warning("SART produced NaN values, falling back to FBP")
                return self.reconstruct_fbp()

            # Run multiple iterations
            for i in range(iterations - 1):
                reconstruction = iradon_sart(
                    sinogram_for_sart,
                    theta=self.theta,
                    image=reconstruction,  # Use previous result
                    relaxation=relaxation,
                    clip=(0, None),
                )

                # Check for NaN after each iteration
                if np.any(np.isnan(reconstruction)):
                    logger.warning(f"NaN detected at iteration {i+2}, stopping early")
                    # Return last valid reconstruction
                    return self.reconstruct_fbp()

                logger.debug(f"  Iteration {i+2}/{iterations}")

            logger.info(f"✅ SART reconstruction complete: {reconstruction.shape}")
            return reconstruction

        except Exception as e:
            logger.error(f"❌ SART reconstruction failed: {e}")
            logger.info("Falling back to FBP reconstruction")
            return self.reconstruct_fbp()

    def compare_methods(
        self, filter_name: str = "ramp", sart_iterations: int = 1
    ) -> Dict[str, np.ndarray]:
        """
        So sánh FBP và SART side-by-side.

        Args:
            filter_name: Filter cho FBP
            sart_iterations: Số iterations cho SART

        Returns:
            Dictionary với keys: 'fbp', 'sart', 'difference'

        Example:
            >>> results = reconstructor.compare_methods()
            >>> plt.subplot(131)
            >>> plt.imshow(results['fbp'], cmap='gray')
            >>> plt.title('FBP')
            >>> plt.subplot(132)
            >>> plt.imshow(results['sart'], cmap='gray')
            >>> plt.title('SART')
            >>> plt.subplot(133)
            >>> plt.imshow(results['difference'], cmap='seismic')
            >>> plt.title('Difference')
        """
        logger.info("Comparing FBP vs SART reconstruction methods")

        # Reconstruct with both methods
        fbp_result = self.reconstruct_fbp(filter_name=filter_name)
        sart_result = self.reconstruct_sart(iterations=sart_iterations)

        # Compute difference
        difference = fbp_result - sart_result

        results = {"fbp": fbp_result, "sart": sart_result, "difference": difference}

        # Log statistics
        logger.info(f"FBP  - Range: [{fbp_result.min():.3f}, {fbp_result.max():.3f}]")
        logger.info(f"SART - Range: [{sart_result.min():.3f}, {sart_result.max():.3f}]")
        logger.info(f"Difference - Mean: {np.abs(difference).mean():.3f}")

        return results

    def visualize_sinogram(
        self, figsize: Tuple[int, int] = (12, 4), save_path: Optional[str] = None
    ) -> None:
        """
        Visualize sinogram và reconstructed images.

        Hiển thị:
        1. Sinogram (projection data)
        2. FBP reconstruction
        3. SART reconstruction

        Args:
            figsize: Figure size (width, height)
            save_path: Nếu provided, save figure to file

        Example:
            >>> reconstructor.visualize_sinogram()
            >>> reconstructor.visualize_sinogram(save_path='ct_reconstruction.png')
        """
        fig, axes = plt.subplots(1, 3, figsize=figsize)

        # Sinogram
        axes[0].imshow(self.sinogram, cmap="gray", aspect="auto")
        axes[0].set_title("Sinogram (Projection Data)")
        axes[0].set_xlabel("Detector Position")
        axes[0].set_ylabel("Projection Angle")

        # FBP
        fbp = self.reconstruct_fbp()
        axes[1].imshow(fbp, cmap="gray")
        axes[1].set_title("FBP Reconstruction")
        axes[1].axis("off")

        # SART
        sart = self.reconstruct_sart(iterations=1)
        axes[2].imshow(sart, cmap="gray")
        axes[2].set_title("SART Reconstruction (1 iter)")
        axes[2].axis("off")

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            logger.info(f"Saved visualization to: {save_path}")

        plt.show()

    def get_quality_metrics(
        self, ground_truth: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """
        Tính quality metrics cho reconstruction.

        Nếu có ground truth (ảnh gốc), tính:
        - MSE (Mean Squared Error)
        - PSNR (Peak Signal-to-Noise Ratio)
        - SSIM (Structural Similarity Index)

        Args:
            ground_truth: Original image (nếu có)

        Returns:
            Dictionary với metrics
        """
        if ground_truth is None:
            logger.warning("No ground truth provided, skipping quality metrics")
            return {}

        fbp = self.reconstruct_fbp()

        # MSE
        mse = np.mean((fbp - ground_truth) ** 2)

        # PSNR (higher is better)
        max_pixel = np.max(ground_truth)
        psnr = 10 * np.log10((max_pixel**2) / mse) if mse > 0 else float("inf")

        metrics = {
            "mse": float(mse),
            "psnr": float(psnr),
        }

        logger.info(f"Quality Metrics: MSE={mse:.3f}, PSNR={psnr:.2f} dB")

        return metrics


# Convenience function
def reconstruct_ct(sinogram: np.ndarray, method: str = "fbp", **kwargs) -> np.ndarray:
    """
    Convenience function cho quick CT reconstruction.

    Args:
        sinogram: Projection data
        method: 'fbp' hoặc 'sart'
        **kwargs: Additional parameters cho method

    Returns:
        Reconstructed image

    Example:
        >>> image = reconstruct_ct(sinogram, method='fbp', filter_name='ramp')
        >>> image = reconstruct_ct(sinogram, method='sart', iterations=5)
    """
    reconstructor = CTReconstructor(sinogram)

    if method.lower() == "fbp":
        return reconstructor.reconstruct_fbp(**kwargs)
    elif method.lower() == "sart":
        return reconstructor.reconstruct_sart(**kwargs)
    else:
        raise ValueError(f"Unknown method: {method}. Use 'fbp' or 'sart'")
