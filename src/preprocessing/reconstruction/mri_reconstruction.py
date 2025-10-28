"""
MRI Reconstruction Module

Implements MRI image reconstruction from k-space data using:
- Forward FFT: Image → K-space
- Inverse FFT: K-space → Image
- Magnitude and Phase extraction
- K-space visualization
- Partial Fourier reconstruction

GIẢI THÍCH:
-----------
MRI reconstruction là quá trình chuyển đổi K-space data sang ảnh.

K-space là gì?
- K-space = Frequency domain (miền tần số)
- Center của K-space = Contrast information (độ sáng tối)
- Edges của K-space = Detail information (chi tiết, cạnh)

Quy trình:
1. MRI Scanner thu thập K-space data (complex numbers)
2. Inverse FFT 2D: K-space → Complex Image
3. Extract Magnitude (|complex|) và Phase (angle)

Author: HaiSGU
Date: 2025-10-27
"""

import numpy as np
import logging
from typing import Tuple, Optional, Dict
import matplotlib.pyplot as plt
from scipy import ndimage

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MRIReconstructor:
    """
    MRI Image Reconstruction from K-space data.

    MRI reconstruction chuyển đổi K-space (frequency domain)
    thành image (spatial domain) bằng FFT.

    Attributes:
        kspace: 2D complex array (K-space data)

    Examples:
        >>> # Load K-space data
        >>> kspace = np.load('kspace.npy')
        >>>
        >>> # Create reconstructor
        >>> reconstructor = MRIReconstructor(kspace)
        >>>
        >>> # Reconstruct magnitude image
        >>> magnitude = reconstructor.reconstruct_magnitude()
        >>>
        >>> # Reconstruct phase image
        >>> phase = reconstructor.reconstruct_phase()
        >>>
        >>> # Visualize K-space
        >>> reconstructor.visualize_kspace()
    """

    def __init__(self, kspace: np.ndarray):
        """
        Initialize MRI Reconstructor.

        Args:
            kspace: 2D complex array of K-space data
                   Shape: (height, width)
                   - Complex numbers (có phần thực và ảo)
                   - Center = low frequencies (contrast)
                   - Edges = high frequencies (detail)

        Example:
            >>> kspace = np.random.randn(256, 256) + 1j * np.random.randn(256, 256)
            >>> reconstructor = MRIReconstructor(kspace)
        """
        if not np.iscomplexobj(kspace):
            logger.warning("K-space data is not complex, converting to complex")
            kspace = kspace.astype(np.complex128)

        self.kspace = kspace

        logger.info(f"MRIReconstructor initialized:")
        logger.info(f"  K-space shape: {kspace.shape}")
        logger.info(f"  Data type: {kspace.dtype}")
        logger.info(f"  Is complex: {np.iscomplexobj(kspace)}")

    def kspace_to_image(self, kspace: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Convert K-space to image using Inverse FFT 2D.

        GIẢI THÍCH:
        -----------
        K-space (frequency domain) → Image (spatial domain)

        Công thức:
        Image = IFFT2D(K-space)

        FFT = Fast Fourier Transform (biến đổi Fourier nhanh)
        IFFT = Inverse FFT (FFT ngược)

        Args:
            kspace: K-space data (nếu None, dùng self.kspace)

        Returns:
            Complex image (2D array)

        Example:
            >>> image_complex = reconstructor.kspace_to_image()
            >>> print(image_complex.dtype)  # complex128
        """
        if kspace is None:
            kspace = self.kspace

        logger.debug("Converting K-space to image (IFFT2D)")

        # Inverse FFT 2D
        # fftshift: Đưa zero frequency về center
        # ifft2: Inverse FFT 2 chiều
        # ifftshift: Đưa zero frequency về corner (chuẩn cho image)
        image_complex = np.fft.ifftshift(np.fft.ifft2(np.fft.ifftshift(kspace)))

        return image_complex

    def image_to_kspace(self, image: np.ndarray) -> np.ndarray:
        """
        Convert image to K-space using Forward FFT 2D.

        GIẢI THÍCH:
        -----------
        Image (spatial domain) → K-space (frequency domain)

        Công thức:
        K-space = FFT2D(Image)

        Args:
            image: 2D image array (có thể real hoặc complex)

        Returns:
            K-space data (complex array)

        Example:
            >>> kspace = reconstructor.image_to_kspace(image)
            >>> print(kspace.dtype)  # complex128
        """
        logger.debug("Converting image to K-space (FFT2D)")

        # Forward FFT 2D
        kspace = np.fft.fftshift(np.fft.fft2(np.fft.fftshift(image)))

        return kspace

    def reconstruct_magnitude(self) -> np.ndarray:
        """
        Reconstruct magnitude image from K-space.

        GIẢI THÍCH:
        -----------
        Magnitude = |Complex Image| = sqrt(real^2 + imag^2)

        Đây là ảnh MRI thông thường mà chúng ta thấy:
        - Bright = signal mạnh
        - Dark = signal yếu

        Returns:
            Magnitude image (real 2D array)

        Example:
            >>> magnitude = reconstructor.reconstruct_magnitude()
            >>> plt.imshow(magnitude, cmap='gray')
        """
        logger.info("Reconstructing magnitude image")

        # K-space → Complex image
        image_complex = self.kspace_to_image()

        # Magnitude = |complex|
        magnitude = np.abs(image_complex)

        logger.info(
            f"  Magnitude range: [{magnitude.min():.3f}, {magnitude.max():.3f}]"
        )

        return magnitude

    def reconstruct_phase(self) -> np.ndarray:
        """
        Reconstruct phase image from K-space.

        GIẢI THÍCH:
        -----------
        Phase = angle(Complex Image) = arctan(imag / real)

        Phase image chứa thông tin về:
        - Sự khác biệt từ tính (magnetic susceptibility)
        - Flow (máu chảy)
        - Hữu ích cho advanced MRI techniques

        Returns:
            Phase image (real 2D array, range: -π to π)

        Example:
            >>> phase = reconstructor.reconstruct_phase()
            >>> plt.imshow(phase, cmap='hsv')  # Dùng colormap 'hsv' cho phase
        """
        logger.info("Reconstructing phase image")

        # K-space → Complex image
        image_complex = self.kspace_to_image()

        # Phase = angle(complex)
        phase = np.angle(image_complex)

        logger.info(f"  Phase range: [{phase.min():.3f}, {phase.max():.3f}] rad")

        return phase

    def reconstruct_both(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Reconstruct both magnitude and phase images.

        Returns:
            Tuple of (magnitude, phase)

        Example:
            >>> magnitude, phase = reconstructor.reconstruct_both()
        """
        logger.info("Reconstructing magnitude and phase images")

        image_complex = self.kspace_to_image()

        magnitude = np.abs(image_complex)
        phase = np.angle(image_complex)

        return magnitude, phase

    def partial_fourier_reconstruct(
        self, factor: float = 0.75, method: str = "zero-fill"
    ) -> np.ndarray:
        """
        Partial Fourier reconstruction.

        GIẢI THÍCH:
        -----------
        Partial Fourier = Thu thập chỉ một phần K-space (để scan nhanh hơn)

        Ví dụ: Chỉ thu thập 75% K-space (bỏ 25% phần trên)
        → Reconstruction từ dữ liệu không đầy đủ

        Methods:
        - 'zero-fill': Điền zeros vào phần thiếu (đơn giản)
        - 'homodyne': Exploit conjugate symmetry (phức tạp hơn)

        Args:
            factor: Tỷ lệ K-space thu thập (0.5 - 1.0)
                   0.75 = 75% K-space
            method: Reconstruction method

        Returns:
            Magnitude image from partial K-space

        Example:
            >>> # Reconstruct from 75% K-space
            >>> image = reconstructor.partial_fourier_reconstruct(factor=0.75)
        """
        logger.info(f"Partial Fourier reconstruction: {factor*100:.0f}% K-space")

        # Create partial K-space
        kspace_partial = self.kspace.copy()
        height, width = kspace_partial.shape

        # Zero-fill top portion
        lines_to_remove = int(height * (1 - factor))
        kspace_partial[:lines_to_remove, :] = 0

        logger.debug(f"  Zeroed {lines_to_remove}/{height} lines")

        if method == "zero-fill":
            # Simple zero-filling
            image_complex = self.kspace_to_image(kspace_partial)
            magnitude = np.abs(image_complex)

        elif method == "homodyne":
            # Homodyne reconstruction (conjugate symmetry)
            # More advanced, better quality
            logger.warning("Homodyne method not implemented, using zero-fill")
            image_complex = self.kspace_to_image(kspace_partial)
            magnitude = np.abs(image_complex)

        else:
            raise ValueError(f"Unknown method: {method}")

        return magnitude

    def get_kspace_center(self, crop_size: int = 64) -> np.ndarray:
        """
        Extract center region of K-space.

        GIẢI THÍCH:
        -----------
        Center của K-space chứa thông tin contrast chính.
        Có thể dùng để quick preview với data size nhỏ.

        Args:
            crop_size: Size of center crop

        Returns:
            Center region of K-space
        """
        h, w = self.kspace.shape
        h_start = (h - crop_size) // 2
        w_start = (w - crop_size) // 2

        center = self.kspace[
            h_start : h_start + crop_size, w_start : w_start + crop_size
        ]

        return center

    def visualize_kspace(
        self,
        log_scale: bool = True,
        figsize: Tuple[int, int] = (15, 5),
        save_path: Optional[str] = None,
    ) -> None:
        """
        Visualize K-space and reconstructed images.

        Hiển thị:
        1. K-space magnitude (log scale)
        2. Magnitude image
        3. Phase image

        Args:
            log_scale: Use log scale for K-space (easier to see)
            figsize: Figure size
            save_path: Path to save figure

        Example:
            >>> reconstructor.visualize_kspace()
            >>> reconstructor.visualize_kspace(save_path='mri_recon.png')
        """
        fig, axes = plt.subplots(1, 3, figsize=figsize)

        # K-space magnitude
        kspace_mag = np.abs(self.kspace)
        if log_scale:
            kspace_mag = np.log(kspace_mag + 1)  # +1 để tránh log(0)

        axes[0].imshow(kspace_mag, cmap="gray")
        title = "K-space (log scale)" if log_scale else "K-space"
        axes[0].set_title(title)
        axes[0].axis("off")

        # Magnitude image
        magnitude = self.reconstruct_magnitude()
        axes[1].imshow(magnitude, cmap="gray")
        axes[1].set_title("Magnitude Image")
        axes[1].axis("off")

        # Phase image
        phase = self.reconstruct_phase()
        axes[2].imshow(phase, cmap="hsv", vmin=-np.pi, vmax=np.pi)
        axes[2].set_title("Phase Image")
        axes[2].axis("off")

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            logger.info(f"Saved visualization to: {save_path}")

        plt.show()

    def compare_partial_fourier(
        self,
        factors: list = [1.0, 0.75, 0.5],
        figsize: Tuple[int, int] = (15, 5),
        save_path: Optional[str] = None,
    ) -> None:
        """
        Compare full vs partial Fourier reconstructions.

        Args:
            factors: List of K-space sampling factors
            figsize: Figure size
            save_path: Path to save figure

        Example:
            >>> reconstructor.compare_partial_fourier([1.0, 0.75, 0.5])
        """
        fig, axes = plt.subplots(1, len(factors), figsize=figsize)

        if len(factors) == 1:
            axes = [axes]

        for ax, factor in zip(axes, factors):
            if factor == 1.0:
                image = self.reconstruct_magnitude()
                title = "Full K-space (100%)"
            else:
                image = self.partial_fourier_reconstruct(factor)
                title = f"Partial Fourier ({factor*100:.0f}%)"

            ax.imshow(image, cmap="gray")
            ax.set_title(title)
            ax.axis("off")

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            logger.info(f"Saved comparison to: {save_path}")

        plt.show()


def create_synthetic_kspace(size: int = 256, phantom_type: str = "brain") -> np.ndarray:
    """
    Create synthetic K-space data for testing.

    Args:
        size: Size of K-space (size x size)
        phantom_type: Type of phantom ('brain', 'circle', 'random')

    Returns:
        Synthetic K-space data (complex array)

    Example:
        >>> kspace = create_synthetic_kspace(256, 'brain')
        >>> reconstructor = MRIReconstructor(kspace)
    """
    logger.info(f"Creating synthetic K-space: {size}x{size}, type={phantom_type}")

    if phantom_type == "brain":
        # Create brain-like phantom
        image = np.zeros((size, size))

        # Main brain region (ellipse)
        y, x = np.ogrid[:size, :size]
        cy, cx = size // 2, size // 2

        # Brain outline
        ellipse = (x - cx) ** 2 / (size * 0.35) ** 2 + (y - cy) ** 2 / (
            size * 0.4
        ) ** 2 <= 1
        image[ellipse] = 0.8

        # Gray matter regions
        for _ in range(3):
            offset_x = np.random.randint(-size // 4, size // 4)
            offset_y = np.random.randint(-size // 4, size // 4)
            radius = np.random.randint(size // 10, size // 6)

            circle = (x - (cx + offset_x)) ** 2 + (
                y - (cy + offset_y)
            ) ** 2 <= radius**2
            image[circle] = 0.5

    elif phantom_type == "circle":
        # Simple circle
        image = np.zeros((size, size))
        y, x = np.ogrid[:size, :size]
        cy, cx = size // 2, size // 2
        circle = (x - cx) ** 2 + (y - cy) ** 2 <= (size * 0.3) ** 2
        image[circle] = 1.0

    elif phantom_type == "random":
        # Random pattern
        image = np.random.rand(size, size)
        image = ndimage.gaussian_filter(image, sigma=5)

    else:
        raise ValueError(f"Unknown phantom type: {phantom_type}")

    # Convert to K-space
    kspace = np.fft.fftshift(np.fft.fft2(np.fft.fftshift(image)))

    return kspace


# Convenience functions
def reconstruct_mri(kspace: np.ndarray, output: str = "magnitude") -> np.ndarray:
    """
    Convenience function for quick MRI reconstruction.

    Args:
        kspace: K-space data
        output: 'magnitude', 'phase', or 'both'

    Returns:
        Reconstructed image(s)

    Example:
        >>> magnitude = reconstruct_mri(kspace, output='magnitude')
        >>> phase = reconstruct_mri(kspace, output='phase')
        >>> mag, phase = reconstruct_mri(kspace, output='both')
    """
    reconstructor = MRIReconstructor(kspace)

    if output == "magnitude":
        return reconstructor.reconstruct_magnitude()
    elif output == "phase":
        return reconstructor.reconstruct_phase()
    elif output == "both":
        return reconstructor.reconstruct_both()
    else:
        raise ValueError(f"Unknown output: {output}")
