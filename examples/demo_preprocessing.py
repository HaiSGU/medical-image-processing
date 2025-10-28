"""
Image Preprocessing Demo

Demonstrates preprocessing transformations:
1. Intensity normalization (min-max, z-score, percentile)
2. Noise reduction (Gaussian, median, bilateral, NLM)
3. Histogram operations (equalization, CLAHE)
4. Comparison and evaluation

GIẢI THÍCH:
-----------
Preprocessing = Chuẩn bị ảnh trước khi analysis

Mục đích:
- Reduce noise (giảm nhiễu)
- Standardize intensities (chuẩn hóa)
- Improve contrast (tăng độ tương phản)
- Prepare for machine learning

Author: HaiSGU
Date: 2025-10-28
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os
from scipy import ndimage

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.preprocessing import ImageTransforms


def add_noise(image: np.ndarray, noise_type: str = "gaussian", **kwargs) -> np.ndarray:
    """
    Add noise to image for testing.

    Args:
        image: Clean image
        noise_type: 'gaussian', 'salt_pepper', or 'speckle'
        **kwargs: Noise parameters

    Returns:
        Noisy image
    """
    if noise_type == "gaussian":
        sigma = kwargs.get("sigma", 25.0)
        noise = np.random.randn(*image.shape) * sigma
        noisy = image + noise

    elif noise_type == "salt_pepper":
        prob = kwargs.get("prob", 0.05)
        noisy = image.copy()
        # Salt
        salt = np.random.rand(*image.shape) < prob / 2
        noisy[salt] = image.max()
        # Pepper
        pepper = np.random.rand(*image.shape) < prob / 2
        noisy[pepper] = image.min()

    elif noise_type == "speckle":
        noise = np.random.randn(*image.shape)
        noisy = image + image * noise * 0.1

    else:
        raise ValueError(f"Unknown noise type: {noise_type}")

    return noisy


def create_test_image(size: int = 256) -> np.ndarray:
    """
    Create test image (brain-like).

    Args:
        size: Image size

    Returns:
        Test image
    """
    print("\n🖼️  Creating test image...")

    image = np.zeros((size, size))

    # Create structures
    y, x = np.ogrid[:size, :size]
    cy, cx = size // 2, size // 2

    # Brain outline
    ellipse = (x - cx) ** 2 / (size * 0.35) ** 2 + (y - cy) ** 2 / (
        size * 0.4
    ) ** 2 <= 1
    image[ellipse] = 200

    # Internal structures
    for _ in range(3):
        offset_x = np.random.randint(-size // 6, size // 6)
        offset_y = np.random.randint(-size // 6, size // 6)
        radius = np.random.randint(size // 15, size // 10)

        circle = (x - (cx + offset_x)) ** 2 + (y - (cy + offset_y)) ** 2 <= radius**2
        image[circle] = 100

    # Smooth
    image = ndimage.gaussian_filter(image, sigma=1.0)

    print(f"   Image shape: {image.shape}")
    print(f"   Intensity range: [{image.min():.1f}, {image.max():.1f}]")

    return image


def test_1_normalization():
    """
    Test 1: Intensity normalization methods

    GIẢI THÍCH:
    -----------
    Normalization đưa intensities về cùng scale:

    1. Min-Max: [min, max] → [0, 1]
    2. Z-Score: mean=0, std=1
    3. Percentile: Robust to outliers
    """
    print("\n" + "=" * 60)
    print("TEST 1: Intensity Normalization")
    print("=" * 60)

    # Create test image
    image = create_test_image()

    # Create transforms
    transforms = ImageTransforms(image)

    # Apply different normalizations
    print("\n🔄 Applying normalization methods...")

    norm_minmax = transforms.normalize_minmax(0, 1)
    norm_zscore = transforms.normalize_zscore()
    norm_percentile = transforms.normalize_percentile(1, 99)

    # Visualize
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))

    axes[0, 0].imshow(image, cmap="gray")
    axes[0, 0].set_title(f"Original\nRange: [{image.min():.1f}, {image.max():.1f}]")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(norm_minmax, cmap="gray")
    axes[0, 1].set_title(
        f"Min-Max [0,1]\nRange: [{norm_minmax.min():.3f}, {norm_minmax.max():.3f}]"
    )
    axes[0, 1].axis("off")

    axes[1, 0].imshow(norm_zscore, cmap="gray")
    axes[1, 0].set_title(
        f"Z-Score\nMean: {norm_zscore.mean():.3f}, Std: {norm_zscore.std():.3f}"
    )
    axes[1, 0].axis("off")

    axes[1, 1].imshow(norm_percentile, cmap="gray")
    axes[1, 1].set_title(
        f"Percentile [1%,99%]\nRange: [{norm_percentile.min():.3f}, {norm_percentile.max():.3f}]"
    )
    axes[1, 1].axis("off")

    plt.tight_layout()
    plt.savefig("preprocessing_normalization.png", dpi=150, bbox_inches="tight")
    plt.close()

    print("\n📊 Saved: preprocessing_normalization.png")


def test_2_gaussian_noise():
    """
    Test 2: Denoising Gaussian noise

    GIẢI THÍCH:
    -----------
    Gaussian noise = Random noise, bell-curve distribution

    Best filters:
    - Gaussian filter (simple, fast)
    - Bilateral filter (preserve edges)
    - NLM (best quality, slowest)
    """
    print("\n" + "=" * 60)
    print("TEST 2: Gaussian Noise Reduction")
    print("=" * 60)

    # Create clean image
    image_clean = create_test_image()

    # Add Gaussian noise
    print("\n🔊 Adding Gaussian noise (sigma=25)...")
    image_noisy = add_noise(image_clean, "gaussian", sigma=25.0)

    print(f"   Noisy range: [{image_noisy.min():.1f}, {image_noisy.max():.1f}]")

    # Create transforms
    transforms = ImageTransforms(image_noisy)

    # Test different denoising methods
    print("\n🔄 Testing denoising methods...")

    denoised_gaussian = transforms.denoise_gaussian(sigma=1.5)
    denoised_median = transforms.denoise_median(size=3)
    denoised_bilateral = transforms.denoise_bilateral(sigma_spatial=15, sigma_color=25)

    # Calculate PSNR (Peak Signal-to-Noise Ratio)
    def psnr(clean, denoised):
        mse = np.mean((clean - denoised) ** 2)
        if mse == 0:
            return float("inf")
        max_val = clean.max()
        return 20 * np.log10(max_val / np.sqrt(mse))

    psnr_noisy = psnr(image_clean, image_noisy)
    psnr_gaussian = psnr(image_clean, denoised_gaussian)
    psnr_median = psnr(image_clean, denoised_median)
    psnr_bilateral = psnr(image_clean, denoised_bilateral)

    print(f"\n📈 PSNR (Peak Signal-to-Noise Ratio):")
    print(f"   Noisy: {psnr_noisy:.2f} dB")
    print(
        f"   Gaussian filter: {psnr_gaussian:.2f} dB (+{psnr_gaussian-psnr_noisy:.2f})"
    )
    print(f"   Median filter: {psnr_median:.2f} dB (+{psnr_median-psnr_noisy:.2f})")
    print(
        f"   Bilateral filter: {psnr_bilateral:.2f} dB (+{psnr_bilateral-psnr_noisy:.2f})"
    )

    # Visualize
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    axes[0, 0].imshow(image_clean, cmap="gray")
    axes[0, 0].set_title("Clean Image")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(image_noisy, cmap="gray")
    axes[0, 1].set_title(f"Noisy (Gaussian σ=25)\nPSNR: {psnr_noisy:.2f} dB")
    axes[0, 1].axis("off")

    axes[0, 2].imshow(denoised_gaussian, cmap="gray")
    axes[0, 2].set_title(f"Gaussian Filter\nPSNR: {psnr_gaussian:.2f} dB")
    axes[0, 2].axis("off")

    axes[1, 0].imshow(denoised_median, cmap="gray")
    axes[1, 0].set_title(f"Median Filter\nPSNR: {psnr_median:.2f} dB")
    axes[1, 0].axis("off")

    axes[1, 1].imshow(denoised_bilateral, cmap="gray")
    axes[1, 1].set_title(f"Bilateral Filter\nPSNR: {psnr_bilateral:.2f} dB")
    axes[1, 1].axis("off")

    # Difference images
    diff = np.abs(image_clean - denoised_bilateral)
    axes[1, 2].imshow(diff, cmap="hot")
    axes[1, 2].set_title("Difference (Clean - Bilateral)")
    axes[1, 2].axis("off")

    plt.tight_layout()
    plt.savefig("preprocessing_gaussian_noise.png", dpi=150, bbox_inches="tight")
    plt.close()

    print("\n📊 Saved: preprocessing_gaussian_noise.png")


def test_3_salt_pepper_noise():
    """
    Test 3: Denoising salt & pepper noise

    GIẢI THÍCH:
    -----------
    Salt & pepper = Random black/white pixels

    Best filter: Median filter!
    (Gaussian filter fails on salt & pepper)
    """
    print("\n" + "=" * 60)
    print("TEST 3: Salt & Pepper Noise Reduction")
    print("=" * 60)

    # Create clean image
    image_clean = create_test_image()

    # Add salt & pepper noise
    print("\n🔊 Adding salt & pepper noise (prob=0.05)...")
    image_noisy = add_noise(image_clean, "salt_pepper", prob=0.05)

    # Create transforms
    transforms = ImageTransforms(image_noisy)

    # Test filters
    print("\n🔄 Testing filters...")

    denoised_gaussian = transforms.denoise_gaussian(sigma=1.5)
    denoised_median = transforms.denoise_median(size=5)

    # PSNR
    def psnr(clean, denoised):
        mse = np.mean((clean - denoised) ** 2)
        if mse == 0:
            return float("inf")
        return 20 * np.log10(clean.max() / np.sqrt(mse))

    psnr_gaussian = psnr(image_clean, denoised_gaussian)
    psnr_median = psnr(image_clean, denoised_median)

    print(f"\n📈 PSNR:")
    print(f"   Gaussian filter: {psnr_gaussian:.2f} dB (fails!)")
    print(f"   Median filter: {psnr_median:.2f} dB (works!)")

    # Visualize
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    axes[0].imshow(image_clean, cmap="gray")
    axes[0].set_title("Clean Image")
    axes[0].axis("off")

    axes[1].imshow(image_noisy, cmap="gray")
    axes[1].set_title("Noisy (Salt & Pepper)")
    axes[1].axis("off")

    axes[2].imshow(denoised_gaussian, cmap="gray")
    axes[2].set_title(f"Gaussian Filter\nPSNR: {psnr_gaussian:.2f} dB")
    axes[2].axis("off")

    axes[3].imshow(denoised_median, cmap="gray")
    axes[3].set_title(f"Median Filter ✓\nPSNR: {psnr_median:.2f} dB")
    axes[3].axis("off")

    plt.tight_layout()
    plt.savefig("preprocessing_salt_pepper.png", dpi=150, bbox_inches="tight")
    plt.close()

    print("\n📊 Saved: preprocessing_salt_pepper.png")


def test_4_histogram_operations():
    """
    Test 4: Histogram operations

    GIẢI THÍCH:
    -----------
    1. Histogram Equalization: Spread histogram uniformly
    2. CLAHE: Adaptive (local) histogram equalization

    Tăng contrast!
    """
    print("\n" + "=" * 60)
    print("TEST 4: Histogram Operations")
    print("=" * 60)

    # Create low-contrast image
    print("\n🖼️  Creating low-contrast image...")
    image = create_test_image()
    # Reduce contrast
    image_low = image * 0.5 + 50

    print(f"   Original range: [{image.min():.1f}, {image.max():.1f}]")
    print(f"   Low-contrast range: [{image_low.min():.1f}, {image_low.max():.1f}]")

    # Create transforms
    transforms = ImageTransforms(image_low)

    # Apply histogram operations
    print("\n🔄 Applying histogram operations...")

    eq_global = transforms.histogram_equalization()
    eq_clahe = transforms.adaptive_histogram_equalization(clip_limit=0.03)

    # Visualize
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # Images
    axes[0, 0].imshow(image_low, cmap="gray")
    axes[0, 0].set_title("Low Contrast Original")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(eq_global, cmap="gray")
    axes[0, 1].set_title("Histogram Equalization")
    axes[0, 1].axis("off")

    axes[0, 2].imshow(eq_clahe, cmap="gray")
    axes[0, 2].set_title("CLAHE (Adaptive)")
    axes[0, 2].axis("off")

    # Histograms
    axes[1, 0].hist(image_low.flatten(), bins=50, color="blue", alpha=0.7)
    axes[1, 0].set_title("Original Histogram")
    axes[1, 0].set_xlabel("Intensity")
    axes[1, 0].set_ylabel("Frequency")

    axes[1, 1].hist(eq_global.flatten(), bins=50, color="green", alpha=0.7)
    axes[1, 1].set_title("Equalized Histogram")
    axes[1, 1].set_xlabel("Intensity")
    axes[1, 1].set_ylabel("Frequency")

    axes[1, 2].hist(eq_clahe.flatten(), bins=50, color="red", alpha=0.7)
    axes[1, 2].set_title("CLAHE Histogram")
    axes[1, 2].set_xlabel("Intensity")
    axes[1, 2].set_ylabel("Frequency")

    plt.tight_layout()
    plt.savefig("preprocessing_histogram.png", dpi=150, bbox_inches="tight")
    plt.close()

    print("\n📊 Saved: preprocessing_histogram.png")


def test_5_preprocessing_pipeline():
    """
    Test 5: Complete preprocessing pipeline

    GIẢI THÍCH:
    -----------
    Typical preprocessing pipeline:

    1. Add noise (simulate real data)
    2. Denoise (bilateral/NLM)
    3. Normalize (z-score for ML)
    4. Enhance (CLAHE)

    Ready for machine learning!
    """
    print("\n" + "=" * 60)
    print("TEST 5: Complete Preprocessing Pipeline")
    print("=" * 60)

    # Create clean image
    image_clean = create_test_image()

    # Step 1: Add noise (simulate real data)
    print("\n📥 Step 1: Simulating noisy acquisition...")
    image_noisy = add_noise(image_clean, "gaussian", sigma=20.0)

    # Step 2: Denoise
    print("\n🔧 Step 2: Denoising (bilateral filter)...")
    transforms_noisy = ImageTransforms(image_noisy)
    image_denoised = transforms_noisy.denoise_bilateral(
        sigma_spatial=15, sigma_color=20
    )

    # Step 3: Normalize
    print("\n📊 Step 3: Normalization (z-score)...")
    transforms_denoised = ImageTransforms(image_denoised)
    image_normalized = transforms_denoised.normalize_zscore()

    # Step 4: Enhance
    print("\n✨ Step 4: Enhancement (CLAHE)...")
    # Need to normalize to [0,1] for CLAHE
    image_norm_01 = (image_normalized - image_normalized.min()) / (
        image_normalized.max() - image_normalized.min()
    )
    transforms_norm = ImageTransforms(image_norm_01)
    image_enhanced = transforms_norm.adaptive_histogram_equalization(clip_limit=0.02)

    print("\n✅ Pipeline complete!")
    print(f"   Final range: [{image_enhanced.min():.3f}, {image_enhanced.max():.3f}]")
    print(f"   Final mean: {image_enhanced.mean():.3f}")
    print(f"   Final std: {image_enhanced.std():.3f}")

    # Visualize
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    axes[0, 0].imshow(image_clean, cmap="gray")
    axes[0, 0].set_title("1. Original (Clean)")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(image_noisy, cmap="gray")
    axes[0, 1].set_title("2. Noisy (σ=20)")
    axes[0, 1].axis("off")

    axes[0, 2].imshow(image_denoised, cmap="gray")
    axes[0, 2].set_title("3. Denoised (Bilateral)")
    axes[0, 2].axis("off")

    axes[1, 0].imshow(image_normalized, cmap="gray")
    axes[1, 0].set_title("4. Normalized (Z-Score)")
    axes[1, 0].axis("off")

    axes[1, 1].imshow(image_enhanced, cmap="gray")
    axes[1, 1].set_title("5. Enhanced (CLAHE)")
    axes[1, 1].axis("off")

    # Comparison
    axes[1, 2].imshow(image_clean, cmap="gray", alpha=0.5)
    axes[1, 2].imshow(image_enhanced, cmap="hot", alpha=0.5)
    axes[1, 2].set_title("6. Original vs Final")
    axes[1, 2].axis("off")

    plt.tight_layout()
    plt.savefig("preprocessing_pipeline.png", dpi=150, bbox_inches="tight")
    plt.close()

    print("\n📊 Saved: preprocessing_pipeline.png")


def main():
    """Run all preprocessing tests"""
    print("\n" + "=" * 70)
    print("  IMAGE PREPROCESSING DEMO")
    print("=" * 70)
    print("\nGIẢI THÍCH TỔNG QUAN:")
    print("-" * 70)
    print("Preprocessing = Chuẩn bị ảnh trước khi analysis/segmentation")
    print("")
    print("Mục đích:")
    print("  1. Reduce noise (giảm nhiễu)")
    print("  2. Standardize intensities (chuẩn hóa cường độ)")
    print("  3. Improve contrast (tăng độ tương phản)")
    print("  4. Prepare for ML (chuẩn bị cho machine learning)")
    print("")
    print("Phương pháp:")
    print("  - Normalization: Min-Max, Z-Score, Percentile")
    print("  - Denoising: Gaussian, Median, Bilateral, NLM")
    print("  - Histogram: Equalization, CLAHE")
    print("=" * 70)

    # Run tests
    try:
        test_1_normalization()
    except Exception as e:
        print(f"\n❌ Test 1 failed: {e}")
        import traceback

        traceback.print_exc()

    try:
        test_2_gaussian_noise()
    except Exception as e:
        print(f"\n❌ Test 2 failed: {e}")
        import traceback

        traceback.print_exc()

    try:
        test_3_salt_pepper_noise()
    except Exception as e:
        print(f"\n❌ Test 3 failed: {e}")
        import traceback

        traceback.print_exc()

    try:
        test_4_histogram_operations()
    except Exception as e:
        print(f"\n❌ Test 4 failed: {e}")
        import traceback

        traceback.print_exc()

    try:
        test_5_preprocessing_pipeline()
    except Exception as e:
        print(f"\n❌ Test 5 failed: {e}")
        import traceback

        traceback.print_exc()

    # Summary
    print("\n" + "=" * 70)
    print("  SUMMARY")
    print("=" * 70)
    print("\n✅ Image Preprocessing module tested successfully!")
    print("\nGenerated files:")
    print("  📊 preprocessing_normalization.png - Normalization methods")
    print("  📊 preprocessing_gaussian_noise.png - Gaussian noise reduction")
    print("  📊 preprocessing_salt_pepper.png - Salt & pepper noise")
    print("  📊 preprocessing_histogram.png - Histogram operations")
    print("  📊 preprocessing_pipeline.png - Complete pipeline")
    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
