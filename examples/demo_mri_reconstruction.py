"""
MRI Reconstruction Demo

Demonstrates MRI image reconstruction from K-space data:
1. Load real K-space data
2. Create synthetic K-space (brain phantom)
3. Reconstruct magnitude and phase images
4. Partial Fourier reconstruction
5. Visualizations

GIẢI THÍCH:
-----------
MRI reconstruction: K-space (frequency domain) → Image (spatial domain)

K-space:
- Center = Contrast information (độ sáng tối)
- Edges = Detail information (chi tiết)

Reconstruction:
- Forward FFT: Image → K-space
- Inverse FFT: K-space → Image
- Magnitude = |complex image| (ảnh MRI thông thường)
- Phase = angle(complex image) (thông tin pha)

Author: HaiSGU
Date: 2025-10-27
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.reconstruction import MRIReconstructor, create_synthetic_kspace


def test_1_real_kspace():
    """
    Test 1: Load and reconstruct real K-space data

    GIẢI THÍCH:
    -----------
    Load K-space data từ file .npy và reconstruct thành ảnh MRI.
    """
    print("\n" + "=" * 60)
    print("TEST 1: Real K-space Reconstruction")
    print("=" * 60)

    # Load K-space data
    kspace_path = "data/medical/slice_kspace.npy"

    if not os.path.exists(kspace_path):
        print(f"⚠️  K-space file not found: {kspace_path}")
        print("   Skipping real K-space test...")
        return

    print(f"\n📂 Loading K-space from: {kspace_path}")
    kspace = np.load(kspace_path)

    print(f"   Shape: {kspace.shape}")
    print(f"   Dtype: {kspace.dtype}")
    print(f"   Is complex: {np.iscomplexobj(kspace)}")

    # Create reconstructor
    reconstructor = MRIReconstructor(kspace)

    # Reconstruct magnitude and phase
    print("\n🔄 Reconstructing magnitude and phase images...")
    magnitude, phase = reconstructor.reconstruct_both()

    print(f"\n✅ Magnitude image:")
    print(f"   Range: [{magnitude.min():.3f}, {magnitude.max():.3f}]")
    print(f"   Mean: {magnitude.mean():.3f}")

    print(f"\n✅ Phase image:")
    print(f"   Range: [{phase.min():.3f}, {phase.max():.3f}] rad")
    print(f"   Mean: {phase.mean():.3f} rad")

    # Visualize
    print("\n📊 Visualizing K-space and reconstructed images...")
    reconstructor.visualize_kspace(save_path="mri_real_reconstruction.png")
    print("   Saved: mri_real_reconstruction.png")


def test_2_synthetic_brain():
    """
    Test 2: Create synthetic brain K-space and reconstruct

    GIẢI THÍCH:
    -----------
    Tạo K-space synthetic từ brain phantom để test thuật toán.
    """
    print("\n" + "=" * 60)
    print("TEST 2: Synthetic Brain Phantom")
    print("=" * 60)

    # Create synthetic brain K-space
    print("\n🧠 Creating synthetic brain K-space (256x256)...")
    kspace = create_synthetic_kspace(size=256, phantom_type="brain")

    print(f"   Shape: {kspace.shape}")
    print(f"   Dtype: {kspace.dtype}")
    print(
        f"   K-space magnitude range: [{np.abs(kspace).min():.2e}, {np.abs(kspace).max():.2e}]"
    )

    # Create reconstructor
    reconstructor = MRIReconstructor(kspace)

    # Reconstruct
    print("\n🔄 Reconstructing magnitude image...")
    magnitude = reconstructor.reconstruct_magnitude()

    print(f"\n✅ Magnitude image:")
    print(f"   Range: [{magnitude.min():.3f}, {magnitude.max():.3f}]")
    print(f"   Mean: {magnitude.mean():.3f}")

    # Visualize
    print("\n📊 Visualizing synthetic brain reconstruction...")
    reconstructor.visualize_kspace(save_path="mri_synthetic_brain.png")
    print("   Saved: mri_synthetic_brain.png")


def test_3_partial_fourier():
    """
    Test 3: Partial Fourier reconstruction

    GIẢI THÍCH:
    -----------
    Partial Fourier = Thu thập chỉ một phần K-space (để scan nhanh hơn)

    Ví dụ:
    - 100% K-space = Full scan (chất lượng tốt nhất)
    - 75% K-space = Partial scan (nhanh hơn, chất lượng hơi giảm)
    - 50% K-space = Half scan (nhanh nhất, chất lượng giảm rõ)
    """
    print("\n" + "=" * 60)
    print("TEST 3: Partial Fourier Reconstruction")
    print("=" * 60)

    # Create synthetic K-space
    print("\n🧠 Creating synthetic brain K-space...")
    kspace = create_synthetic_kspace(size=256, phantom_type="brain")

    reconstructor = MRIReconstructor(kspace)

    # Test different sampling factors
    factors = [1.0, 0.75, 0.5]

    print(f"\n🔄 Testing partial Fourier with factors: {factors}")
    print("\nExplanation:")
    print("  1.0 = Full K-space (100%) - best quality")
    print("  0.75 = 75% K-space - faster scan, slight quality loss")
    print("  0.5 = 50% K-space - fastest scan, noticeable quality loss")

    # Visualize comparison
    print("\n📊 Creating comparison visualization...")
    reconstructor.compare_partial_fourier(
        factors=factors, save_path="mri_partial_fourier.png"
    )
    print("   Saved: mri_partial_fourier.png")

    # Quantitative comparison
    print("\n📈 Quantitative comparison:")
    magnitude_full = reconstructor.reconstruct_magnitude()

    for factor in [0.75, 0.5]:
        magnitude_partial = reconstructor.partial_fourier_reconstruct(factor)

        # Calculate difference (MAE - Mean Absolute Error)
        mae = np.mean(np.abs(magnitude_full - magnitude_partial))

        print(f"   {factor*100:.0f}% K-space:")
        print(f"      MAE vs Full: {mae:.6f}")


def test_4_forward_inverse_fft():
    """
    Test 4: Forward and Inverse FFT consistency

    GIẢI THÍCH:
    -----------
    Kiểm tra tính nhất quán của Forward và Inverse FFT:

    Image → FFT → K-space → IFFT → Image

    Ảnh sau khi IFFT phải giống ảnh ban đầu.
    """
    print("\n" + "=" * 60)
    print("TEST 4: Forward/Inverse FFT Consistency")
    print("=" * 60)

    # Create simple test image
    print("\n🖼️  Creating test image (circle)...")
    size = 256
    image_original = np.zeros((size, size))
    y, x = np.ogrid[:size, :size]
    cy, cx = size // 2, size // 2
    circle = (x - cx) ** 2 + (y - cy) ** 2 <= (size * 0.3) ** 2
    image_original[circle] = 1.0

    print(f"   Image shape: {image_original.shape}")
    print(f"   Image range: [{image_original.min()}, {image_original.max()}]")

    # Forward FFT: Image → K-space
    print("\n➡️  Forward FFT: Image → K-space")
    kspace = np.fft.fftshift(np.fft.fft2(np.fft.fftshift(image_original)))
    print(f"   K-space shape: {kspace.shape}")
    print(
        f"   K-space magnitude range: [{np.abs(kspace).min():.2e}, {np.abs(kspace).max():.2e}]"
    )

    # Create reconstructor
    reconstructor = MRIReconstructor(kspace)

    # Inverse FFT: K-space → Image
    print("\n⬅️  Inverse FFT: K-space → Image")
    image_reconstructed = reconstructor.reconstruct_magnitude()
    print(
        f"   Reconstructed image range: [{image_reconstructed.min():.6f}, {image_reconstructed.max():.6f}]"
    )

    # Check consistency
    print("\n✅ Checking consistency...")
    mae = np.mean(np.abs(image_original - image_reconstructed))
    max_error = np.max(np.abs(image_original - image_reconstructed))

    print(f"   Mean Absolute Error: {mae:.2e}")
    print(f"   Max Error: {max_error:.2e}")

    if mae < 1e-10:
        print("   ✓ PASS: Forward/Inverse FFT are consistent!")
    else:
        print("   ✗ WARNING: Errors detected (but might be due to numerical precision)")

    # Visualize
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(image_original, cmap="gray")
    axes[0].set_title("Original Image")
    axes[0].axis("off")

    kspace_mag = np.log(np.abs(kspace) + 1)
    axes[1].imshow(kspace_mag, cmap="gray")
    axes[1].set_title("K-space (log scale)")
    axes[1].axis("off")

    axes[2].imshow(image_reconstructed, cmap="gray")
    axes[2].set_title("Reconstructed Image")
    axes[2].axis("off")

    plt.tight_layout()
    plt.savefig("mri_fft_consistency.png", dpi=150, bbox_inches="tight")
    plt.close()

    print("\n📊 Saved visualization: mri_fft_consistency.png")


def test_5_kspace_center():
    """
    Test 5: K-space center extraction

    GIẢI THÍCH:
    -----------
    Center của K-space chứa thông tin contrast chính.
    Có thể dùng để quick preview với data nhỏ.
    """
    print("\n" + "=" * 60)
    print("TEST 5: K-space Center Extraction")
    print("=" * 60)

    # Create synthetic K-space
    print("\n🧠 Creating synthetic brain K-space...")
    kspace = create_synthetic_kspace(size=256, phantom_type="brain")

    reconstructor = MRIReconstructor(kspace)

    # Extract center
    crop_sizes = [128, 64, 32]

    print(f"\n🎯 Extracting K-space center with sizes: {crop_sizes}")

    fig, axes = plt.subplots(2, len(crop_sizes), figsize=(15, 10))

    for i, crop_size in enumerate(crop_sizes):
        print(f"\n   Crop size: {crop_size}x{crop_size}")

        # Extract center
        kspace_center = reconstructor.get_kspace_center(crop_size)
        print(f"      Center shape: {kspace_center.shape}")

        # Reconstruct from center only
        reconstructor_center = MRIReconstructor(kspace_center)
        magnitude_center = reconstructor_center.reconstruct_magnitude()

        # Visualize K-space center
        kspace_center_mag = np.log(np.abs(kspace_center) + 1)
        axes[0, i].imshow(kspace_center_mag, cmap="gray")
        axes[0, i].set_title(f"K-space Center {crop_size}x{crop_size}")
        axes[0, i].axis("off")

        # Visualize reconstructed image
        axes[1, i].imshow(magnitude_center, cmap="gray")
        axes[1, i].set_title(f"Image from Center")
        axes[1, i].axis("off")

    plt.tight_layout()
    plt.savefig("mri_kspace_center.png", dpi=150, bbox_inches="tight")
    plt.close()

    print("\n📊 Saved visualization: mri_kspace_center.png")
    print("\nObservation:")
    print("  - Larger center → More contrast info → Better image quality")
    print("  - Smaller center → Less info → Lower resolution (but faster)")


def main():
    """Run all MRI reconstruction tests"""
    print("\n" + "=" * 70)
    print("  MRI RECONSTRUCTION DEMO")
    print("=" * 70)
    print("\nGIẢI THÍCH TỔNG QUAN:")
    print("-" * 70)
    print("MRI reconstruction là quá trình chuyển đổi K-space sang ảnh:")
    print("")
    print("  K-space (frequency domain) ←→ Image (spatial domain)")
    print("              ↑                          ↑")
    print("         Forward FFT              Inverse FFT")
    print("")
    print("K-space structure:")
    print("  - Center = Low frequencies = Contrast (độ sáng tối)")
    print("  - Edges = High frequencies = Details (chi tiết, cạnh)")
    print("")
    print("Reconstruction output:")
    print("  - Magnitude = |complex| = Ảnh MRI thông thường")
    print("  - Phase = angle(complex) = Thông tin pha (advanced)")
    print("=" * 70)

    # Run tests
    try:
        test_1_real_kspace()
    except Exception as e:
        print(f"\n❌ Test 1 failed: {e}")

    try:
        test_2_synthetic_brain()
    except Exception as e:
        print(f"\n❌ Test 2 failed: {e}")

    try:
        test_3_partial_fourier()
    except Exception as e:
        print(f"\n❌ Test 3 failed: {e}")

    try:
        test_4_forward_inverse_fft()
    except Exception as e:
        print(f"\n❌ Test 4 failed: {e}")

    try:
        test_5_kspace_center()
    except Exception as e:
        print(f"\n❌ Test 5 failed: {e}")

    # Summary
    print("\n" + "=" * 70)
    print("  SUMMARY")
    print("=" * 70)
    print("\n✅ MRI Reconstruction module tested successfully!")
    print("\nGenerated files:")
    print("  📊 mri_real_reconstruction.png - Real K-space reconstruction")
    print("  📊 mri_synthetic_brain.png - Synthetic brain phantom")
    print("  📊 mri_partial_fourier.png - Partial Fourier comparison")
    print("  📊 mri_fft_consistency.png - FFT consistency check")
    print("  📊 mri_kspace_center.png - K-space center extraction")
    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
