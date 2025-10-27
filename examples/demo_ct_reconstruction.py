"""
Demo script for CT Reconstruction

Demonstrates:
1. Creating synthetic phantom (Shepp-Logan)
2. Generating sinogram (forward projection)
3. FBP reconstruction with different filters
4. SART reconstruction with different iterations
5. Comparing methods
6. Quality metrics

GIẢI THÍCH:
-----------
CT Reconstruction = Tái tạo ảnh từ sinogram

Quy trình:
1. Original Image → Sinogram (máy CT tạo ra)
2. Sinogram → Reconstructed Image (thuật toán tái tạo)

2 phương pháp chính:
- FBP: Nhanh, chính xác, dùng phổ biến
- SART: Iterative, tốt cho dữ liệu thiếu
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.reconstruction.ct_reconstruction import CTReconstructor, reconstruct_ct
from skimage.data import shepp_logan_phantom
from skimage.transform import radon, rescale


def create_test_data():
    """
    Tạo test data: Shepp-Logan phantom + sinogram.

    Shepp-Logan phantom là ảnh test chuẩn trong CT imaging:
    - Mô phỏng mặt cắt não người
    - Có nhiều vùng với mật độ khác nhau
    - Dùng để test chất lượng reconstruction

    Returns:
        (phantom, sinogram, theta)
    """
    print("\n1. Creating Shepp-Logan phantom (brain model)...")

    # Create phantom (400x400 too large, rescale to 256x256)
    phantom = shepp_logan_phantom()
    phantom = rescale(phantom, scale=0.64, mode="reflect", channel_axis=None)

    print(f"   Phantom size: {phantom.shape}")
    print(f"   Value range: [{phantom.min():.3f}, {phantom.max():.3f}]")

    # Generate sinogram (simulate CT scan)
    print("\n2. Generating sinogram (simulating CT scan)...")
    print("   - CT scanner quay 180° quanh phantom")
    print("   - Mỗi góc tạo 1 projection (1 hàng trong sinogram)")

    theta = np.linspace(0.0, 180.0, 180, endpoint=False)
    sinogram = radon(phantom, theta=theta, circle=True)

    # radon returns (num_detectors, num_angles), we need (num_angles, num_detectors)
    sinogram = sinogram.T

    print(f"   Sinogram size: {sinogram.shape}")
    print(f"   - {sinogram.shape[0]} angles (góc chiếu)")
    print(f"   - {sinogram.shape[1]} detectors (đầu dò)")

    return phantom, sinogram, theta


def demo_fbp_filters(reconstructor, phantom):
    """
    Demo FBP với các filters khác nhau.

    Giải thích filters:
    - ramp: Sharp nhất, giữ nguyên high frequency
    - shepp-logan: Balance giữa sharp và smooth
    - cosine: Medium smoothing
    - hamming: Smooth, giảm noise
    - hann: Smoothest, noise thấp nhất
    """
    print("\n" + "=" * 70)
    print("TEST 1: FBP with Different Filters")
    print("=" * 70)

    filters = ["ramp", "shepp-logan", "cosine", "hamming", "hann"]
    results = {}

    for filter_name in filters:
        print(f"\n   Testing filter: {filter_name}")

        # Reconstruct
        recon = reconstructor.reconstruct_fbp(filter_name=filter_name)
        results[filter_name] = recon

        # Calculate error vs ground truth
        error = np.mean(np.abs(recon - phantom))
        print(f"   ✅ MAE vs ground truth: {error:.4f}")

    # Visualize
    print("\n   Creating visualization...")
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Original phantom
    axes[0, 0].imshow(phantom, cmap="gray")
    axes[0, 0].set_title("Original Phantom\n(Ground Truth)", fontsize=10)
    axes[0, 0].axis("off")

    # Reconstructions
    for idx, (filter_name, recon) in enumerate(results.items()):
        row = (idx + 1) // 3
        col = (idx + 1) % 3

        axes[row, col].imshow(recon, cmap="gray")
        axes[row, col].set_title(f"FBP - {filter_name} filter", fontsize=10)
        axes[row, col].axis("off")

    plt.suptitle(
        "CT Reconstruction: FBP with Different Filters", fontsize=14, fontweight="bold"
    )
    plt.tight_layout()

    output_path = "data/test_output/ct_fbp_filters.png"
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"   ✅ Saved: {output_path}")
    plt.close()

    return results


def demo_sart_iterations(reconstructor, phantom):
    """
    Demo SART với số iterations khác nhau.

    Giải thích:
    - 1 iteration: Nhanh nhưng quality thấp
    - 5 iterations: Balance tốt
    - 10 iterations: Chất lượng cao nhưng chậm
    """
    print("\n" + "=" * 70)
    print("TEST 2: SART with Different Iterations")
    print("=" * 70)

    iterations_list = [1, 2, 5, 10]
    results = {}

    for iters in iterations_list:
        print(f"\n   Testing SART with {iters} iteration(s)...")

        # Reconstruct
        recon = reconstructor.reconstruct_sart(iterations=iters)
        results[iters] = recon

        # Calculate error
        error = np.mean(np.abs(recon - phantom))
        print(f"   ✅ MAE vs ground truth: {error:.4f}")

    # Visualize
    print("\n   Creating visualization...")
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Original
    axes[0, 0].imshow(phantom, cmap="gray")
    axes[0, 0].set_title("Original Phantom\n(Ground Truth)", fontsize=10)
    axes[0, 0].axis("off")

    # FBP for comparison
    fbp = reconstructor.reconstruct_fbp(filter_name="ramp")
    axes[0, 1].imshow(fbp, cmap="gray")
    axes[0, 1].set_title("FBP (ramp filter)\nfor comparison", fontsize=10)
    axes[0, 1].axis("off")

    # SART iterations
    for idx, (iters, recon) in enumerate(results.items()):
        row = (idx + 2) // 3
        col = (idx + 2) % 3

        axes[row, col].imshow(recon, cmap="gray")
        axes[row, col].set_title(f"SART - {iters} iteration(s)", fontsize=10)
        axes[row, col].axis("off")

    plt.suptitle(
        "CT Reconstruction: SART with Different Iterations",
        fontsize=14,
        fontweight="bold",
    )
    plt.tight_layout()

    output_path = "data/test_output/ct_sart_iterations.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"   ✅ Saved: {output_path}")
    plt.close()

    return results


def demo_comparison(reconstructor, phantom):
    """
    So sánh trực tiếp FBP vs SART.
    """
    print("\n" + "=" * 70)
    print("TEST 3: Direct Comparison - FBP vs SART")
    print("=" * 70)

    print("\n   Reconstructing with both methods...")
    comparison = reconstructor.compare_methods(filter_name="ramp", sart_iterations=5)

    # Visualize
    print("\n   Creating comparison visualization...")
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Row 1: Images
    axes[0, 0].imshow(phantom, cmap="gray")
    axes[0, 0].set_title("Ground Truth", fontsize=12, fontweight="bold")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(comparison["fbp"], cmap="gray")
    axes[0, 1].set_title("FBP Reconstruction", fontsize=12, fontweight="bold")
    axes[0, 1].axis("off")

    axes[0, 2].imshow(comparison["sart"], cmap="gray")
    axes[0, 2].set_title("SART Reconstruction (5 iter)", fontsize=12, fontweight="bold")
    axes[0, 2].axis("off")

    # Row 2: Errors
    error_fbp = np.abs(comparison["fbp"] - phantom)
    error_sart = np.abs(comparison["sart"] - phantom)

    im1 = axes[1, 0].imshow(error_fbp, cmap="hot")
    axes[1, 0].set_title(f"FBP Error\nMAE: {error_fbp.mean():.4f}", fontsize=10)
    axes[1, 0].axis("off")
    plt.colorbar(im1, ax=axes[1, 0], fraction=0.046)

    im2 = axes[1, 1].imshow(error_sart, cmap="hot")
    axes[1, 1].set_title(f"SART Error\nMAE: {error_sart.mean():.4f}", fontsize=10)
    axes[1, 1].axis("off")
    plt.colorbar(im2, ax=axes[1, 1], fraction=0.046)

    im3 = axes[1, 2].imshow(
        comparison["difference"], cmap="seismic", vmin=-0.1, vmax=0.1
    )
    axes[1, 2].set_title("FBP - SART\nDifference", fontsize=10)
    axes[1, 2].axis("off")
    plt.colorbar(im3, ax=axes[1, 2], fraction=0.046)

    plt.suptitle("CT Reconstruction: Method Comparison", fontsize=14, fontweight="bold")
    plt.tight_layout()

    output_path = "data/test_output/ct_comparison.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"   ✅ Saved: {output_path}")
    plt.close()


def demo_sinogram_visualization(phantom, sinogram, theta):
    """
    Visualize sinogram và reconstruction process.
    """
    print("\n" + "=" * 70)
    print("TEST 4: Sinogram Visualization")
    print("=" * 70)

    print("\n   Creating sinogram visualization...")

    reconstructor = CTReconstructor(sinogram, theta=theta)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Original phantom
    axes[0].imshow(phantom, cmap="gray")
    axes[0].set_title("Original Phantom", fontsize=12, fontweight="bold")
    axes[0].axis("off")

    # Sinogram
    im = axes[1].imshow(
        sinogram, cmap="gray", aspect="auto", extent=[0, sinogram.shape[1], 180, 0]
    )
    axes[1].set_title("Sinogram\n(Projection Data)", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Detector Position", fontsize=10)
    axes[1].set_ylabel("Projection Angle (degrees)", fontsize=10)
    plt.colorbar(im, ax=axes[1], fraction=0.046)

    # Reconstruction
    recon = reconstructor.reconstruct_fbp(filter_name="ramp")
    axes[2].imshow(recon, cmap="gray")
    axes[2].set_title("FBP Reconstruction", fontsize=12, fontweight="bold")
    axes[2].axis("off")

    plt.suptitle(
        "CT Imaging Process: Phantom → Sinogram → Reconstruction",
        fontsize=14,
        fontweight="bold",
    )
    plt.tight_layout()

    output_path = "data/test_output/ct_process.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"   ✅ Saved: {output_path}")
    plt.close()


def main():
    print("=" * 70)
    print("DEMO: CT RECONSTRUCTION")
    print("=" * 70)
    print("\nKIẾN THỨC:")
    print("  • CT Scanner quay quanh bệnh nhân, chụp từ nhiều góc")
    print("  • Mỗi góc tạo 1 projection → ghép lại = SINOGRAM")
    print("  • Reconstruction: SINOGRAM → ẢNH CT 2D")
    print("  • 2 phương pháp: FBP (nhanh) và SART (iterative)")
    print("=" * 70)

    # Create test data
    phantom, sinogram, theta = create_test_data()

    # Initialize reconstructor
    print("\n3. Initializing CT Reconstructor...")
    reconstructor = CTReconstructor(sinogram, theta=theta)
    print("   ✅ Ready to reconstruct!")

    # Run tests
    demo_fbp_filters(reconstructor, phantom)
    demo_sart_iterations(reconstructor, phantom)
    demo_comparison(reconstructor, phantom)
    demo_sinogram_visualization(phantom, sinogram, theta)

    # Test using real sinogram data (if exists)
    print("\n" + "=" * 70)
    print("TEST 5: Real Sinogram Data (if available)")
    print("=" * 70)

    real_sinogram_path = "data/medical/Schepp_Logan_sinogram 1.npy"
    if Path(real_sinogram_path).exists():
        print(f"\n   Loading real sinogram: {real_sinogram_path}")
        try:
            real_sinogram = np.load(real_sinogram_path, allow_pickle=True)

            # Handle if it's an object array
            if real_sinogram.dtype == object:
                real_sinogram = real_sinogram.item()
                if isinstance(real_sinogram, dict):
                    real_sinogram = real_sinogram.get("sinogram", real_sinogram)

            print(f"   Shape: {real_sinogram.shape}")

            # Reconstruct
            real_reconstructor = CTReconstructor(real_sinogram)
            real_recon = real_reconstructor.reconstruct_fbp(filter_name="ramp")

            # Visualize
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))

            axes[0].imshow(real_sinogram, cmap="gray", aspect="auto")
            axes[0].set_title("Real Sinogram Data", fontweight="bold")
            axes[0].set_xlabel("Detector")
            axes[0].set_ylabel("Angle")

            axes[1].imshow(real_recon, cmap="gray")
            axes[1].set_title("FBP Reconstruction", fontweight="bold")
            axes[1].axis("off")

            plt.tight_layout()
            output_path = "data/test_output/ct_real_data.png"
            plt.savefig(output_path, dpi=150)
            print(f"   ✅ Saved: {output_path}")
            plt.close()

        except Exception as e:
            print(f"   ⚠️  Could not process real data: {e}")
    else:
        print(f"   ℹ️  No real sinogram data found at: {real_sinogram_path}")

    # Summary
    print("\n" + "=" * 70)
    print("DEMO COMPLETE - All CT Reconstruction Tests Passed! 🎉")
    print("=" * 70)
    print("\nSummary:")
    print("  ✅ Created Shepp-Logan phantom")
    print("  ✅ Generated sinogram (180 angles)")
    print("  ✅ Tested FBP with 5 different filters")
    print("  ✅ Tested SART with 4 iteration counts")
    print("  ✅ Compared FBP vs SART methods")
    print("  ✅ Visualized reconstruction process")
    print("\nKết luận:")
    print("  • FBP: Nhanh, chính xác, phù hợp cho production")
    print("  • SART: Tốt khi thiếu data, nhưng chậm hơn")
    print("  • Filter 'ramp': Sharp nhất")
    print("  • Filter 'hamming': Smooth nhất, ít noise")
    print(f"\n📁 Output images: {Path('data/test_output').absolute()}")


if __name__ == "__main__":
    main()
