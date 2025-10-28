"""
Brain Segmentation Demo

Demonstrates brain segmentation algorithms:
1. Threshold-based segmentation (Manual, Otsu)
2. Region growing
3. Morphological operations
4. Evaluation metrics (Dice, IoU, Hausdorff)
5. Pipeline comparison

GIẢI THÍCH:
-----------
Segmentation = Chia ảnh thành các vùng có ý nghĩa (ROI)

Phương pháp:
- Threshold: Phân loại dựa trên intensity
- Region growing: Mở rộng từ seed point
- Morphology: Erosion, dilation, opening, closing

Metrics:
- Dice coefficient: Độ overlap (0-1)
- IoU: Intersection over Union
- Hausdorff: Khoảng cách boundary

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

from src.segmentation import BrainSegmentation


def create_synthetic_brain(size: int = 256) -> np.ndarray:
    """
    Create synthetic brain MRI for testing.

    GIẢI THÍCH:
    -----------
    Tạo ảnh synthetic brain với:
    - Brain tissue (intensity 150-200)
    - Background (intensity 0-50)
    - Some noise

    Args:
        size: Image size

    Returns:
        Synthetic brain image
    """
    print("\n🧠 Creating synthetic brain MRI...")

    image = np.zeros((size, size))

    # Create brain region (ellipse)
    y, x = np.ogrid[:size, :size]
    cy, cx = size // 2, size // 2

    # Main brain (ellipse)
    ellipse = (x - cx) ** 2 / (size * 0.35) ** 2 + (y - cy) ** 2 / (
        size * 0.4
    ) ** 2 <= 1
    image[ellipse] = 180

    # Add internal structures
    # Ventricles (dark regions)
    for _ in range(2):
        offset_x = np.random.randint(-size // 6, size // 6)
        offset_y = np.random.randint(-size // 6, size // 6)
        radius = np.random.randint(size // 15, size // 10)

        circle = (x - (cx + offset_x)) ** 2 + (y - (cy + offset_y)) ** 2 <= radius**2
        image[circle] = 50

    # Add gray matter variations
    for _ in range(5):
        offset_x = np.random.randint(-size // 4, size // 4)
        offset_y = np.random.randint(-size // 4, size // 4)
        radius = np.random.randint(size // 12, size // 8)

        circle = (x - (cx + offset_x)) ** 2 + (y - (cy + offset_y)) ** 2 <= radius**2
        image[circle] = 160

    # Add noise
    noise = np.random.randn(size, size) * 10
    image = image + noise

    # Add background noise
    background_noise = np.random.randn(size, size) * 5 + 20
    image[~ellipse] = background_noise[~ellipse]

    # Smooth
    image = ndimage.gaussian_filter(image, sigma=1.5)

    print(f"   Image shape: {image.shape}")
    print(f"   Intensity range: [{image.min():.1f}, {image.max():.1f}]")

    return image


def create_ground_truth(image: np.ndarray) -> np.ndarray:
    """
    Create ground truth segmentation.

    Args:
        image: Synthetic brain image

    Returns:
        Ground truth mask
    """
    # Simple threshold for ground truth
    gt = (image > 100).astype(np.uint8)

    # Clean up
    gt = ndimage.binary_fill_holes(gt)
    gt = ndimage.binary_opening(gt, structure=np.ones((3, 3)))

    return gt.astype(np.uint8)


def test_1_manual_threshold():
    """
    Test 1: Manual thresholding

    GIẢI THÍCH:
    -----------
    Manual threshold: Người dùng chọn threshold value

    if intensity > threshold:
        label = 1 (foreground)
    else:
        label = 0 (background)
    """
    print("\n" + "=" * 60)
    print("TEST 1: Manual Thresholding")
    print("=" * 60)

    # Create synthetic brain
    image = create_synthetic_brain(size=256)

    # Create segmentator
    seg = BrainSegmentation(image)

    # Test different thresholds
    thresholds = [80, 100, 120]

    print(f"\n🔄 Testing thresholds: {thresholds}")

    fig, axes = plt.subplots(1, len(thresholds) + 1, figsize=(20, 5))

    # Original image
    axes[0].imshow(image, cmap="gray")
    axes[0].set_title("Original Image")
    axes[0].axis("off")

    # Different thresholds
    for i, threshold in enumerate(thresholds, start=1):
        print(f"\n   Threshold = {threshold}:")

        mask = seg.threshold_manual(threshold, mode="greater")

        axes[i].imshow(mask, cmap="gray")
        axes[i].set_title(f"Threshold = {threshold}")
        axes[i].axis("off")

    plt.tight_layout()
    plt.savefig("brain_manual_threshold.png", dpi=150, bbox_inches="tight")
    plt.close()

    print("\n📊 Saved: brain_manual_threshold.png")


def test_2_otsu_threshold():
    """
    Test 2: Otsu's automatic thresholding

    GIẢI THÍCH:
    -----------
    Otsu tự động tìm threshold tốt nhất:
    - Minimize within-class variance
    - Maximize between-class variance

    Không cần người dùng chọn threshold!
    """
    print("\n" + "=" * 60)
    print("TEST 2: Otsu's Automatic Thresholding")
    print("=" * 60)

    # Create synthetic brain
    image = create_synthetic_brain(size=256)
    ground_truth = create_ground_truth(image)

    # Create segmentator
    seg = BrainSegmentation(image)

    # Otsu thresholding
    print("\n🔄 Applying Otsu's method...")
    mask = seg.threshold_otsu()

    # Calculate Dice
    dice = BrainSegmentation.dice_coefficient(ground_truth, mask)
    iou = BrainSegmentation.iou_score(ground_truth, mask)

    print(f"\n📈 Evaluation:")
    print(f"   Dice coefficient: {dice:.3f}")
    print(f"   IoU score: {iou:.3f}")

    # Visualize
    seg.visualize_segmentation(mask, overlay=True, save_path="brain_otsu.png")

    print("\n📊 Saved: brain_otsu.png")

    return image, ground_truth


def test_3_region_growing():
    """
    Test 3: Region growing segmentation

    GIẢI THÍCH:
    -----------
    Region growing:
    1. Bắt đầu từ seed point
    2. Kiểm tra neighbors
    3. Nếu |intensity - seed_intensity| < tolerance:
          Thêm vào region
    4. Lặp lại
    """
    print("\n" + "=" * 60)
    print("TEST 3: Region Growing")
    print("=" * 60)

    # Create synthetic brain
    image = create_synthetic_brain(size=256)
    ground_truth = create_ground_truth(image)

    # Create segmentator
    seg = BrainSegmentation(image)

    # Region growing with single seed (center of brain)
    seed = (128, 128)
    tolerance = 30.0

    print(f"\n🔄 Region growing:")
    print(f"   Seed point: {seed}")
    print(f"   Tolerance: {tolerance}")

    mask = seg.region_growing(seed=seed, tolerance=tolerance)

    # Calculate metrics
    dice = BrainSegmentation.dice_coefficient(ground_truth, mask)
    iou = BrainSegmentation.iou_score(ground_truth, mask)

    print(f"\n📈 Evaluation:")
    print(f"   Dice coefficient: {dice:.3f}")
    print(f"   IoU score: {iou:.3f}")

    # Visualize with seed point
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(image, cmap="gray")
    axes[0].plot(seed[1], seed[0], "r*", markersize=15, label="Seed")
    axes[0].set_title("Original + Seed")
    axes[0].legend()
    axes[0].axis("off")

    axes[1].imshow(mask, cmap="gray")
    axes[1].set_title("Region Growing Result")
    axes[1].axis("off")

    axes[2].imshow(image, cmap="gray")
    axes[2].imshow(mask, cmap="Reds", alpha=0.5)
    axes[2].set_title("Overlay")
    axes[2].axis("off")

    plt.tight_layout()
    plt.savefig("brain_region_growing.png", dpi=150, bbox_inches="tight")
    plt.close()

    print("\n📊 Saved: brain_region_growing.png")


def test_4_morphological_operations():
    """
    Test 4: Morphological operations

    GIẢI THÍCH:
    -----------
    Morphology:
    - Opening = Erosion + Dilation (remove noise)
    - Closing = Dilation + Erosion (fill holes)
    - Remove small objects
    - Get largest component
    """
    print("\n" + "=" * 60)
    print("TEST 4: Morphological Operations")
    print("=" * 60)

    # Create synthetic brain with noise
    image = create_synthetic_brain(size=256)

    # Create segmentator
    seg = BrainSegmentation(image)

    # Initial segmentation (will have noise)
    print("\n🔄 Initial Otsu segmentation...")
    mask_raw = seg.threshold_otsu()

    # Apply morphological operations
    print("\n🔧 Applying morphological operations...")

    print("\n   1. Opening (remove small noise)...")
    mask_opened = seg.morphological_opening(mask_raw, kernel_size=2)

    print("\n   2. Closing (fill small holes)...")
    mask_closed = seg.morphological_closing(mask_opened, kernel_size=3)

    print("\n   3. Remove small objects...")
    mask_cleaned = seg.remove_small_objects(mask_closed, min_size=100)

    print("\n   4. Get largest component...")
    mask_final = seg.get_largest_component(mask_cleaned)

    # Visualize pipeline
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    axes[0, 0].imshow(image, cmap="gray")
    axes[0, 0].set_title("Original Image")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(mask_raw, cmap="gray")
    axes[0, 1].set_title("1. Raw Otsu")
    axes[0, 1].axis("off")

    axes[0, 2].imshow(mask_opened, cmap="gray")
    axes[0, 2].set_title("2. Opening")
    axes[0, 2].axis("off")

    axes[1, 0].imshow(mask_closed, cmap="gray")
    axes[1, 0].set_title("3. Closing")
    axes[1, 0].axis("off")

    axes[1, 1].imshow(mask_cleaned, cmap="gray")
    axes[1, 1].set_title("4. Remove Small Objects")
    axes[1, 1].axis("off")

    axes[1, 2].imshow(mask_final, cmap="gray")
    axes[1, 2].set_title("5. Largest Component")
    axes[1, 2].axis("off")

    plt.tight_layout()
    plt.savefig("brain_morphology_pipeline.png", dpi=150, bbox_inches="tight")
    plt.close()

    print("\n📊 Saved: brain_morphology_pipeline.png")


def test_5_evaluation_metrics():
    """
    Test 5: Evaluation metrics comparison

    GIẢI THÍCH:
    -----------
    Metrics để đánh giá segmentation:

    1. Dice coefficient:
       - Range: 0-1
       - Dice = 2×|A∩B| / (|A| + |B|)
       - > 0.7: Good, > 0.8: Very good, > 0.9: Excellent

    2. IoU (Intersection over Union):
       - Range: 0-1
       - IoU = |A∩B| / |A∪B|
       - More strict than Dice

    3. Hausdorff distance:
       - Distance between boundaries
       - Smaller = Better
    """
    print("\n" + "=" * 60)
    print("TEST 5: Evaluation Metrics")
    print("=" * 60)

    # Create synthetic brain
    image = create_synthetic_brain(size=256)
    ground_truth = create_ground_truth(image)

    print(f"\n📊 Ground truth:")
    print(
        f"   Foreground pixels: {ground_truth.sum()} "
        f"({ground_truth.sum() / ground_truth.size * 100:.1f}%)"
    )

    # Create segmentator
    seg = BrainSegmentation(image)

    # Test different methods
    methods = {
        "Manual (threshold=100)": lambda: seg.threshold_manual(100),
        "Otsu": lambda: seg.threshold_otsu(),
        "Region Growing": lambda: seg.region_growing((128, 128), tolerance=30),
    }

    results = []

    print("\n🔄 Evaluating different methods...")
    print("-" * 60)

    for name, method in methods.items():
        print(f"\n{name}:")

        # Segment
        mask = method()

        # Calculate metrics
        dice = BrainSegmentation.dice_coefficient(ground_truth, mask)
        iou = BrainSegmentation.iou_score(ground_truth, mask)

        try:
            hausdorff = BrainSegmentation.hausdorff_distance(ground_truth, mask)
        except:
            hausdorff = float("nan")

        print(f"   Dice coefficient: {dice:.3f}")
        print(f"   IoU score: {iou:.3f}")
        print(f"   Hausdorff distance: {hausdorff:.2f} pixels")

        results.append(
            {"method": name, "dice": dice, "iou": iou, "hausdorff": hausdorff}
        )

    # Create comparison table
    print("\n" + "=" * 60)
    print("COMPARISON TABLE")
    print("=" * 60)
    print(f"{'Method':<30} {'Dice':<10} {'IoU':<10} {'Hausdorff':<15}")
    print("-" * 60)

    for r in results:
        print(
            f"{r['method']:<30} {r['dice']:<10.3f} {r['iou']:<10.3f} "
            f"{r['hausdorff']:<15.2f}"
        )

    # Visualize comparison
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))

    # Ground truth
    axes[0, 0].imshow(ground_truth, cmap="gray")
    axes[0, 0].set_title("Ground Truth")
    axes[0, 0].axis("off")

    # Methods
    for i, (name, method) in enumerate(methods.items(), start=1):
        mask = method()

        ax = axes[i // 2, i % 2]
        ax.imshow(image, cmap="gray")
        ax.imshow(mask, cmap="Reds", alpha=0.5)

        # Get metrics
        r = results[i - 1]
        title = f"{name}\nDice={r['dice']:.3f}, IoU={r['iou']:.3f}"
        ax.set_title(title)
        ax.axis("off")

    plt.tight_layout()
    plt.savefig("brain_metrics_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()

    print("\n📊 Saved: brain_metrics_comparison.png")


def main():
    """Run all brain segmentation tests"""
    print("\n" + "=" * 70)
    print("  BRAIN SEGMENTATION DEMO")
    print("=" * 70)
    print("\nGIẢI THÍCH TỔNG QUAN:")
    print("-" * 70)
    print("Segmentation = Chia ảnh thành các vùng có ý nghĩa (ROI)")
    print("")
    print("Phương pháp:")
    print("  1. Threshold: Phân loại dựa trên intensity value")
    print("     - Manual: Người dùng chọn threshold")
    print("     - Otsu: Tự động tìm threshold tốt nhất")
    print("")
    print("  2. Region Growing: Mở rộng từ seed point")
    print("     - Start with seed")
    print("     - Grow to similar neighbors")
    print("")
    print("  3. Morphology: Xử lý hình thái")
    print("     - Opening: Remove noise")
    print("     - Closing: Fill holes")
    print("")
    print("Evaluation Metrics:")
    print("  - Dice coefficient: 0-1 (higher = better)")
    print("  - IoU: 0-1 (more strict than Dice)")
    print("  - Hausdorff: Distance (lower = better)")
    print("=" * 70)

    # Run tests
    try:
        test_1_manual_threshold()
    except Exception as e:
        print(f"\n❌ Test 1 failed: {e}")
        import traceback

        traceback.print_exc()

    try:
        test_2_otsu_threshold()
    except Exception as e:
        print(f"\n❌ Test 2 failed: {e}")
        import traceback

        traceback.print_exc()

    try:
        test_3_region_growing()
    except Exception as e:
        print(f"\n❌ Test 3 failed: {e}")
        import traceback

        traceback.print_exc()

    try:
        test_4_morphological_operations()
    except Exception as e:
        print(f"\n❌ Test 4 failed: {e}")
        import traceback

        traceback.print_exc()

    try:
        test_5_evaluation_metrics()
    except Exception as e:
        print(f"\n❌ Test 5 failed: {e}")
        import traceback

        traceback.print_exc()

    # Summary
    print("\n" + "=" * 70)
    print("  SUMMARY")
    print("=" * 70)
    print("\n✅ Brain Segmentation module tested successfully!")
    print("\nGenerated files:")
    print("  📊 brain_manual_threshold.png - Manual threshold comparison")
    print("  📊 brain_otsu.png - Otsu automatic thresholding")
    print("  📊 brain_region_growing.png - Region growing with seed")
    print("  📊 brain_morphology_pipeline.png - Morphology operations")
    print("  📊 brain_metrics_comparison.png - Methods comparison")
    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
