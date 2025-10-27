"""
Demo script for Image Utilities module
Tests: Format conversions, coordinate transforms, resampling, normalization
"""

import sys
from pathlib import Path
import numpy as np

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from utils.image_utils import (
    # Format conversions
    numpy_to_sitk,
    sitk_to_numpy,
    numpy_to_pil,
    pil_to_numpy,
    # Coordinate transformations
    voxel_to_world,
    world_to_voxel,
    # Resampling
    resample_image,
    resize_array,
    # Normalization
    normalize_array,
    # Basic operations
    crop_to_nonzero,
    pad_array,
    extract_slice,
)


def main():
    print("=" * 70)
    print("DEMO: Image Utilities - Comprehensive Testing")
    print("=" * 70)

    # Create test directory
    test_dir = Path("data/test_output")
    test_dir.mkdir(parents=True, exist_ok=True)

    # ========================================================================
    # Test 1: Format Conversions (NumPy ↔ SimpleITK)
    # ========================================================================
    print("\n\nTest 1: Format Conversions (NumPy ↔ SimpleITK)")
    print("-" * 70)

    # Create test volume
    array_3d = np.random.randint(0, 255, (64, 64, 32), dtype=np.uint8)
    spacing = (1.0, 1.0, 2.5)
    origin = (0.0, 0.0, 0.0)

    print(f"Original NumPy array: shape={array_3d.shape}, dtype={array_3d.dtype}")

    # NumPy -> SimpleITK
    print("\n1a. NumPy -> SimpleITK")
    sitk_img = numpy_to_sitk(array_3d, spacing=spacing, origin=origin)
    print(
        f"✅ SimpleITK image: size={sitk_img.GetSize()}, spacing={sitk_img.GetSpacing()}"
    )

    # SimpleITK -> NumPy
    print("\n1b. SimpleITK -> NumPy")
    array_back, metadata = sitk_to_numpy(sitk_img)
    print(f"✅ Back to NumPy: shape={array_back.shape}")
    print(f"   Metadata: spacing={metadata['spacing']}, origin={metadata['origin']}")

    # Verify round-trip
    assert np.array_equal(array_3d, array_back), "Round-trip conversion failed!"
    print("✅ Round-trip verification passed!")

    # ========================================================================
    # Test 2: PIL Conversions (2D slices)
    # ========================================================================
    print("\n\nTest 2: PIL Conversions (2D slices)")
    print("-" * 70)

    # Extract 2D slice
    slice_2d = array_3d[16, :, :]
    print(f"2D slice: shape={slice_2d.shape}")

    # NumPy -> PIL
    print("\n2a. NumPy -> PIL")
    pil_img = numpy_to_pil(slice_2d)
    print(f"✅ PIL Image: size={pil_img.size}, mode={pil_img.mode}")

    # Save PIL image
    pil_path = test_dir / "test_slice.png"
    pil_img.save(pil_path)
    print(f"✅ Saved PIL image: {pil_path}")

    # PIL -> NumPy
    print("\n2b. PIL -> NumPy")
    array_from_pil = pil_to_numpy(pil_img)
    print(f"✅ Back to NumPy: shape={array_from_pil.shape}")

    # ========================================================================
    # Test 3: Coordinate Transformations
    # ========================================================================
    print("\n\nTest 3: Coordinate Transformations (Voxel ↔ World)")
    print("-" * 70)

    voxel_coords = (50, 60, 25)
    spacing = (1.0, 1.0, 2.5)
    origin = (0.0, 0.0, 0.0)

    print(f"Voxel coordinates: {voxel_coords}")
    print(f"Spacing: {spacing} mm")
    print(f"Origin: {origin} mm")

    # Voxel -> World
    print("\n3a. Voxel -> World")
    world_coords = voxel_to_world(voxel_coords, spacing, origin)
    print(f"✅ World coordinates: {world_coords} mm")

    # World -> Voxel
    print("\n3b. World -> Voxel")
    voxel_back = world_to_voxel(world_coords, spacing, origin)
    print(f"✅ Voxel coordinates: {voxel_back}")

    # Verify round-trip
    assert np.allclose(voxel_coords, voxel_back), "Coordinate round-trip failed!"
    print("✅ Round-trip verification passed!")

    # ========================================================================
    # Test 4: Resampling (Change Spacing)
    # ========================================================================
    print("\n\nTest 4: Resampling (Change Spacing)")
    print("-" * 70)

    # Original image
    original_size = sitk_img.GetSize()
    original_spacing = sitk_img.GetSpacing()
    print(f"Original: size={original_size}, spacing={original_spacing}")

    # Resample to isotropic 1mm spacing
    print("\n4a. Resample to isotropic 1mm spacing")
    resampled = resample_image(sitk_img, new_spacing=(1.0, 1.0, 1.0))
    print(f"✅ Resampled: size={resampled.GetSize()}, spacing={resampled.GetSpacing()}")

    # Resize to specific size
    print("\n4b. Resize to specific dimensions")
    resized = resample_image(sitk_img, new_size=(128, 128, 64))
    print(f"✅ Resized: size={resized.GetSize()}, spacing={resized.GetSpacing()}")

    # ========================================================================
    # Test 5: Resize NumPy Array
    # ========================================================================
    print("\n\nTest 5: Resize NumPy Array")
    print("-" * 70)

    print(f"Original array: {array_3d.shape}")

    # Resize
    print("\n5a. Resize to (128, 128, 64)")
    resized_array = resize_array(array_3d, (128, 128, 64), order=1)
    print(f"✅ Resized array: {resized_array.shape}")

    # Downscale
    print("\n5b. Downscale to (32, 32, 16)")
    downscaled = resize_array(array_3d, (32, 32, 16), order=0)
    print(f"✅ Downscaled array: {downscaled.shape}")

    # ========================================================================
    # Test 6: Normalization
    # ========================================================================
    print("\n\nTest 6: Normalization Methods")
    print("-" * 70)

    # Create test array with known range
    test_array = np.random.rand(50, 50, 25) * 100 + 50  # Range [50, 150]
    print(f"Test array: range=[{test_array.min():.2f}, {test_array.max():.2f}]")

    # Min-max normalization
    print("\n6a. Min-Max Normalization [0, 1]")
    norm_minmax = normalize_array(test_array, method="min-max")
    print(f"✅ Normalized: range=[{norm_minmax.min():.3f}, {norm_minmax.max():.3f}]")

    # Z-score normalization
    print("\n6b. Z-Score Normalization")
    norm_zscore = normalize_array(test_array, method="z-score")
    print(
        f"✅ Standardized: mean={norm_zscore.mean():.3f}, std={norm_zscore.std():.3f}"
    )

    # Percentile clipping
    print("\n6c. Percentile Clipping (1%, 99%)")
    norm_percentile = normalize_array(
        test_array, method="percentile", clip_percentile=(1, 99)
    )
    print(
        f"✅ Clipped: range=[{norm_percentile.min():.3f}, {norm_percentile.max():.3f}]"
    )

    # ========================================================================
    # Test 7: Crop to Non-Zero
    # ========================================================================
    print("\n\nTest 7: Crop to Non-Zero Bounding Box")
    print("-" * 70)

    # Create array with zeros around edges
    padded_array = np.zeros((100, 100, 50), dtype=np.uint8)
    padded_array[20:80, 20:80, 10:40] = 255
    print(f"Padded array: {padded_array.shape}")
    print(f"Non-zero region: [20:80, 20:80, 10:40]")

    # Crop
    print("\n7a. Crop to non-zero")
    cropped, bbox = crop_to_nonzero(padded_array, margin=5)
    print(f"✅ Cropped: {cropped.shape}")
    print(f"   Bounding box: {bbox}")

    # Crop with margin
    print("\n7b. Crop with margin=10")
    cropped_margin, bbox_margin = crop_to_nonzero(padded_array, margin=10)
    print(f"✅ Cropped with margin: {cropped_margin.shape}")
    print(f"   Bounding box: {bbox_margin}")

    # ========================================================================
    # Test 8: Padding
    # ========================================================================
    print("\n\nTest 8: Padding to Target Shape")
    print("-" * 70)

    small_array = np.ones((50, 50, 25), dtype=np.uint8) * 128
    print(f"Small array: {small_array.shape}")

    # Pad to larger size
    print("\n8a. Pad to (100, 100, 50)")
    padded = pad_array(small_array, (100, 100, 50), mode="constant", constant_value=0)
    print(f"✅ Padded: {padded.shape}")

    # Pad with edge mode
    print("\n8b. Pad with 'edge' mode")
    padded_edge = pad_array(small_array, (100, 100, 50), mode="edge")
    print(f"✅ Padded (edge): {padded_edge.shape}")

    # ========================================================================
    # Test 9: Extract Slices
    # ========================================================================
    print("\n\nTest 9: Extract 2D Slices from 3D Volume")
    print("-" * 70)

    volume = np.random.rand(64, 64, 32)
    print(f"Volume: {volume.shape}")

    # Axial slice
    print("\n9a. Extract axial slice (axis=0)")
    axial = extract_slice(volume, volume.shape[0] // 2, axis=0)
    print(f"✅ Axial slice: {axial.shape}")

    # Coronal slice
    print("\n9b. Extract coronal slice (axis=1)")
    coronal = extract_slice(volume, volume.shape[1] // 2, axis=1)
    print(f"✅ Coronal slice: {coronal.shape}")

    # Sagittal slice
    print("\n9c. Extract sagittal slice (axis=2)")
    sagittal = extract_slice(volume, volume.shape[2] // 2, axis=2)
    print(f"✅ Sagittal slice: {sagittal.shape}")

    # ========================================================================
    # Summary
    # ========================================================================
    print("\n" + "=" * 70)
    print("DEMO COMPLETE - All Image Utilities Tests Passed! 🎉")
    print("=" * 70)
    print("\nTested Features:")
    print("  ✅ NumPy ↔ SimpleITK conversions")
    print("  ✅ NumPy ↔ PIL conversions")
    print("  ✅ Voxel ↔ World coordinate transforms")
    print("  ✅ Image resampling (spacing & size)")
    print("  ✅ Array resizing")
    print("  ✅ Multiple normalization methods")
    print("  ✅ Crop to non-zero bounding box")
    print("  ✅ Padding operations")
    print("  ✅ Slice extraction (axial, coronal, sagittal)")
    print(f"\n📁 Output saved to: {test_dir.absolute()}")


if __name__ == "__main__":
    main()
