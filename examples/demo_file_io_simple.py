"""
Simple demo script for File I/O module with generated test data
"""

import sys
from pathlib import Path
import numpy as np

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from utils.file_io import read_image, write_image, get_image_info


def main():
    print("=" * 70)
    print("DEMO: Medical Image File I/O - Comprehensive Format Testing")
    print("=" * 70)

    # Create test directory
    test_dir = Path("data/test_output")
    test_dir.mkdir(parents=True, exist_ok=True)

    # Generate test data
    print("\nGenerating test data...")
    print("-" * 70)

    # Create a simple 3D volume (simulating a medical image)
    image_3d = np.random.randint(0, 255, (64, 64, 32), dtype=np.uint8)
    print(f"✅ Generated 3D volume: {image_3d.shape}, dtype={image_3d.dtype}")

    # Create metadata
    metadata = {
        "spacing": (1.0, 1.0, 2.5),  # mm
        "origin": (0.0, 0.0, 0.0),
        "direction": np.eye(3).flatten().tolist(),
    }
    print(f"✅ Created metadata: spacing={metadata['spacing']}")

    # Test 1: Write and Read NIfTI
    print("\n\nTest 1: NIfTI Format (.nii.gz)")
    print("-" * 70)
    nifti_file = test_dir / "test_volume.nii.gz"
    try:
        print("Writing NIfTI file...")
        write_image(image_3d, str(nifti_file), metadata)
        print(f"✅ Written: {nifti_file}")

        print("Reading NIfTI file...")
        img_read, meta_read = read_image(str(nifti_file))
        print(f"✅ Read: shape={img_read.shape}, dtype={img_read.dtype}")
        print(f"   Spacing: {meta_read.get('spacing', 'N/A')}")

        # Verify
        assert img_read.shape == image_3d.shape, "Shape mismatch!"
        print("✅ Verification passed!")
    except Exception as e:
        print(f"❌ Error: {str(e)}")

    # Test 2: Write and Read NumPy
    print("\n\nTest 2: NumPy Format (.npy)")
    print("-" * 70)
    numpy_file = test_dir / "test_array.npy"
    try:
        print("Writing NumPy file...")
        write_image(image_3d, str(numpy_file), metadata)
        print(f"✅ Written: {numpy_file}")

        print("Reading NumPy file...")
        img_read, meta_read = read_image(str(numpy_file))
        print(f"✅ Read: shape={img_read.shape}, dtype={img_read.dtype}")

        # Verify
        assert np.array_equal(img_read, image_3d), "Data mismatch!"
        print("✅ Verification passed!")
    except Exception as e:
        print(f"❌ Error: {str(e)}")

    # Test 3: Write and Read NRRD
    print("\n\nTest 3: NRRD Format (.nrrd)")
    print("-" * 70)
    nrrd_file = test_dir / "test_volume.nrrd"
    try:
        print("Writing NRRD file...")
        write_image(image_3d, str(nrrd_file), metadata)
        print(f"✅ Written: {nrrd_file}")

        print("Reading NRRD file...")
        img_read, meta_read = read_image(str(nrrd_file))
        print(f"✅ Read: shape={img_read.shape}, dtype={img_read.dtype}")
        print(f"   Spacing: {meta_read.get('spacing', 'N/A')}")

        # Verify
        assert img_read.shape == image_3d.shape, "Shape mismatch!"
        print("✅ Verification passed!")
    except Exception as e:
        print(f"❌ Error: {str(e)}")

    # Test 4: Write and Read MetaImage
    print("\n\nTest 4: MetaImage Format (.mha)")
    print("-" * 70)
    mha_file = test_dir / "test_volume.mha"
    try:
        print("Writing MetaImage file...")
        write_image(image_3d, str(mha_file), metadata)
        print(f"✅ Written: {mha_file}")

        print("Reading MetaImage file...")
        img_read, meta_read = read_image(str(mha_file))
        print(f"✅ Read: shape={img_read.shape}, dtype={img_read.dtype}")
        print(f"   Spacing: {meta_read.get('spacing', 'N/A')}")

        # Verify
        assert img_read.shape == image_3d.shape, "Shape mismatch!"
        print("✅ Verification passed!")
    except Exception as e:
        print(f"❌ Error: {str(e)}")

    # Test 5: Get file info
    print("\n\nTest 5: Get Image Info (without loading)")
    print("-" * 70)
    try:
        info = get_image_info(str(nifti_file))
        print(f"File: {nifti_file.name}")
        print(f"✅ Size: {info['file_size_mb']:.4f} MB")
        print(f"✅ Format: {info['format']}")
    except Exception as e:
        print(f"❌ Error: {str(e)}")

    # Test 6: Test existing NumPy files (with allow_pickle)
    print("\n\nTest 6: Reading Existing NumPy Files")
    print("-" * 70)
    existing_numpy_files = [
        "data/medical/slice_kspace.npy",
        "data/medical/Schepp_Logan_sinogram 1.npy",
    ]

    for file_path in existing_numpy_files:
        if Path(file_path).exists():
            print(f"\nReading: {file_path}")
            try:
                img, meta = read_image(file_path)
                print(f"✅ Success: shape={img.shape}, dtype={img.dtype}")
            except Exception as e:
                print(f"❌ Error: {str(e)}")
        else:
            print(f"⚠️  File not found: {file_path}")

    print("\n" + "=" * 70)
    print("DEMO COMPLETE - All format tests finished!")
    print("=" * 70)
    print(f"\n📁 Test files created in: {test_dir.absolute()}")


if __name__ == "__main__":
    main()
