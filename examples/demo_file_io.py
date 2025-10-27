"""
Demo script for File I/O module
"""

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from utils.file_io import read_image, write_image, get_image_info


def main():
    print("=" * 60)
    print("DEMO: Medical Image File I/O")
    print("=" * 60)

    # Test files
    test_files = [
        ("DICOM", "data/anonym/our_sample_dicom.dcm"),
        ("DICOM (X-ray)", "data/sitk/digital_xray.dcm"),
        ("NRRD (T1)", "data/sitk/A1_grayT1.nrrd"),
        ("NRRD (T2)", "data/sitk/A1_grayT2.nrrd"),
        ("MetaImage (CT)", "data/sitk/training_001_ct.mha"),
        ("MetaImage (MR)", "data/sitk/training_001_mr_T1.mha"),
        ("NumPy (K-space)", "data/medical/slice_kspace.npy"),
        ("NumPy (Sinogram)", "data/medical/Schepp_Logan_sinogram 1.npy"),
    ]

    successful_reads = []

    for i, (desc, file_path) in enumerate(test_files, 1):
        print(f"\n{i}. Reading {desc} file...")
        print(f"   Path: {file_path}")
        try:
            image, metadata = read_image(file_path)
            print(f"   ✅ Success!")
            print(f"   Shape: {image.shape}")
            print(f"   Data type: {image.dtype}")
            if "spacing" in metadata and metadata["spacing"] is not None:
                print(f"   Spacing: {metadata['spacing']}")
            if "modality" in metadata:
                print(f"   Modality: {metadata['modality']}")
            successful_reads.append((desc, file_path, image, metadata))
        except Exception as e:
            print(f"   ❌ Error: {str(e)}")

    # Get info without loading (test on first successful file)
    if successful_reads:
        print(f"\n{len(test_files) + 1}. Getting image info (without loading)...")
        desc, file_path, _, _ = successful_reads[0]
        try:
            info = get_image_info(file_path)
            print(f"   File: {file_path}")
            print(f"   Size: {info['file_size_mb']:.4f} MB")
            print(f"   Format: {info['format']}")
            print(f"   ✅ Success!")
        except Exception as e:
            print(f"   ❌ Error: {str(e)}")

    # Write output (test with first successful read)
    if successful_reads:
        print(f"\n{len(test_files) + 2}. Writing test file...")
        desc, _, image, metadata = successful_reads[0]
        output_path = "data/output_test.nii.gz"
        try:
            write_image(image, output_path, metadata)
            print(f"   ✅ Written to: {output_path}")

            # Verify by reading back
            image_verify, _ = read_image(output_path)
            print(f"   ✅ Verified: Shape {image_verify.shape}")
        except Exception as e:
            print(f"   ❌ Error: {str(e)}")

    print("\n" + "=" * 60)
    print(
        f"DEMO COMPLETE - {len(successful_reads)}/{len(test_files)} files read successfully"
    )
    print("=" * 60)


if __name__ == "__main__":
    main()
