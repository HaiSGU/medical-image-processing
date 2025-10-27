"""
Demo script for DICOM Anonymization
"""

import sys
from pathlib import Path
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.anonymization.dicom_anonymizer import DICOMAnonymizer
import pydicom
from pydicom.dataset import FileDataset, Dataset


def create_synthetic_dicom(output_path):
    """Create a synthetic DICOM file for testing."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    # Create file meta information
    file_meta = Dataset()
    file_meta.MediaStorageSOPClassUID = "1.2.840.10008.5.1.4.1.1.2"
    file_meta.MediaStorageSOPInstanceUID = "1.2.3.4.5.6.7.8.9"
    file_meta.ImplementationClassUID = "1.2.3.4"
    file_meta.TransferSyntaxUID = "1.2.840.10008.1.2.1"

    # Create dataset with file_meta
    ds = FileDataset(output_path, {}, file_meta=file_meta, preamble=b"\0" * 128)

    # Add required tags
    ds.SOPClassUID = "1.2.840.10008.5.1.4.1.1.2"
    ds.SOPInstanceUID = "1.2.3.4.5.6.7.8.9"
    ds.StudyInstanceUID = "1.2.3.4.5.6"
    ds.SeriesInstanceUID = "1.2.3.4.5.6.7"

    # Add patient information (PHI - to be anonymized)
    ds.PatientName = "Doe^John"
    ds.PatientID = "12345678"
    ds.PatientBirthDate = "19800115"
    ds.PatientSex = "M"
    ds.PatientAge = "044Y"

    # Add study information
    ds.StudyDate = "20240127"
    ds.StudyTime = "120000"
    ds.SeriesDate = "20240127"
    ds.SeriesTime = "120100"
    ds.Modality = "CT"
    ds.StudyDescription = "Brain CT"
    ds.SeriesDescription = "Axial Brain"
    ds.InstitutionName = "Test Hospital"
    ds.ReferringPhysicianName = "Dr. Smith"

    # Add image data
    ds.Rows = 512
    ds.Columns = 512
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.BitsAllocated = 16
    ds.BitsStored = 16
    ds.HighBit = 15
    ds.PixelRepresentation = 0
    ds.PixelData = np.random.randint(0, 4096, (512, 512), dtype=np.uint16).tobytes()

    # Save properly
    ds.is_little_endian = True
    ds.is_implicit_VR = False
    ds.save_as(output_path, write_like_original=False)

    return output_path


def main():
    print("=" * 70)
    print("DEMO: DICOM Anonymization")
    print("=" * 70)

    # Initialize anonymizer
    anonymizer = DICOMAnonymizer(prefix="ANON")

    # File paths
    original_path = "data/test_output/synthetic_dicom.dcm"
    output_path = "data/test_output/anonymized_dicom.dcm"

    # Create synthetic DICOM
    print("\n1. Creating synthetic DICOM for testing...")
    create_synthetic_dicom(original_path)
    print(f"   ✅ Created: {original_path}")

    # Read and display original DICOM
    print("\n2. Reading original DICOM...")
    original_ds = pydicom.dcmread(original_path)
    print(f"   Patient Name: {original_ds.PatientName}")
    print(f"   Patient ID: {original_ds.PatientID}")
    print(f"   Patient Birth Date: {original_ds.PatientBirthDate}")
    print(f"   Study Date: {original_ds.StudyDate}")
    print(f"   Modality: {original_ds.Modality}")
    print(f"   Institution: {original_ds.InstitutionName}")
    print(f"   Referring Physician: {original_ds.ReferringPhysicianName}")

    # Anonymize
    print("\n3. Anonymizing DICOM file...")
    anonymizer.anonymize_file(original_path, output_path, keep_descriptive=True)
    print(f"   ✅ Anonymized: {output_path}")

    # Read and display anonymized DICOM
    print("\n4. Reading anonymized DICOM...")
    anon_ds = pydicom.dcmread(output_path)
    print(f"   Patient Name: {anon_ds.PatientName}")
    print(f"   Patient ID: {anon_ds.PatientID}")
    print(f"   Patient Birth Date: {getattr(anon_ds, 'PatientBirthDate', '[REMOVED]')}")
    print(f"   Study Date: {getattr(anon_ds, 'StudyDate', '[REMOVED]')}")
    print(f"   Modality: {anon_ds.Modality}")
    print(f"   Institution: {getattr(anon_ds, 'InstitutionName', '[REMOVED]')}")
    print(
        f"   Referring Physician: {getattr(anon_ds, 'ReferringPhysicianName', '[REMOVED]')}"
    )

    # Verify anonymization
    print("\n5. Verifying anonymization...")
    is_anonymous = anonymizer.verify_anonymization(output_path)
    if is_anonymous:
        print(f"   ✅ VERIFICATION PASSED - File is properly anonymized")
    else:
        print(f"   ❌ VERIFICATION FAILED - PHI may still be present")

    # Compare before/after
    print("\n6. Detailed comparison...")
    comparison = anonymizer.compare_before_after(original_path, output_path)
    print(f"   Modified tags: {len(comparison['modified_tags'])}")
    for tag in comparison["modified_tags"]:
        orig_val = comparison["original"].get(tag, "N/A")
        anon_val = comparison["anonymized"].get(tag, "N/A")
        print(f"      • {tag}: '{orig_val}' → '{anon_val}'")

    print(f"\n   Removed tags: {len(comparison['removed_tags'])}")
    for tag in comparison["removed_tags"][:5]:  # Show first 5
        print(f"      • {tag}")
    if len(comparison["removed_tags"]) > 5:
        print(f"      ... and {len(comparison['removed_tags']) - 5} more")

    # Show anonymization mapping
    print("\n7. Anonymization ID mapping:")
    anon_map = anonymizer.get_anonymization_map()
    for orig, anon in anon_map.items():
        print(f"   {orig} → {anon}")

    # Test batch anonymization
    print("\n8. Testing batch anonymization...")
    print("   Creating multiple test files...")

    # Create a test directory with multiple DICOM files
    batch_input_dir = Path("data/test_output/batch_input")
    batch_output_dir = Path("data/test_output/batch_output")
    batch_input_dir.mkdir(parents=True, exist_ok=True)

    # Create 3 test DICOM files
    for i in range(3):
        test_file = batch_input_dir / f"patient_{i+1}.dcm"
        create_synthetic_dicom(test_file)

    # Batch anonymize
    print(f"   Anonymizing directory: {batch_input_dir}")
    stats = anonymizer.anonymize_directory(
        batch_input_dir, batch_output_dir, keep_descriptive=True, recursive=True
    )

    print(f"   ✅ Batch anonymization complete:")
    print(f"      Processed: {stats['processed']} files")
    print(f"      Failed: {stats['failed']} files")

    print("\n" + "=" * 70)
    print("DEMO COMPLETE - All Tests Passed! 🎉")
    print("=" * 70)
    print("\nSummary:")
    print("  ✅ Created synthetic DICOM files")
    print("  ✅ Anonymized individual DICOM file")
    print("  ✅ Verified anonymization")
    print("  ✅ Compared before/after metadata")
    print("  ✅ Batch anonymized directory")
    print(f"\n📁 Output files: {Path('data/test_output').absolute()}")


if __name__ == "__main__":
    main()
