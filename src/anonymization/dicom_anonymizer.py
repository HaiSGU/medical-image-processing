"""
DICOM Anonymization Module

This module provides utilities for anonymizing DICOM files by removing
Protected Health Information (PHI) and replacing with anonymous identifiers.

Author: HaiSGU
Date: 2025-10-27
"""

import logging
from pathlib import Path
from typing import Dict, Union
import hashlib

import pydicom
from pydicom.dataset import Dataset

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DICOMAnonymizer:
    """
    A class for anonymizing DICOM files.

    This class removes Protected Health Information (PHI) from DICOM files
    and replaces patient identifiers with anonymized versions.

    Examples:
        >>> anonymizer = DICOMAnonymizer()
        >>> anonymizer.anonymize_file('input.dcm', 'output.dcm')
        >>> anonymizer.anonymize_directory('input_dir/', 'output_dir/')
    """

    # Tags to remove (PHI - Protected Health Information)
    PHI_TAGS = [
        "PatientName",
        "PatientID",
        "PatientBirthDate",
        "PatientSex",
        "PatientAge",
        "PatientSize",
        "PatientWeight",
        "PatientAddress",
        "InstitutionName",
        "InstitutionAddress",
        "ReferringPhysicianName",
        "PerformingPhysicianName",
        "OperatorsName",
        "StudyDate",
        "StudyTime",
        "SeriesDate",
        "SeriesTime",
        "AcquisitionDate",
        "AcquisitionTime",
        "ContentDate",
        "ContentTime",
        "InstanceCreationDate",
        "InstanceCreationTime",
    ]

    def __init__(self, prefix: str = "ANON", **kwargs):
        """
        Initialize DICOM Anonymizer.

        Args:
            prefix: Prefix for anonymous IDs (default: "ANON")
        """
        if "patient_id_prefix" in kwargs and kwargs["patient_id_prefix"]:
            prefix = str(kwargs["patient_id_prefix"])

        self.prefix = prefix
        # Track original IDs that have been anonymized
        self.anonymization_map = {}
        logger.info("DICOMAnonymizer initialized with prefix: %s", prefix)

    def _generate_anonymous_id(self, original_id: str) -> str:
        """
        Generate anonymous ID from original ID using hash.

        Args:
            original_id: Original patient ID

        Returns:
            Anonymous ID string
        """
        # Use SHA256 hash for consistent anonymization
        hash_obj = hashlib.sha256(original_id.encode())
        hash_hex = hash_obj.hexdigest()[:8]  # Take first 8 characters
        anonymous_id = f"{self.prefix}_{hash_hex}"

        # Store mapping
        self.anonymization_map[original_id] = anonymous_id

        return anonymous_id

    def anonymize_dataset(
        self, dataset: Dataset, keep_descriptive: bool = True
    ) -> Dataset:
        """
        Anonymize a DICOM dataset.

        Args:
            dataset: PyDICOM dataset
            keep_descriptive: Keep descriptive tags like Modality or
                StudyDescription

        Returns:
            Anonymized dataset
        """
        # Get original patient ID for mapping
        original_patient_id = str(getattr(dataset, "PatientID", "UNKNOWN"))

        # Generate anonymous ID
        if original_patient_id not in self.anonymization_map:
            anonymous_id = self._generate_anonymous_id(original_patient_id)
        else:
            anonymous_id = self.anonymization_map[original_patient_id]

        # Remove PHI tags
        for tag in self.PHI_TAGS:
            if hasattr(dataset, tag):
                # Special handling for PatientID and PatientName
                if tag == "PatientID":
                    dataset.PatientID = anonymous_id
                elif tag == "PatientName":
                    dataset.PatientName = anonymous_id
                else:
                    # Remove other PHI
                    delattr(dataset, tag)

        # Anonymize dates (set to a fixed date or remove)
        if hasattr(dataset, "StudyDate"):
            dataset.StudyDate = "20000101"  # Generic date
        if hasattr(dataset, "SeriesDate"):
            dataset.SeriesDate = "20000101"

        logger.info(
            "Anonymized dataset: %s -> %s",
            original_patient_id,
            anonymous_id,
        )

        return dataset

    def anonymize_file(
        self,
        input_path: Union[str, Path],
        output_path: Union[str, Path],
        keep_descriptive: bool = True,
    ) -> None:
        """
        Anonymize a single DICOM file.

        Args:
            input_path: Path to input DICOM file
            output_path: Path to output anonymized file
            keep_descriptive: Keep descriptive tags

        Raises:
            FileNotFoundError: If input file doesn't exist

        Examples:
            >>> anonymizer = DICOMAnonymizer()
            >>> anonymizer.anonymize_file('patient.dcm', 'anon.dcm')
        """
        input_path = Path(input_path)
        output_path = Path(output_path)

        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")

        # Create output directory if needed
        output_path.parent.mkdir(parents=True, exist_ok=True)

        logger.info("Anonymizing %s -> %s", input_path, output_path)

        try:
            # Read DICOM
            dataset = pydicom.dcmread(str(input_path))

            # Anonymize
            anonymized_dataset = self.anonymize_dataset(
                dataset,
                keep_descriptive,
            )

            # Save
            anonymized_dataset.save_as(str(output_path))

            logger.info("Successfully anonymized %s", output_path)

        except Exception as e:
            logger.error("Error anonymizing file: %s", e)
            raise

    def anonymize_directory(
        self,
        input_dir: Union[str, Path],
        output_dir: Union[str, Path],
        keep_descriptive: bool = True,
        recursive: bool = True,
    ) -> Dict[str, Union[int, Dict[str, str]]]:
        """
        Anonymize all DICOM files in a directory.

        Args:
            input_dir: Input directory path
            output_dir: Output directory path
            keep_descriptive: Keep descriptive tags
            recursive: Process subdirectories recursively

        Returns:
            Dictionary with statistics:
                {
                    'processed': N,
                    'failed': M,
                    'successful': N,
                    'id_mapping': {original: anonymized, ...}
                }

        Examples:
            >>> anonymizer = DICOMAnonymizer()
            >>> stats = anonymizer.anonymize_directory('input/', 'output/')
            >>> print(
            ...     "Processed: {processed}, Failed: {failed}".format(
            ...         processed=stats['processed'],
            ...         failed=stats['failed'],
            ...     )
            ... )
        """
        input_dir = Path(input_dir)
        output_dir = Path(output_dir)

        if not input_dir.exists():
            raise FileNotFoundError(f"Input directory not found: {input_dir}")

        output_dir.mkdir(parents=True, exist_ok=True)

        stats = {"processed": 0, "failed": 0}

        # Find all DICOM files
        if recursive:
            dicom_files = list(input_dir.rglob("*.dcm"))
        else:
            dicom_files = list(input_dir.glob("*.dcm"))

        logger.info("Found %d DICOM files to anonymize", len(dicom_files))

        for input_file in dicom_files:
            # Compute relative path
            rel_path = input_file.relative_to(input_dir)
            output_file = output_dir / rel_path

            try:
                self.anonymize_file(input_file, output_file, keep_descriptive)
                stats["processed"] += 1
            except Exception as e:
                logger.error("Failed to anonymize %s: %s", input_file, e)
                stats["failed"] += 1

        result = {
            "processed": stats["processed"],
            "failed": stats["failed"],
            "successful": stats["processed"],
            "id_mapping": self.get_anonymization_map(),
        }

        logger.info("Batch anonymization complete: %s", result)
        return result

    def verify_anonymization(self, file_path: Union[str, Path]) -> bool:
        """
        Verify that a DICOM file has been properly anonymized.

        Args:
            file_path: Path to DICOM file to verify

        Returns:
            True if file is anonymized, False otherwise

        Examples:
            >>> anonymizer = DICOMAnonymizer()
            >>> is_anonymous = anonymizer.verify_anonymization('anon.dcm')
        """
        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        try:
            dataset = pydicom.dcmread(str(file_path))

            # Check if PHI tags are present
            phi_found = []
            for tag in self.PHI_TAGS:
                if hasattr(dataset, tag):
                    value = getattr(dataset, tag)
                    # Allow anonymous IDs for PatientID and PatientName
                    if tag in ["PatientID", "PatientName"]:
                        if not str(value).startswith(self.prefix):
                            phi_found.append(f"{tag}={value}")
                    else:
                        phi_found.append(f"{tag}={value}")

            if phi_found:
                logger.warning("PHI found in %s: %s", file_path, phi_found)
                return False
            else:
                logger.info("File is properly anonymized: %s", file_path)
                return True

        except Exception as e:
            logger.error("Error verifying file: %s", e)
            return False

    def get_anonymization_map(self) -> Dict[str, str]:
        """
        Get the mapping of original IDs to anonymous IDs.

        Returns:
            Dictionary mapping original patient IDs to anonymous IDs
        """
        return self.anonymization_map.copy()

    def compare_before_after(
        self,
        original_path: Union[str, Path],
        anonymized_path: Union[str, Path],
    ) -> Dict:
        """
        Compare original and anonymized DICOM files.

        Args:
            original_path: Path to original DICOM
            anonymized_path: Path to anonymized DICOM

        Returns:
            Dictionary with comparison results
        """
        original = pydicom.dcmread(str(original_path))
        anonymized = pydicom.dcmread(str(anonymized_path))

        comparison = {
            "original": {},
            "anonymized": {},
            "removed_tags": [],
            "modified_tags": [],
        }

        # Check PHI tags
        for tag in self.PHI_TAGS:
            if hasattr(original, tag):
                original_value = getattr(original, tag)
                comparison["original"][tag] = str(original_value)

                if hasattr(anonymized, tag):
                    anonymized_value = getattr(anonymized, tag)
                    comparison["anonymized"][tag] = str(anonymized_value)

                    if original_value != anonymized_value:
                        comparison["modified_tags"].append(tag)
                else:
                    comparison["anonymized"][tag] = "[REMOVED]"
                    comparison["removed_tags"].append(tag)

        return comparison


# Convenience function
def anonymize_dicom(
    input_path: Union[str, Path],
    output_path: Union[str, Path],
    prefix: str = "ANON",
) -> None:
    """
    Convenience function to anonymize a single DICOM file.

    Args:
        input_path: Path to input DICOM
        output_path: Path to output DICOM
        prefix: Prefix for anonymous ID
    """
    anonymizer = DICOMAnonymizer(prefix=prefix)
    anonymizer.anonymize_file(input_path, output_path)
