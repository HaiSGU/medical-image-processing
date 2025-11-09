"""
Medical Image File I/O Module

This module provides utilities for reading and writing medical images
in various formats including NIfTI, DICOM, NRRD, MetaImage, and NumPy.

"""

import logging
import pickle
from pathlib import Path
from typing import Tuple, Dict, Optional, Union
import numpy as np

# Medical imaging libraries
import nibabel as nib
import pydicom
import SimpleITK as sitk

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MedicalImageIO:
    """
    A class for reading and writing medical images in multiple formats.

    Supported formats:
    - NIfTI (.nii, .nii.gz)
    - DICOM (.dcm)
    - NRRD (.nrrd)
    - MetaImage (.mha, .mhd)
    - NumPy (.npy)

    Examples:
        >>> io = MedicalImageIO()
        >>> image, metadata = io.read_image('brain.nii.gz')
        >>> io.write_image(image, 'output.nrrd', metadata)
    """

    def __init__(self):
        """Initialize MedicalImageIO."""
        self.supported_formats = {
            "nifti": [".nii", ".nii.gz"],
            "dicom": [".dcm"],
            "nrrd": [".nrrd"],
            "metaimage": [".mha", ".mhd"],
            "numpy": [".npy"],
        }
        logger.info("MedicalImageIO initialized")

    def read_image(self, file_path: Union[str, Path]) -> Tuple[np.ndarray, Dict]:
        """
        Read a medical image from file.

        Args:
            file_path: Path to the image file

        Returns:
            Tuple of (image_array, metadata_dict)
            - image_array: NumPy array containing image data
            - metadata_dict: Dictionary with image metadata

        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If file format is not supported

        Examples:
            >>> io = MedicalImageIO()
            >>> image, meta = io.read_image('data/mri/brain.nii')
            >>> print(image.shape, meta['spacing'])
        """
        file_path = Path(file_path)

        # Check file exists
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        # Detect format
        file_format = self._detect_format(file_path)
        logger.info(f"Reading {file_format} file: {file_path}")

        # Read based on format
        if file_format == "nifti":
            return self._read_nifti(file_path)
        elif file_format == "dicom":
            return self._read_dicom(file_path)
        elif file_format == "nrrd":
            return self._read_nrrd(file_path)
        elif file_format == "metaimage":
            return self._read_metaimage(file_path)
        elif file_format == "numpy":
            return self._read_numpy(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_path.suffix}")

    def write_image(
        self,
        image: np.ndarray,
        file_path: Union[str, Path],
        metadata: Optional[Dict] = None,
    ) -> None:
        """
        Write a medical image to file.

        Args:
            image: NumPy array containing image data
            file_path: Output file path
            metadata: Optional metadata dictionary

        Raises:
            ValueError: If output format is not supported for writing

        Examples:
            >>> io = MedicalImageIO()
            >>> image = np.random.rand(100, 100, 50)
            >>> metadata = {'spacing': [1.0, 1.0, 2.0]}
            >>> io.write_image(image, 'output.nii.gz', metadata)
        """
        file_path = Path(file_path)
        metadata = metadata or {}

        # Create output directory if needed
        file_path.parent.mkdir(parents=True, exist_ok=True)

        # Detect format
        file_format = self._detect_format(file_path)
        logger.info(f"Writing {file_format} file: {file_path}")

        # Write based on format
        if file_format == "nifti":
            self._write_nifti(image, file_path, metadata)
        elif file_format == "nrrd":
            self._write_nrrd(image, file_path, metadata)
        elif file_format == "metaimage":
            self._write_metaimage(image, file_path, metadata)
        elif file_format == "numpy":
            self._write_numpy(image, file_path)
        else:
            raise ValueError(f"Writing not supported for format: {file_path.suffix}")

        logger.info(f"Successfully wrote image to {file_path}")

    def get_image_info(self, file_path: Union[str, Path]) -> Dict:
        """
        Get metadata information about an image without loading pixel data.

        Args:
            file_path: Path to the image file

        Returns:
            Dictionary containing image metadata

        Examples:
            >>> io = MedicalImageIO()
            >>> info = io.get_image_info('brain.nii.gz')
            >>> print(info['shape'], info['dtype'])
        """
        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        file_format = self._detect_format(file_path)
        logger.info(f"Getting info for {file_format} file: {file_path}")

        # Get basic info
        info = {
            "file_path": str(file_path),
            "file_size_mb": file_path.stat().st_size / (1024 * 1024),
            "format": file_format,
        }

        # Format-specific info
        try:
            if file_format == "nifti":
                img = nib.load(str(file_path))
                info.update(
                    {
                        "shape": img.shape,
                        "dtype": str(img.get_data_dtype()),
                        "affine": img.affine.tolist(),
                        "spacing": img.header.get_zooms(),
                    }
                )
            elif file_format == "dicom":
                ds = pydicom.dcmread(str(file_path))
                info.update(
                    {
                        "shape": (int(ds.Rows), int(ds.Columns)),
                        "patient_id": getattr(ds, "PatientID", "N/A"),
                        "modality": getattr(ds, "Modality", "N/A"),
                        "pixel_spacing": getattr(ds, "PixelSpacing", None),
                    }
                )
            else:
                # Use SimpleITK for other formats
                img = sitk.ReadImage(str(file_path))
                info.update(
                    {
                        "shape": sitk.GetArrayFromImage(img).shape,
                        "spacing": img.GetSpacing(),
                        "origin": img.GetOrigin(),
                        "direction": img.GetDirection(),
                    }
                )
        except Exception as e:
            logger.warning(f"Could not read full metadata: {e}")

        return info

    # ==================== PRIVATE METHODS ====================

    def _detect_format(self, file_path: Path) -> str:
        """Detect image format from file extension."""
        suffix = file_path.suffix.lower()

        # Special case for .nii.gz
        if file_path.name.endswith(".nii.gz"):
            return "nifti"

        for format_name, extensions in self.supported_formats.items():
            if suffix in extensions:
                return format_name

        raise ValueError(f"Unknown file format: {suffix}")

    def _read_nifti(self, file_path: Path) -> Tuple[np.ndarray, Dict]:
        """Read NIfTI format."""
        try:
            img = nib.load(str(file_path))
            image_array = img.get_fdata()

            metadata = {
                "format": "nifti",
                "affine": img.affine,
                "header": dict(img.header),
                "spacing": img.header.get_zooms(),
                "shape": image_array.shape,
                "dtype": str(image_array.dtype),
            }

            logger.info(f"Loaded NIfTI image: shape={image_array.shape}")
            return image_array, metadata
        except Exception as e:
            logger.error(f"Error reading NIfTI file: {e}")
            raise

    def _read_dicom(self, file_path: Path) -> Tuple[np.ndarray, Dict]:
        """Read DICOM format."""
        try:
            ds = pydicom.dcmread(str(file_path))
            image_array = ds.pixel_array.astype(np.float32)

            metadata = {
                "format": "dicom",
                "patient_id": getattr(ds, "PatientID", None),
                "patient_name": str(getattr(ds, "PatientName", None)),
                "modality": getattr(ds, "Modality", None),
                "study_date": getattr(ds, "StudyDate", None),
                "pixel_spacing": getattr(ds, "PixelSpacing", None),
                "slice_thickness": getattr(ds, "SliceThickness", None),
                "shape": image_array.shape,
                "dtype": str(image_array.dtype),
                "dicom_dataset": ds,  # Store full dataset for later use
            }

            logger.info(f"Loaded DICOM image: shape={image_array.shape}")
            return image_array, metadata
        except Exception as e:
            logger.error(f"Error reading DICOM file: {e}")
            raise

    def _read_nrrd(self, file_path: Path) -> Tuple[np.ndarray, Dict]:
        """Read NRRD format using SimpleITK."""
        try:
            # Convert to absolute path with proper encoding for Windows
            abs_path = file_path.resolve()

            # Try reading with different path formats
            try:
                # First try: normal string path
                img = sitk.ReadImage(str(abs_path))
            except Exception as e1:
                logger.warning(f"First attempt failed, trying alternative: {e1}")
                try:
                    # Second try: use forward slashes
                    path_str = str(abs_path).replace("\\", "/")
                    img = sitk.ReadImage(path_str)
                except Exception as e2:
                    logger.warning(f"Second attempt failed, trying bytes: {e2}")
                    # Third try: Read as bytes and save to temp with simple name
                    import tempfile

                    with open(abs_path, "rb") as f:
                        data = f.read()
                    with tempfile.NamedTemporaryFile(
                        delete=False, suffix=".nrrd", mode="wb"
                    ) as tmp:
                        tmp.write(data)
                        tmp_path = tmp.name
                    img = sitk.ReadImage(tmp_path)
                    # Clean up temp file
                    import os

                    try:
                        os.unlink(tmp_path)
                    except:
                        pass

            image_array = sitk.GetArrayFromImage(img)

            metadata = {
                "format": "nrrd",
                "spacing": img.GetSpacing(),
                "origin": img.GetOrigin(),
                "direction": img.GetDirection(),
                "shape": image_array.shape,
                "dtype": str(image_array.dtype),
                "sitk_image": img,  # Store for later use
            }

            logger.info(f"Loaded NRRD image: shape={image_array.shape}")
            return image_array, metadata
        except Exception as e:
            logger.error(f"Error reading NRRD file: {e}")
            raise

    def _read_metaimage(self, file_path: Path) -> Tuple[np.ndarray, Dict]:
        """Read MetaImage format (.mha, .mhd)."""
        try:
            # Convert to absolute path
            abs_path = file_path.resolve()

            # Try different path formats for Windows compatibility
            try:
                img = sitk.ReadImage(str(abs_path))
            except Exception:
                # Try with forward slashes
                path_str = str(abs_path).replace("\\", "/")
                img = sitk.ReadImage(path_str)

            image_array = sitk.GetArrayFromImage(img)

            metadata = {
                "format": "metaimage",
                "spacing": img.GetSpacing(),
                "origin": img.GetOrigin(),
                "direction": img.GetDirection(),
                "shape": image_array.shape,
                "dtype": str(image_array.dtype),
                "sitk_image": img,
            }

            logger.info(f"Loaded MetaImage: shape={image_array.shape}")
            return image_array, metadata
        except Exception as e:
            logger.error(f"Error reading MetaImage file: {e}")
            raise

    def _read_numpy(self, file_path: Path) -> Tuple[np.ndarray, Dict]:
        """Read NumPy format."""
        # Thử đọc không dùng pickle trước (an toàn hơn và nhanh hơn)
        try:
            image_array = np.load(str(file_path), allow_pickle=False)
            logger.info(f"✅ Loaded NumPy array (no pickle): shape={image_array.shape}")
        except (ValueError, pickle.UnpicklingError) as e:
            # File chứa pickled data, thử dùng allow_pickle=True
            logger.warning(f"⚠️ File contains pickled data, using allow_pickle=True")
            try:
                image_array = np.load(str(file_path), allow_pickle=True)
                logger.info(
                    f"✅ Loaded NumPy array (with pickle): shape={image_array.shape}"
                )
            except Exception as e2:
                logger.error(f"❌ Error reading NumPy file even with pickle: {e2}")
                raise

        metadata = {
            "format": "numpy",
            "shape": image_array.shape,
            "dtype": str(image_array.dtype),
        }

        return image_array, metadata

    def _write_nifti(self, image: np.ndarray, file_path: Path, metadata: Dict) -> None:
        """Write NIfTI format."""
        try:
            # Get affine from metadata or create identity
            affine = metadata.get("affine", np.eye(4))

            # Create NIfTI image
            nifti_img = nib.Nifti1Image(image, affine)

            # Set spacing if provided
            if "spacing" in metadata:
                nifti_img.header.set_zooms(metadata["spacing"])

            # Save
            nib.save(nifti_img, str(file_path))
            logger.info(f"Wrote NIfTI file: {file_path}")
        except Exception as e:
            logger.error(f"Error writing NIfTI file: {e}")
            raise

    def _write_nrrd(self, image: np.ndarray, file_path: Path, metadata: Dict) -> None:
        """Write NRRD format."""
        try:
            # Create SimpleITK image
            sitk_img = sitk.GetImageFromArray(image)

            # Set metadata if provided
            if "spacing" in metadata:
                sitk_img.SetSpacing(metadata["spacing"])
            if "origin" in metadata:
                sitk_img.SetOrigin(metadata["origin"])
            if "direction" in metadata:
                sitk_img.SetDirection(metadata["direction"])

            # Write
            sitk.WriteImage(sitk_img, str(file_path))
            logger.info(f"Wrote NRRD file: {file_path}")
        except Exception as e:
            logger.error(f"Error writing NRRD file: {e}")
            raise

    def _write_metaimage(
        self, image: np.ndarray, file_path: Path, metadata: Dict
    ) -> None:
        """Write MetaImage format (.mha/.mhd)."""
        try:
            # Create SimpleITK image
            sitk_img = sitk.GetImageFromArray(image)

            # Set metadata if provided
            if "spacing" in metadata:
                sitk_img.SetSpacing(metadata["spacing"])
            if "origin" in metadata:
                sitk_img.SetOrigin(metadata["origin"])
            if "direction" in metadata:
                sitk_img.SetDirection(metadata["direction"])

            # Write
            sitk.WriteImage(sitk_img, str(file_path))
            logger.info(f"Wrote MetaImage file: {file_path}")
        except Exception as e:
            logger.error(f"Error writing MetaImage file: {e}")
            raise

    def _write_numpy(self, image: np.ndarray, file_path: Path) -> None:
        """Write NumPy format."""
        try:
            np.save(str(file_path), image)
            logger.info(f"Wrote NumPy file: {file_path}")
        except Exception as e:
            logger.error(f"Error writing NumPy file: {e}")
            raise


# Convenience functions
def read_image(file_path: Union[str, Path]) -> Tuple[np.ndarray, Dict]:
    """
    Convenience function to read a medical image.

    Args:
        file_path: Path to the image file

    Returns:
        Tuple of (image_array, metadata_dict)
    """
    io = MedicalImageIO()
    return io.read_image(file_path)


def write_image(
    image: np.ndarray,
    file_path: Union[str, Path],
    metadata: Optional[Dict] = None,
) -> None:
    """
    Convenience function to write a medical image.

    Args:
        image: NumPy array containing image data
        file_path: Output file path
        metadata: Optional metadata dictionary
    """
    io = MedicalImageIO()
    io.write_image(image, file_path, metadata)


def get_image_info(file_path: Union[str, Path]) -> Dict:
    """
    Convenience function to get image metadata.

    Args:
        file_path: Path to the image file

    Returns:
        Dictionary containing image metadata
    """
    io = MedicalImageIO()
    return io.get_image_info(file_path)
