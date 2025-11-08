"""
Image utilities for medical image processing.

This module provides utilities for:
- Format conversions (NumPy ↔ SimpleITK ↔ PIL)
- Coordinate transformations (world ↔ voxel)
- Resampling and resizing
- Normalization and basic operations
"""

import logging
from typing import Tuple, Union, Optional
import numpy as np
import SimpleITK as sitk
from PIL import Image


# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================================
# FORMAT CONVERSIONS
# ============================================================================


def numpy_to_sitk(
    array: np.ndarray,
    spacing: Optional[Tuple[float, ...]] = None,
    origin: Optional[Tuple[float, ...]] = None,
    direction: Optional[Tuple[float, ...]] = None,
) -> sitk.Image:
    """
    Convert NumPy array to SimpleITK image.

    Args:
        array: NumPy array (2D or 3D)
        spacing: Voxel spacing in mm (x, y, z)
        origin: Image origin in mm (x, y, z)
        direction: Direction cosines (9 values for 3D)

    Returns:
        SimpleITK image object

    Examples:
        >>> array = np.random.rand(100, 100, 50)
        >>> spacing = (1.0, 1.0, 2.5)
        >>> sitk_img = numpy_to_sitk(array, spacing=spacing)
    """
    try:
        # Convert to SimpleITK (note: SimpleITK uses Z,Y,X ordering)
        sitk_img = sitk.GetImageFromArray(array)

        # Set metadata if provided
        if spacing is not None:
            sitk_img.SetSpacing(spacing)
        if origin is not None:
            sitk_img.SetOrigin(origin)
        if direction is not None:
            sitk_img.SetDirection(direction)

        logger.info(
            f"Converted NumPy to SimpleITK: {array.shape} -> " f"{sitk_img.GetSize()}"
        )
        return sitk_img

    except Exception as e:
        logger.error(f"Error converting NumPy to SimpleITK: {e}")
        raise


def sitk_to_numpy(image: sitk.Image) -> Tuple[np.ndarray, dict]:
    """
    Convert SimpleITK image to NumPy array with metadata.

    Args:
        image: SimpleITK image object

    Returns:
        Tuple of (array, metadata_dict)

    Examples:
        >>> sitk_img = sitk.ReadImage('image.nii.gz')
        >>> array, metadata = sitk_to_numpy(sitk_img)
        >>> print(array.shape, metadata['spacing'])
    """
    try:
        # Convert to NumPy
        array = sitk.GetArrayFromImage(image)

        # Extract metadata
        metadata = {
            "spacing": image.GetSpacing(),
            "origin": image.GetOrigin(),
            "direction": image.GetDirection(),
            "size": image.GetSize(),
            "pixel_type": image.GetPixelIDTypeAsString(),
        }

        logger.info(
            f"Converted SimpleITK to NumPy: {metadata['size']} -> {array.shape}"
        )
        return array, metadata

    except Exception as e:
        logger.error(f"Error converting SimpleITK to NumPy: {e}")
        raise


def numpy_to_pil(array: np.ndarray, mode: str = "L") -> Image.Image:
    """
    Convert NumPy array to PIL Image (for 2D slices).

    Args:
        array: 2D NumPy array
        mode: PIL image mode ('L' for grayscale, 'RGB' for color)

    Returns:
        PIL Image object

    Examples:
        >>> slice_2d = volume[50, :, :]
        >>> pil_img = numpy_to_pil(slice_2d)
        >>> pil_img.save('slice.png')
    """
    try:
        if array.ndim != 2:
            raise ValueError(f"Expected 2D array, got shape {array.shape}")

        # Normalize to 0-255 range
        array_normalized = normalize_array(array, method="min-max")
        array_uint8 = (array_normalized * 255).astype(np.uint8)

        # Convert to PIL
        pil_img = Image.fromarray(array_uint8, mode=mode)

        logger.info(f"Converted NumPy to PIL: {array.shape}")
        return pil_img

    except Exception as e:
        logger.error(f"Error converting NumPy to PIL: {e}")
        raise


def pil_to_numpy(image: Image.Image) -> np.ndarray:
    """
    Convert PIL Image to NumPy array.

    Args:
        image: PIL Image object

    Returns:
        NumPy array

    Examples:
        >>> pil_img = Image.open('slice.png')
        >>> array = pil_to_numpy(pil_img)
    """
    try:
        array = np.array(image)
        logger.info(f"Converted PIL to NumPy: {image.size} -> {array.shape}")
        return array

    except Exception as e:
        logger.error(f"Error converting PIL to NumPy: {e}")
        raise


# ============================================================================
# COORDINATE TRANSFORMATIONS
# ============================================================================


def voxel_to_world(
    voxel_coords: Union[Tuple[int, ...], np.ndarray],
    spacing: Tuple[float, ...],
    origin: Tuple[float, ...],
    direction: Optional[Tuple[float, ...]] = None,
) -> np.ndarray:
    """
    Convert voxel coordinates to world (physical) coordinates.

    Args:
        voxel_coords: Voxel indices (x, y, z)
        spacing: Voxel spacing in mm
        origin: Image origin in mm
        direction: Direction cosines (optional, defaults to identity)

    Returns:
        World coordinates in mm

    Examples:
        >>> voxel = (50, 60, 25)
        >>> spacing = (1.0, 1.0, 2.5)
        >>> origin = (0, 0, 0)
        >>> world = voxel_to_world(voxel, spacing, origin)
        >>> print(world)  # [50.0, 60.0, 62.5]
    """
    try:
        voxel = np.array(voxel_coords, dtype=float)
        spacing = np.array(spacing)
        origin = np.array(origin)

        # Simple case: no rotation (identity direction)
        if direction is None:
            world = voxel * spacing + origin
        else:
            # With rotation matrix
            direction = np.array(direction).reshape(3, 3)
            world = direction @ (voxel * spacing) + origin

        return world

    except Exception as e:
        logger.error(f"Error in voxel_to_world: {e}")
        raise


def world_to_voxel(
    world_coords: Union[Tuple[float, ...], np.ndarray],
    spacing: Tuple[float, ...],
    origin: Tuple[float, ...],
    direction: Optional[Tuple[float, ...]] = None,
) -> np.ndarray:
    """
    Convert world (physical) coordinates to voxel coordinates.

    Args:
        world_coords: World coordinates in mm (x, y, z)
        spacing: Voxel spacing in mm
        origin: Image origin in mm
        direction: Direction cosines (optional, defaults to identity)

    Returns:
        Voxel indices (may be fractional, round as needed)

    Examples:
        >>> world = (50.0, 60.0, 62.5)
        >>> spacing = (1.0, 1.0, 2.5)
        >>> origin = (0, 0, 0)
        >>> voxel = world_to_voxel(world, spacing, origin)
        >>> print(voxel)  # [50.0, 60.0, 25.0]
    """
    try:
        world = np.array(world_coords, dtype=float)
        spacing = np.array(spacing)
        origin = np.array(origin)

        # Simple case: no rotation
        if direction is None:
            voxel = (world - origin) / spacing
        else:
            # With rotation matrix
            direction = np.array(direction).reshape(3, 3)
            voxel = (np.linalg.inv(direction) @ (world - origin)) / spacing

        return voxel

    except Exception as e:
        logger.error(f"Error in world_to_voxel: {e}")
        raise


# ============================================================================
# RESAMPLING
# ============================================================================


def resample_image(
    image: sitk.Image,
    new_spacing: Optional[Tuple[float, ...]] = None,
    new_size: Optional[Tuple[int, ...]] = None,
    interpolator: int = sitk.sitkLinear,
) -> sitk.Image:
    """
    Resample image to new spacing or size.

    Args:
        image: SimpleITK image
        new_spacing: Target spacing in mm (mutually exclusive with new_size)
        new_size: Target size in voxels (mutually exclusive with new_spacing)
        interpolator: Interpolation method (sitkLinear, sitkNearestNeighbor, etc.)

    Returns:
        Resampled SimpleITK image

    Examples:
        >>> # Resample to isotropic 1mm spacing
        >>> resampled = resample_image(image, new_spacing=(1.0, 1.0, 1.0))
        >>> # Resize to specific dimensions
        >>> resized = resample_image(image, new_size=(256, 256, 128))
    """
    try:
        original_spacing = image.GetSpacing()
        original_size = image.GetSize()

        # Calculate new size or spacing
        if new_spacing is not None and new_size is None:
            # Calculate size from spacing
            new_spacing = np.array(new_spacing)
            original_spacing = np.array(original_spacing)
            original_size = np.array(original_size)
            new_size_array = np.round(
                original_size * original_spacing / new_spacing
            ).astype(int)
            new_size = tuple(new_size_array.tolist())
        elif new_size is not None and new_spacing is None:
            # Calculate spacing from size
            new_size = np.array(new_size)
            original_size = np.array(original_size)
            original_spacing = np.array(original_spacing)
            new_spacing = tuple((original_spacing * original_size / new_size).tolist())
        elif new_spacing is not None and new_size is not None:
            raise ValueError("Specify either new_spacing or new_size, not both")
        else:
            raise ValueError("Must specify either new_spacing or new_size")

        # Setup resampler
        resampler = sitk.ResampleImageFilter()
        resampler.SetSize([int(s) for s in new_size])
        resampler.SetOutputSpacing(new_spacing)
        resampler.SetOutputOrigin(image.GetOrigin())
        resampler.SetOutputDirection(image.GetDirection())
        resampler.SetInterpolator(interpolator)
        resampler.SetDefaultPixelValue(0)

        # Resample
        resampled = resampler.Execute(image)

        logger.info(
            f"Resampled: {original_size} @ {tuple(np.round(original_spacing, 2))}mm "
            f"-> {new_size} @ {tuple(np.round(new_spacing, 2))}mm"
        )

        return resampled

    except Exception as e:
        logger.error(f"Error resampling image: {e}")
        raise


def resize_array(
    array: np.ndarray,
    new_shape: Tuple[int, ...],
    order: int = 1,
) -> np.ndarray:
    """
    Resize NumPy array using scipy interpolation.

    Args:
        array: Input array
        new_shape: Target shape
        order: Interpolation order (0=nearest, 1=linear, 3=cubic)

    Returns:
        Resized array

    Examples:
        >>> array = np.random.rand(100, 100, 50)
        >>> resized = resize_array(array, (200, 200, 100))
    """
    try:
        from scipy import ndimage

        # Calculate zoom factors
        zoom_factors = np.array(new_shape) / np.array(array.shape)

        # Resize
        resized = ndimage.zoom(array, zoom_factors, order=order)

        logger.info(f"Resized array: {array.shape} -> {resized.shape}")
        return resized

    except ImportError:
        logger.error("scipy is required for resize_array")
        raise
    except Exception as e:
        logger.error(f"Error resizing array: {e}")
        raise


# ============================================================================
# NORMALIZATION
# ============================================================================


def normalize_array(
    array: np.ndarray,
    method: str = "min-max",
    clip_percentile: Optional[Tuple[float, float]] = None,
) -> np.ndarray:
    """
    Normalize array intensities.

    Args:
        array: Input array
        method: Normalization method ('min-max', 'z-score', 'percentile')
        clip_percentile: Percentile clipping range (e.g., (1, 99))

    Returns:
        Normalized array

    Examples:
        >>> # Min-max normalization to [0, 1]
        >>> normalized = normalize_array(array, method='min-max')
        >>> # Z-score normalization
        >>> standardized = normalize_array(array, method='z-score')
        >>> # Percentile clipping
        >>> clipped = normalize_array(array, method='percentile', clip_percentile=(1, 99))
    """
    try:
        array = array.astype(float)

        # Clip percentiles if specified
        if clip_percentile is not None:
            low, high = clip_percentile
            p_low = np.percentile(array, low)
            p_high = np.percentile(array, high)
            array = np.clip(array, p_low, p_high)

        # Normalize
        if method == "min-max":
            min_val = array.min()
            max_val = array.max()
            if max_val > min_val:
                normalized = (array - min_val) / (max_val - min_val)
            else:
                normalized = np.zeros_like(array)

        elif method == "z-score":
            mean = array.mean()
            std = array.std()
            if std > 0:
                normalized = (array - mean) / std
            else:
                normalized = array - mean

        elif method == "percentile":
            if clip_percentile is None:
                clip_percentile = (0, 100)
            low, high = clip_percentile
            p_low = np.percentile(array, low)
            p_high = np.percentile(array, high)
            if p_high > p_low:
                normalized = (array - p_low) / (p_high - p_low)
                normalized = np.clip(normalized, 0, 1)
            else:
                normalized = np.zeros_like(array)

        else:
            raise ValueError(f"Unknown normalization method: {method}")

        logger.info(
            f"Normalized array using {method}: range [{normalized.min():.3f}, {normalized.max():.3f}]"
        )
        return normalized

    except Exception as e:
        logger.error(f"Error normalizing array: {e}")
        raise


# ============================================================================
# BASIC OPERATIONS
# ============================================================================


def crop_to_nonzero(array: np.ndarray, margin: int = 0) -> Tuple[np.ndarray, Tuple]:
    """
    Crop array to non-zero bounding box.

    Args:
        array: Input array
        margin: Margin to add around bounding box (in voxels)

    Returns:
        Tuple of (cropped_array, bounding_box)
        bounding_box format: ((z_min, z_max), (y_min, y_max), (x_min, x_max))

    Examples:
        >>> cropped, bbox = crop_to_nonzero(array, margin=5)
        >>> print(bbox)
    """
    try:
        # Find non-zero coordinates
        coords = np.argwhere(array != 0)

        if len(coords) == 0:
            logger.warning("Array is all zeros, returning original")
            return array, tuple((0, s) for s in array.shape)

        # Get bounding box
        mins = coords.min(axis=0)
        maxs = coords.max(axis=0)

        # Add margin
        shape = array.shape
        bbox = []
        for i, (min_val, max_val) in enumerate(zip(mins, maxs)):
            min_clip = max(0, min_val - margin)
            max_clip = min(shape[i], max_val + margin + 1)
            bbox.append((min_clip, max_clip))

        # Crop
        if array.ndim == 3:
            cropped = array[
                bbox[0][0] : bbox[0][1],
                bbox[1][0] : bbox[1][1],
                bbox[2][0] : bbox[2][1],
            ]
        elif array.ndim == 2:
            cropped = array[bbox[0][0] : bbox[0][1], bbox[1][0] : bbox[1][1]]
        else:
            raise ValueError(f"Unsupported array dimension: {array.ndim}")

        logger.info(f"Cropped to non-zero: {array.shape} -> {cropped.shape}")
        return cropped, tuple(bbox)

    except Exception as e:
        logger.error(f"Error cropping to non-zero: {e}")
        raise


def pad_array(
    array: np.ndarray,
    target_shape: Tuple[int, ...],
    mode: str = "constant",
    constant_value: float = 0,
) -> np.ndarray:
    """
    Pad array to target shape.

    Args:
        array: Input array
        target_shape: Desired output shape
        mode: Padding mode ('constant', 'edge', 'reflect', 'symmetric')
        constant_value: Value for constant padding

    Returns:
        Padded array

    Examples:
        >>> padded = pad_array(array, (200, 200, 100), mode='constant')
    """
    try:
        current_shape = array.shape

        # Calculate padding
        pad_width = []
        for current, target in zip(current_shape, target_shape):
            if target < current:
                raise ValueError(
                    f"Target shape {target_shape} smaller than current "
                    f"{current_shape}"
                )
            total_pad = target - current
            pad_before = total_pad // 2
            pad_after = total_pad - pad_before
            pad_width.append((pad_before, pad_after))

        # Pad
        if mode == "constant":
            padded = np.pad(array, pad_width, mode=mode, constant_values=constant_value)
        else:
            padded = np.pad(array, pad_width, mode=mode)

        logger.info(f"Padded array: {array.shape} -> {padded.shape}")
        return padded

    except Exception as e:
        logger.error(f"Error padding array: {e}")
        raise


def extract_slice(
    array: np.ndarray,
    slice_index: int,
    axis: int = 0,
) -> np.ndarray:
    """
    Extract 2D slice from 3D volume.

    Args:
        array: 3D array
        slice_index: Index of slice to extract
        axis: Axis along which to slice (0=axial, 1=coronal, 2=sagittal)

    Returns:
        2D slice

    Examples:
        >>> # Extract middle axial slice
        >>> mid_slice = extract_slice(volume, volume.shape[0]//2, axis=0)
    """
    try:
        if array.ndim != 3:
            raise ValueError(f"Expected 3D array, got shape {array.shape}")

        if axis == 0:
            slice_2d = array[slice_index, :, :]
        elif axis == 1:
            slice_2d = array[:, slice_index, :]
        elif axis == 2:
            slice_2d = array[:, :, slice_index]
        else:
            raise ValueError(f"Invalid axis: {axis}")

        return slice_2d

    except Exception as e:
        logger.error(f"Error extracting slice: {e}")
        raise
