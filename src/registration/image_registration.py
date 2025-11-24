"""
Image Registration Module

Implements medical image registration using SimpleITK.
Supports rigid, affine, and deformable registration methods.

Author: HaiSGU
Date: 2025-11-23
"""

import numpy as np
import SimpleITK as sitk
import logging
from typing import Tuple, Optional, Dict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ImageRegistration:
    """
    Medical Image Registration using SimpleITK.

    Supports multiple registration types:
    - Rigid: Translation + rotation only
    - Affine: + scaling + shearing
    - Deformable: B-spline based non-rigid registration

    Attributes:
        fixed_image: Reference image (SimpleITK.Image)
        moving_image: Image to be registered (SimpleITK.Image)
        transform: Final transform after registration
        metric_values: List of metric values during optimization

    Examples:
        >>> # Register two brain MRI scans
        >>> fixed = sitk.ReadImage('brain_t1.nii')
        >>> moving = sitk.ReadImage('brain_t2.nii')
        >>>
        >>> registration = ImageRegistration(fixed, moving)
        >>> registered_image = registration.rigid_registration()
        >>>
        >>> # Get registration metrics
        >>> print(registration.get_metrics())
    """

    def __init__(
        self, fixed_image: sitk.Image, moving_image: sitk.Image, verbose: bool = True
    ):
        """
        Initialize Image Registration.

        Args:
            fixed_image: Reference/target image (SimpleITK format)
            moving_image: Image to align to fixed (SimpleITK format)
            verbose: Print progress messages

        Example:
            >>> fixed = sitk.ReadImage('fixed.nii.gz')
            >>> moving = sitk.ReadImage('moving.nii.gz')
            >>> reg = ImageRegistration(fixed, moving)
        """
        self.fixed_image = fixed_image
        self.moving_image = moving_image
        self.transform = None
        self.metric_values = []
        self.verbose = verbose

        if self.verbose:
            logger.info("ImageRegistration initialized")
            logger.info(f"  Fixed image size: {fixed_image.GetSize()}")
            logger.info(f"  Moving image size: {moving_image.GetSize()}")

    def _setup_registration_method(
        self,
        transform: sitk.Transform,
        metric: str = "mean_squares",
        sampling_percentage: float = 0.1,
        number_of_iterations: int = 100,
        learning_rate: float = 1.0,
        min_step: float = 0.001,
    ) -> sitk.ImageRegistrationMethod:
        """
        Setup registration method with common parameters.

        Args:
            transform: Initial transform
            metric: Similarity metric ('mean_squares' or 'mutual_information')
            sampling_percentage: Percentage of pixels to sample
            number_of_iterations: Max optimizer iterations
            learning_rate: Gradient descent learning rate
            min_step: Minimum step length

        Returns:
            Configured ImageRegistrationMethod
        """
        registration_method = sitk.ImageRegistrationMethod()

        # Metric
        if metric == "mean_squares":
            registration_method.SetMetricAsMeanSquares()
        elif metric == "mutual_information":
            registration_method.SetMetricAsMattesMutualInformation(
                numberOfHistogramBins=50
            )
        else:
            raise ValueError(f"Unknown metric: {metric}")

        # Sampling strategy
        registration_method.SetMetricSamplingStrategy(registration_method.RANDOM)
        registration_method.SetMetricSamplingPercentage(sampling_percentage)

        # Interpolator
        registration_method.SetInterpolator(sitk.sitkLinear)

        # Optimizer
        registration_method.SetOptimizerAsRegularStepGradientDescent(
            learningRate=learning_rate,
            minStep=min_step,
            numberOfIterations=number_of_iterations,
            estimateLearningRate=registration_method.Once,
        )

        # Setup multi-resolution if needed
        registration_method.SetShrinkFactorsPerLevel(shrinkFactors=[4, 2, 1])
        registration_method.SetSmoothingSigmasPerLevel(smoothingSigmas=[2, 1, 0])
        registration_method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

        # Set initial transform
        registration_method.SetInitialTransform(transform, inPlace=False)

        # Observer to collect metrics
        self.metric_values = []

        def metric_observer(method):
            self.metric_values.append(method.GetMetricValue())
            if self.verbose and len(self.metric_values) % 10 == 0:
                logger.info(
                    f"  Iteration {len(self.metric_values)}: "
                    f"Metric = {self.metric_values[-1]:.4f}"
                )

        registration_method.AddCommand(
            sitk.sitkIterationEvent, lambda: metric_observer(registration_method)
        )

        return registration_method

    def rigid_registration(
        self,
        number_of_iterations: int = 100,
        learning_rate: float = 1.0,
        metric: str = "mean_squares",
    ) -> sitk.Image:
        """
        Rigid registration (translation + rotation only).

        GIẢI THÍCH:
        -----------
        Rigid registration chỉ cho phép di chuyển (translation) và
        xoay (rotation), KHÔNG thay đổi kích thước hoặc hình dạng.

        Phù hợp cho:
        - Multi-modal brain MRI (T1, T2)
        - Follow-up scans của cùng bệnh nhân
        - Head motion correction

        Args:
            number_of_iterations: Max iterations
            learning_rate: Optimizer step size
            metric: 'mean_squares' hoặc 'mutual_information'

        Returns:
            Registered image (SimpleITK.Image)

        Example:
            >>> reg = ImageRegistration(fixed, moving)
            >>> registered = reg.rigid_registration(number_of_iterations=200)
        """
        logger.info("Starting rigid registration...")

        # Initialize transform
        # Euler3DTransform: 3 rotations + 3 translations = 6 DOF
        initial_transform = sitk.CenteredTransformInitializer(
            self.fixed_image,
            self.moving_image,
            sitk.Euler3DTransform(),
            sitk.CenteredTransformInitializerFilter.GEOMETRY,
        )

        # Setup registration
        registration_method = self._setup_registration_method(
            transform=initial_transform,
            metric=metric,
            number_of_iterations=number_of_iterations,
            learning_rate=learning_rate,
        )

        # Execute registration
        self.transform = registration_method.Execute(
            self.fixed_image, self.moving_image
        )

        logger.info(f"Rigid registration complete!")
        logger.info(f"  Final metric: {registration_method.GetMetricValue():.4f}")
        logger.info(
            f"  Stop condition: {registration_method.GetOptimizerStopConditionDescription()}"
        )

        # Resample moving image
        registered_image = sitk.Resample(
            self.moving_image,
            self.fixed_image,
            self.transform,
            sitk.sitkLinear,
            0.0,
            self.moving_image.GetPixelID(),
        )

        return registered_image

    def affine_registration(
        self,
        number_of_iterations: int = 100,
        learning_rate: float = 1.0,
        metric: str = "mean_squares",
    ) -> sitk.Image:
        """
        Affine registration (translation, rotation, scaling, shearing).

        GIẢI THÍCH:
        -----------
        Affine registration cho phép:
        - Translation (di chuyển)
        - Rotation (xoay)
        - Scaling (thay đổi kích thước)
        - Shearing (nghiêng)

        Phù hợp cho:
        - Inter-subject registration (người khác nhau)
        - CT-MRI registration
        - Atlas registration

        Args:
            number_of_iterations: Max iterations
            learning_rate: Optimizer step size
            metric: 'mean_squares' hoặc 'mutual_information'

        Returns:
            Registered image

        Example:
            >>> registered = reg.affine_registration(number_of_iterations=200)
        """
        logger.info("Starting affine registration...")

        # Initialize transform
        # AffineTransform: 12 DOF (3x3 matrix + 3 translation)
        initial_transform = sitk.CenteredTransformInitializer(
            self.fixed_image,
            self.moving_image,
            sitk.AffineTransform(3),
            sitk.CenteredTransformInitializerFilter.GEOMETRY,
        )

        # Setup registration
        registration_method = self._setup_registration_method(
            transform=initial_transform,
            metric=metric,
            number_of_iterations=number_of_iterations,
            learning_rate=learning_rate,
        )

        # Execute
        self.transform = registration_method.Execute(
            self.fixed_image, self.moving_image
        )

        logger.info(f"Affine registration complete!")
        logger.info(f"  Final metric: {registration_method.GetMetricValue():.4f}")

        # Resample
        registered_image = sitk.Resample(
            self.moving_image,
            self.fixed_image,
            self.transform,
            sitk.sitkLinear,
            0.0,
            self.moving_image.GetPixelID(),
        )

        return registered_image

    def deformable_registration(
        self,
        number_of_iterations: int = 50,
        mesh_size: int = 5,
        metric: str = "mean_squares",
    ) -> sitk.Image:
        """
        Deformable (non-rigid) registration using B-spline.

        GIẢI THÍCH:
        -----------
        Deformable registration cho phép biến dạng cục bộ (local deformation).
        Mỗi vùng nhỏ có thể di chuyển độc lập.

        Phù hợp cho:
        - Tumor growth tracking
        - Breathing motion compensation
        - Brain deformation due to mass effect

        Warning: Computationally expensive!

        Args:
            number_of_iterations: Max iterations
            mesh_size: B-spline mesh size (smaller = more flexible)
            metric: Similarity metric

        Returns:
            Registered image

        Example:
            >>> # Deformable registration (chậm hơn)
            >>> registered = reg.deformable_registration(mesh_size=8)
        """
        logger.info("Starting deformable registration (B-spline)...")
        logger.warning("  This may take several minutes...")

        # Initialize transform grid
        transform_domain_mesh_size = [mesh_size] * self.fixed_image.GetDimension()
        initial_transform = sitk.BSplineTransformInitializer(
            self.fixed_image, transform_domain_mesh_size
        )

        # Setup registration
        registration_method = self._setup_registration_method(
            transform=initial_transform,
            metric=metric,
            number_of_iterations=number_of_iterations,
            learning_rate=1.0,
            min_step=0.001,
        )

        # Optimizer specific for B-spline
        registration_method.SetOptimizerAsLBFGSB(
            gradientConvergenceTolerance=1e-5, numberOfIterations=number_of_iterations
        )

        # Execute
        self.transform = registration_method.Execute(
            self.fixed_image, self.moving_image
        )

        logger.info(f"Deformable registration complete!")
        logger.info(f"  Final metric: {registration_method.GetMetricValue():.4f}")

        # Resample
        registered_image = sitk.Resample(
            self.moving_image,
            self.fixed_image,
            self.transform,
            sitk.sitkLinear,
            0.0,
            self.moving_image.GetPixelID(),
        )

        return registered_image

    def get_transform(self) -> Optional[sitk.Transform]:
        """Get final transform after registration."""
        return self.transform

    def get_metric_values(self) -> list:
        """Get list of metric values during optimization."""
        return self.metric_values

    def get_metrics(self) -> Dict[str, float]:
        """
        Calculate registration quality metrics.

        Returns:
            Dictionary with metric values
        """
        if self.transform is None:
            logger.warning("No registration performed yet")
            return {}

        # Resample moving image
        registered = sitk.Resample(
            self.moving_image,
            self.fixed_image,
            self.transform,
            sitk.sitkLinear,
            0.0,
            self.moving_image.GetPixelID(),
        )

        # Convert to numpy for metrics
        fixed_array = sitk.GetArrayFromImage(self.fixed_image)
        moving_array = sitk.GetArrayFromImage(self.moving_image)
        registered_array = sitk.GetArrayFromImage(registered)

        # Mean Squared Error
        mse_before = np.mean((fixed_array - moving_array) ** 2)
        mse_after = np.mean((fixed_array - registered_array) ** 2)

        # Normalized Cross Correlation
        def ncc(img1, img2):
            img1_normalized = (img1 - np.mean(img1)) / (np.std(img1) + 1e-8)
            img2_normalized = (img2 - np.mean(img2)) / (np.std(img2) + 1e-8)
            return np.mean(img1_normalized * img2_normalized)

        ncc_before = ncc(fixed_array, moving_array)
        ncc_after = ncc(fixed_array, registered_array)

        metrics = {
            "mse_before": float(mse_before),
            "mse_after": float(mse_after),
            "mse_improvement": float((mse_before - mse_after) / mse_before * 100),
            "ncc_before": float(ncc_before),
            "ncc_after": float(ncc_after),
            "ncc_improvement": float((ncc_after - ncc_before) / abs(ncc_before) * 100),
        }

        return metrics

    def save_transform(self, filename: str):
        """
        Save transform to file.

        Args:
            filename: Output filename (.tfm format)

        Example:
            >>> reg.save_transform('registration_transform.tfm')
        """
        if self.transform is None:
            logger.error("No transform to save. Run registration first.")
            return

        sitk.WriteTransform(self.transform, filename)
        logger.info(f"Transform saved to: {filename}")

    def load_transform(self, filename: str) -> sitk.Transform:
        """
        Load transform from file.

        Args:
            filename: Transform file (.tfm)

        Returns:
            Loaded transform

        Example:
            >>> transform = reg.load_transform('saved_transform.tfm')
            >>> reg.transform = transform
        """
        transform = sitk.ReadTransform(filename)
        logger.info(f"Transform loaded from: {filename}")
        return transform


# Utility functions
def numpy_to_sitk(array: np.ndarray, spacing=(1.0, 1.0, 1.0)) -> sitk.Image:
    """
    Convert NumPy array to SimpleITK Image.

    Args:
        array: 2D or 3D numpy array
        spacing: Voxel spacing (mm)

    Returns:
        SimpleITK Image

    Example:
        >>> img_array = np.random.rand(256, 256, 128)
        >>> sitk_image = numpy_to_sitk(img_array, spacing=(1.0, 1.0, 3.0))
    """
    image = sitk.GetImageFromArray(array)
    image.SetSpacing(spacing)
    return image


def sitk_to_numpy(image: sitk.Image) -> np.ndarray:
    """
    Convert SimpleITK Image to NumPy array.

    Args:
        image: SimpleITK Image

    Returns:
        NumPy array

    Example:
        >>> array = sitk_to_numpy(sitk_image)
    """
    return sitk.GetArrayFromImage(image)
