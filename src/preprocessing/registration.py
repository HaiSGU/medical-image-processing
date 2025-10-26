import SimpleITK as sitk
import numpy as np

def register_images(fixed_image_path, moving_image_path, output_image_path):
    fixed_image = sitk.ReadImage(fixed_image_path, sitk.sitkFloat32)
    moving_image = sitk.ReadImage(moving_image_path, sitk.sitkFloat32)

    # Initialize the registration method
    registration_method = sitk.ImageRegistrationMethod()

    # Set the metric for optimization
    registration_method.SetMetricAsMeanSquares()

    # Set the optimizer
    registration_method.SetOptimizerAsGradientDescent(learningRate=1.0, numberOfIterations=100)

    # Set the initial transform
    initial_transform = sitk.CenteredTransformInitializer(fixed_image, 
                                                         moving_image, 
                                                         sitk.Euler3DTransform())
    registration_method.SetInitialTransform(initial_transform)

    # Execute the registration
    final_transform = registration_method.Execute(fixed_image, moving_image)

    # Resample the moving image using the final transform
    resampled_image = sitk.Resample(moving_image, fixed_image, final_transform, 
                                     sitk.sitkLinear, 0.0, moving_image.GetPixelID())

    # Save the registered image
    sitk.WriteImage(resampled_image, output_image_path)

def main():
    fixed_image_path = "data/sitk/A1_grayT1.nrrd"
    moving_image_path = "data/sitk/A1_grayT2.nrrd"
    output_image_path = "data/sitk/registered_image.nrrd"

    register_images(fixed_image_path, moving_image_path, output_image_path)

if __name__ == "__main__":
    main()