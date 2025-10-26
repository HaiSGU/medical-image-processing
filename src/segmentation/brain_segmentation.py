import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt

def segment_brain(image_path, lower_threshold=370, upper_threshold=480):
    """
    Segments the brain from an MRI image using a thresholding method.

    Parameters:
    - image_path: str, path to the input MRI image.
    - lower_threshold: int, lower threshold for brain segmentation.
    - upper_threshold: int, upper threshold for brain segmentation.

    Returns:
    - brain_mask: SimpleITK Image, binary mask of the segmented brain.
    - brain_image: SimpleITK Image, the original image masked to show only the brain.
    """
    # Read the image
    image = sitk.ReadImage(image_path, sitk.sitkFloat32)

    # Create a binary mask using thresholding
    brain_mask = sitk.BinaryThreshold(image, lower=lower_threshold, upper=upper_threshold)

    # Apply the mask to the original image
    brain_image = sitk.Mask(image, brain_mask)

    return brain_mask, brain_image

def display_images(original_image, brain_mask, brain_image):
    """
    Displays the original image, brain mask, and segmented brain image.

    Parameters:
    - original_image: SimpleITK Image, the original MRI image.
    - brain_mask: SimpleITK Image, binary mask of the segmented brain.
    - brain_image: SimpleITK Image, the original image masked to show only the brain.
    """
    original_np = sitk.GetArrayViewFromImage(original_image)
    mask_np = sitk.GetArrayViewFromImage(brain_mask)
    brain_np = sitk.GetArrayViewFromImage(brain_image)

    # Display the images
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.imshow(original_np[original_np.shape[0] // 2], cmap='gray')
    plt.title('Original Image')
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(mask_np[mask_np.shape[0] // 2], cmap='gray')
    plt.title('Brain Mask')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(brain_np[brain_np.shape[0] // 2], cmap='gray')
    plt.title('Segmented Brain Image')
    plt.axis('off')

    plt.show()