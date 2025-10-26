def load_image(image_path):
    """Load an image from a specified file path."""
    import SimpleITK as sitk
    return sitk.ReadImage(image_path)

def save_image(image, save_path):
    """Save an image to a specified file path."""
    import SimpleITK as sitk
    sitk.WriteImage(image, save_path)

def normalize_image(image):
    """Normalize the image intensity values to the range [0, 1]."""
    import SimpleITK as sitk
    image_array = sitk.GetArrayFromImage(image)
    normalized_array = (image_array - image_array.min()) / (image_array.max() - image_array.min())
    return sitk.GetImageFromArray(normalized_array)

def resize_image(image, new_size):
    """Resize the image to the specified new size."""
    import SimpleITK as sitk
    return sitk.Resample(image, new_size)

def rotate_image(image, angle):
    """Rotate the image by a specified angle (in degrees)."""
    import SimpleITK as sitk
    return sitk.Rotate(image, angle)