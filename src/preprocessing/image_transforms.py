import numpy as np
from skimage import transform

def rotate_image(image, angle):
    return transform.rotate(image, angle)

def resize_image(image, output_shape):
    return transform.resize(image, output_shape, anti_aliasing=True)

def normalize_image(image):
    return (image - np.min(image)) / (np.max(image) - np.min(image))

def apply_transformations(image, angle, output_shape):
    rotated = rotate_image(image, angle)
    resized = resize_image(rotated, output_shape)
    normalized = normalize_image(resized)
    return normalized