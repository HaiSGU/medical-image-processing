import numpy as np
import matplotlib.pyplot as plt
import SimpleITK as sitk

def display_slice(image, slice_index, title="Slice"):
    """Display a single slice of a 3D image."""
    plt.imshow(image[slice_index, :, :], cmap='gray')
    plt.title(title)
    plt.axis('off')
    plt.show()

def view_nifti_slice(file_path, slice_index):
    """Load a NIfTI image and display a specific slice."""
    img = sitk.ReadImage(file_path)
    img_array = sitk.GetArrayFromImage(img)
    display_slice(img_array, slice_index, title="NIfTI Slice")

def view_nrrd_slice(file_path, slice_index):
    """Load a NRRD image and display a specific slice."""
    img = sitk.ReadImage(file_path)
    img_array = sitk.GetArrayFromImage(img)
    display_slice(img_array, slice_index, title="NRRD Slice")

def view_dicom_slice(file_path, slice_index):
    """Load a DICOM image and display a specific slice."""
    img = sitk.ReadImage(file_path)
    img_array = sitk.GetArrayFromImage(img)
    display_slice(img_array, slice_index, title="DICOM Slice")

def view_ct_slice(file_path, slice_index):
    """Load a CT image and display a specific slice."""
    img = sitk.ReadImage(file_path)
    img_array = sitk.GetArrayFromImage(img)
    display_slice(img_array, slice_index, title="CT Slice")

def view_mri_slice(file_path, slice_index):
    """Load an MRI image and display a specific slice."""
    img = sitk.ReadImage(file_path)
    img_array = sitk.GetArrayFromImage(img)
    display_slice(img_array, slice_index, title="MRI Slice")