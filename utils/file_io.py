import os
import numpy as np
import SimpleITK as sitk
import nibabel as nib
import pydicom

def read_nifti(file_path):
    """Read a NIfTI file and return the image data."""
    img = nib.load(file_path)
    return img.get_fdata()

def read_dicom(file_path):
    """Read a DICOM file and return the dataset."""
    return pydicom.dcmread(file_path)

def read_nrrd(file_path):
    """Read a NRRD file and return the image data and header."""
    return sitk.ReadImage(file_path)

def read_numpy(file_path):
    """Read a NumPy file and return the data."""
    return np.load(file_path)

def write_nifti(file_path, data, affine=None):
    """Write data to a NIfTI file."""
    img = nib.Nifti1Image(data, affine)
    nib.save(img, file_path)

def write_dicom(file_path, dataset):
    """Write a DICOM dataset to a file."""
    dataset.save_as(file_path)

def write_nrrd(file_path, image):
    """Write a SimpleITK image to a NRRD file."""
    sitk.WriteImage(image, file_path)

def write_numpy(file_path, data):
    """Write data to a NumPy file."""
    np.save(file_path, data)