# Medical Image Processing Project

This project focuses on various aspects of medical image processing, including anonymization, preprocessing, reconstruction, segmentation, and visualization of medical images. It utilizes different image formats such as DICOM, NIfTI, and NumPy arrays.

## Project Structure

```
medical-image-processing
├── data
│   ├── anonym
│   │   └── our_sample_dicom.dcm
│   ├── medical
│   │   ├── Schepp_Logan_sinogram 1.npy
│   │   └── slice_kspace.npy
│   ├── ml
│   ├── mri
│   │   └── OBJECT_phantom_T2W_TSE_Cor_14_1.nii
│   ├── pathology
│   └── sitk
│       ├── A1_grayT1.nrrd
│       ├── A1_grayT2.nrrd
│       ├── digital_xray.dcm
│       ├── training_001_ct.mha
│       └── training_001_mr_T1.mha
├── notebooks
│   ├── AnonymizingImg.ipynb
│   ├── ImgforML.ipynb
│   ├── MedImgModal.ipynb
│   ├── MRI.ipynb
│   ├── PathologyImg.ipynb
│   └── SITK.ipynb
├── src
│   ├── anonymization
│   │   └── dicom_anonymizer.py
│   ├── preprocessing
│   │   ├── image_transforms.py
│   │   └── registration.py
│   ├── reconstruction
│   │   ├── ct_reconstruction.py
│   │   └── mri_reconstruction.py
│   ├── segmentation
│   │   └── brain_segmentation.py
│   └── visualization
│       └── slice_viewer.py
├── utils
│   ├── file_io.py
│   └── image_utils.py
├── requirements.txt
├── .gitignore
└── README.md
```

## Data Description

- **DICOM Images**: Used for testing anonymization processes and digital X-ray processing.
- **NIfTI Images**: Used for MRI processing.
- **NumPy Files**: Contain sinograms and k-space data for computed tomography and magnetic resonance imaging, respectively.

## Notebooks

The project includes several Jupyter notebooks that demonstrate various techniques in medical image processing:

- **AnonymizingImg.ipynb**: Techniques for anonymizing medical images.
- **ImgforML.ipynb**: Preparing images for machine learning, including augmentation and preprocessing.
- **MedImgModal.ipynb**: Computed tomography and magnetic resonance imaging reconstruction techniques.
- **MRI.ipynb**: Working with MRI images, including reading and visualizing NIfTI images.
- **PathologyImg.ipynb**: Processing histopathology images.
- **SITK.ipynb**: Registration and segmentation using SimpleITK.

## Source Code

The source code is organized into several modules:

- **anonymization**: Functions for anonymizing DICOM images.
- **preprocessing**: Image transformation and registration techniques.
- **reconstruction**: Functions for reconstructing images from CT and MRI data.
- **segmentation**: Brain segmentation functions for MRI data.
- **visualization**: Functions for visualizing image slices.

## Utilities

Utility scripts for file I/O and image processing tasks are included in the `utils` directory.

## Requirements

To install the required packages, refer to the `requirements.txt` file.

## Usage

Follow the instructions in the respective notebooks to explore the functionalities of this project. Each notebook contains detailed explanations and code examples for various medical image processing tasks.