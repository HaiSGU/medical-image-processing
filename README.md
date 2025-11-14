# Medical Image Processing System# Medical Image Processing System



Hệ thống xử lý ảnh y tế toàn diện với giao diện web tương tác.A comprehensive Python toolkit and web application for medical image processing, featuring file I/O, anonymization, reconstruction, segmentation, and preprocessing capabilities.



![Python](https://img.shields.io/badge/Python-3.9%2B-blue)![Python](https://img.shields.io/badge/Python-3.9%2B-blue)

![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-red)![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-red)

![License](https://img.shields.io/badge/License-MIT-green)![License](https://img.shields.io/badge/License-MIT-green)



## 🎯 Tổng quan## Table of Contents



Project cung cấp cả **thư viện Python** và **ứng dụng web** để xử lý ảnh y tế, phù hợp cho sinh viên, nhà nghiên cứu và chuyên gia y tế.- [Overview](#overview)

- [Features](#features)

**Tính năng chính:**- [Demo](#demo)

- 🔐 Ẩn danh hóa DICOM (xóa thông tin cá nhân)- [Installation](#installation)

- 🧠 Phân đoạn não (4 thuật toán)- [Quick Start](#quick-start)

- 💀 Tái tạo CT từ sinogram (FBP, SART)- [Web Application](#web-application)

- 🧲 Tái tạo MRI từ K-space- [Python Library Usage](#python-library-usage)

- 🎨 Tiền xử lý ảnh (normalization, denoising, enhancement)- [Project Structure](#project-structure)

- 📊 Đọc/ghi nhiều định dạng (NIfTI, DICOM, NRRD, MetaImage, NumPy)- [Supported Formats](#supported-formats)

- [Documentation](#documentation)

## 🚀 Cài đặt nhanh- [Contributing](#contributing)

- [License](#license)

```bash

# Clone repository## Overview

git clone https://github.com/HaiSGU/medical-image-processing.git

cd medical-image-processingThis project provides both a **Python library** and an **interactive web application** for medical image processing. Built for researchers, students, and healthcare professionals working with medical imaging data.



# Tạo môi trường ảo**Key Capabilities:**

python -m venv venv- Multi-format medical image I/O (NIfTI, DICOM, NRRD, MetaImage, NumPy)

venv\Scripts\activate  # Windows- DICOM anonymization (PHI removal)

# source venv/bin/activate  # Linux/Mac- Brain segmentation (4 methods)

- CT reconstruction from sinograms (FBP, SART)

# Cài đặt dependencies- MRI K-space reconstruction

pip install -r requirements.txt- Image preprocessing pipeline

- Interactive web interface

# Chạy app

streamlit run app.py## Features

```

### File I/O & Visualization

Mở trình duyệt tại: `http://localhost:8501`- Read/write NIfTI, DICOM, NRRD, MetaImage, NumPy formats

- 2D slice viewing with navigation

## ✨ Tính năng chi tiết- 3D visualization with Plotly

- Metadata extraction and display

### 1. Ẩn danh hóa DICOM

- Xóa thông tin bệnh nhân (PHI)### DICOM Anonymization

- Xử lý hàng loạt- Remove Protected Health Information (PHI)

- Export ZIP files- Batch processing support

- Customizable patient ID mapping

### 2. Phân đoạn Não  - ZIP export for anonymized files

- Ngưỡng thủ công & Otsu

- Region Growing### 3. Phân đoạn Ảnh (Segmentation)

- Tự độngTách vùng quan tâm ra khỏi ảnh (ví dụ: tìm vùng não)

- **Ngưỡng thủ công:** Tự chọn giá trị ngưỡng

### 3. Tái tạo CT- **Otsu:** Tự động tìm ngưỡng tốt nhất

- FBP (4 bộ lọc: ramp, shepp-logan, cosine, hamming)- **Region Growing:** Phát triển vùng từ điểm chọn

- SART (iterative)- **Tự động:** Phân đoạn não hoàn toàn tự động

- Shepp-Logan phantom

- Đo PSNR, SSIM### 4. Tái tạo ảnh CT

Tái tạo ảnh CT từ dữ liệu sinogram (dữ liệu thô từ máy chụp)

### 4. Tái tạo MRI- **FBP (Filtered Backprojection):** Thuật toán tái tạo nhanh

- K-space ↔ Image domain- **SART:** Thuật toán lặp, chính xác hơn

- Magnitude & Phase extraction- Tạo phantom để test

- Partial Fourier- Đo lường chất lượng ảnh tái tạo

- Visualization

### 5. Tái tạo ảnh MRI

### 5. Tiền xử lýTái tạo ảnh MRI từ K-space (dữ liệu tần số)

- **Chuẩn hóa:** Min-Max, Z-Score- Chuyển đổi từ K-space sang ảnh thực

- **Transforms:** Resize, Crop- Hiển thị magnitude (độ lớn) và phase (pha)

- **Khử nhiễu:** Gaussian, Median- Partial Fourier: tái tạo từ dữ liệu thiếu

- **Tăng cường:** Histogram Eq, CLAHE- Trực quan hóa tương tác

- **Augmentation:** Flip, Rotate

### 6. Tiền xử lý Ảnh

## 📊 Dữ liệuCải thiện chất lượng ảnh trước khi phân tích

- **Chuẩn hóa:** Min-Max, Z-Score

Project bao gồm dữ liệu mẫu:- **Thay đổi kích thước:** Resize, Crop

- **Khử nhiễu:** Gaussian blur, Median filter  

```- **Tăng độ tương phản:** Histogram Equalization, CLAHE

data/- **Augmentation:** Lật, xoay, thêm nhiễu

├── anonym/dicom_dir/      # 100 DICOM files

├── sitk/                  # 4 Brain MRI (.nrrd, .mha)### CT Reconstruction

├── medical/               # Sinogram & K-space- **Filtered Backprojection (FBP)** with 4 filters:

├── synthetic_ct/          # CT test data  - Ram-Lak (standard)

├── synthetic_mri/         # MRI test data    - Shepp-Logan (smooth)

└── synthetic_preprocessing/  # Preprocessing test  - Cosine (smoother)

```  - Hamming (smoothest)

- **SART (Simultaneous Algebraic Reconstruction Technique)**

### File test đề xuất- Shepp-Logan phantom generation

- Quality metrics (PSNR, SSIM)

| Trang | File | Path |

|-------|------|------|### MRI Reconstruction

| Anonymization | `ID_0000_AGE_0060_CONTRAST_1_CT.dcm` | `data/anonym/dicom_dir/` |- K-space to image domain conversion

| Segmentation | `A1_grayT1.nrrd` | `data/sitk/` |- Magnitude and phase extraction

| CT Reconstruction | Built-in phantom | (không cần upload) |- Partial Fourier reconstruction

| MRI Reconstruction | `A1_grayT1.nrrd` | `data/sitk/` |- Interactive visualization

| Preprocessing | `A1_grayT1.nrrd` | `data/sitk/` |

### Image Preprocessing

## 🧪 Test nhanh (5 phút)- **Intensity normalization**: Min-Max, Z-Score, Percentile clipping

- **Spatial transforms**: Resize, Crop, Pad

1. **Anonymization:** Upload DICOM → Click Anonymize- **Denoising**: Gaussian blur, Median filter

2. **Segmentation:** Upload `A1_grayT1.nrrd` → Run Segmentation  - **Contrast enhancement**: Histogram Equalization, CLAHE, Gamma correction

3. **CT Reconstruction:** Generate Phantom → FBP → Reconstruct- **Augmentation**: Flip, Rotate, Noise injection

4. **MRI Reconstruction:** Upload ảnh → Auto K-space → Reconstruct- Pipeline builder with JSON export

5. **Preprocessing:** Upload ảnh → Apply CLAHE → Compare

## Cài đặt

## 📁 Cấu trúc

### Yêu cầu hệ thống

```- Python 3.9 trở lên

medical-image-processing/- Windows/Linux/Mac

├── app.py                 # Web app entry

├── pages/                 # 5 trang chức năng### Hướng dẫn cài đặt

│   ├── 1_Anonymization.py

│   ├── 2_Segmentation.py**Bước 1: Tải code về**

│   ├── 3_CT_Reconstruction.py```bash

│   ├── 4_MRI_Reconstruction.pygit clone https://github.com/HaiSGU/medical-image-processing.git

│   └── 5_Preprocessing.pycd medical-image-processing

├── src/                   # Core modules```

│   ├── anonymization/

│   ├── segmentation/**Bước 2: Tạo môi trường ảo (khuyến nghị)**

│   ├── reconstruction/```bash

│   └── preprocessing/# Tạo môi trường ảo

├── utils/                 # Utilitiespython -m venv venv

│   ├── file_io.py

│   ├── image_utils.py# Kích hoạt môi trường

│   └── interpretation.py# Trên Windows:

├── data/                  # Sample datavenv\Scripts\activate

├── examples/              # Example scripts# Trên Linux/Mac:

└── notebooks/             # Jupyter notebookssource venv/bin/activate

``````



## 💻 Sử dụng Python Library**Bước 3: Cài đặt thư viện**

```bash

```pythonpip install -r requirements.txt

# Đọc ảnh```

from utils.file_io import MedicalImageIO

io = MedicalImageIO()### Các thư viện chính

image, metadata = io.read_image("brain.nii.gz")

- `streamlit` - Framework tạo web app

# Phân đoạn- `numpy` - Tính toán số học

from src.segmentation.brain_segmentation import BrainSegmentation- `matplotlib` - Vẽ biểu đồ

seg = BrainSegmentation()- `SimpleITK` - Xử lý ảnh y tế

mask = seg.segment_brain(image, method='auto')- `pydicom` - Đọc/ghi file DICOM

- `nibabel` - Đọc/ghi file NIfTI

# Tái tạo CT- `scikit-image` - Thuật toán xử lý ảnh

from src.reconstruction.ct_reconstruction import CTReconstructor- `scipy` - Tính toán khoa học

recon = CTReconstructor()

ct_image = recon.fbp_reconstruction(sinogram, filter_type='ramp')## Cách sử dụng



# Tiền xử lý### Chạy ứng dụng Web

from src.preprocessing.image_transforms import ImageTransforms

trans = ImageTransforms()**Bước 1: Mở Terminal/Command Prompt**

normalized = trans.normalize_intensity(image, method='minmax')

```**Bước 2: Chạy lệnh**

```bash

## ⚠️ Lưu ýstreamlit run app.py

```

- **Mục đích:** Học tập & nghiên cứu - KHÔNG dùng cho chẩn đoán thực tế

- **Python:** Yêu cầu 3.9+**Bước 3: Mở trình duyệt**

- **Data:** Tất cả file mẫu đã có sẵn trong `data/`- Tự động mở hoặc vào: `http://localhost:8501`



## 🔧 Troubleshooting**Bước 4: Sử dụng**

1. Tải ảnh lên từ máy tính

**Module not found:**2. Xem thông tin và thống kê

```bash3. Chọn công cụ xử lý ở menu bên trái

pip install -r requirements.txt4. Làm theo hướng dẫn trong từng công cụ

```

### Ví dụ sử dụng code Python

**Port đã sử dụng:**

```bash```python

streamlit run app.py --server.port 8502# Đọc ảnh y tế

```from utils.file_io import MedicalImageIO



**File không đọc được:**io_handler = MedicalImageIO()

- Kiểm tra định dạng (.nii, .dcm, .nrrd, .mha, .npy)image, metadata = io_handler.read_image("path/to/image.nii")

- Thử file khác trong `data/`print(f"Kích thước ảnh: {image.shape}")



## 📄 License# Phân đoạn não

from src.segmentation.brain_segmentation import BrainSegmentation

MIT License - Tự do sử dụng cho học tập và nghiên cứu.

segmentor = BrainSegmentation(image)

## 👨‍💻 Tác giảmask = segmentor.segment(method="automatic")

print(f"Đã tìm thấy vùng não với {mask.sum()} pixels")

**HaiSGU**  

- Repository: https://github.com/HaiSGU/medical-image-processing  # Load medical image

- Issues: https://github.com/HaiSGU/medical-image-processing/issuesio_handler = MedicalImageIO()

image, metadata = io_handler.read_image("brain_mri.nii.gz")

---

# Segment brain

**Made with ❤️ for the medical imaging community**segmenter = BrainSegmentation()

mask = segmenter.segment_brain(image, method='auto')

# Save result
io_handler.save_image(mask, "brain_mask.nii.gz", metadata)
```

## Cấu trúc Project

```
medical-image-processing/
│
├── app.py                      # Trang chủ - Tải và xem ảnh
├── requirements.txt            # Danh sách thư viện cần cài
├── README.md                   # File này
│
├── pages/                      # Các trang công cụ
│   ├── 1_Anonymization.py     # Ẩn danh hóa DICOM
│   ├── 2_Segmentation.py      # Phân đoạn ảnh
│   ├── 3_CT_Reconstruction.py # Tái tạo CT
│   ├── 4_MRI_Reconstruction.py# Tái tạo MRI
│   └── 5_Preprocessing.py     # Tiền xử lý
│
├── src/                        # Mã nguồn xử lý
│   ├── anonymization/         # Module ẩn danh
│   ├── segmentation/          # Module phân đoạn
│   ├── reconstruction/        # Module tái tạo
│   └── preprocessing/         # Module tiền xử lý
│
├── utils/                      # Công cụ hỗ trợ
│   ├── file_io.py             # Đọc/ghi file ảnh
│   └── image_utils.py         # Xử lý ảnh cơ bản
│
├── data/                       # Thư mục chứa ảnh mẫu
│   ├── test_output/           # Ảnh test có dữ liệu
│   └── medical/               # Ảnh y tế mẫu
│
└── examples/                   # Code ví dụ
    └── demo_*.py              # Các file demo
```

## Định dạng ảnh hỗ trợ

| Định dạng | Phần mở rộng | Mô tả | Dùng cho |
|-----------|--------------|-------|----------|
| **NIfTI** | .nii, .nii.gz | Phổ biến trong nghiên cứu não | MRI, fMRI |
| **DICOM** | .dcm | Tiêu chuẩn y tế quốc tế | CT, MRI, X-quang |
| **NRRD** | .nrrd | Nearly Raw Raster Data | Nghiên cứu |
| **MetaImage** | .mha, .mhd | ITK format | Xử lý ảnh y tế |
| **NumPy** | .npy | Mảng Python | Dữ liệu đã xử lý |

## Lưu ý

### Dành cho sinh viên

1. **Mục đích:** Đồ án này phục vụ học tập, nghiên cứu. KHÔNG dùng để chẩn đoán y khoa thực tế.

2. **File test:** Sử dụng file trong `data/test_output/` để test các chức năng:
   - `synthetic_dicom.dcm` - Test ẩn danh hóa
   - `test_volume.mha` - Test phân đoạn/tiền xử lý
   - `slice_kspace.npy` - Test tái tạo MRI

3. **Hiểu thuật toán:** Mỗi công cụ có giải thích ngắn gọn về thuật toán. Đọc kỹ để hiểu cách hoạt động.

4. **Tham khảo code:** Xem code trong `src/` và `examples/` để hiểu cách implement.

### Xử lý lỗi thường gặp

**Lỗi: "Module not found"**
```bash
# Cài lại requirements
pip install -r requirements.txt
```

**Lỗi: "Port already in use"**
```bash
# Đổi port
streamlit run app.py --server.port 8502
```

**Lỗi: "Cannot read file"**
- Kiểm tra định dạng file có đúng không
- Thử file khác trong `data/test_output/`
- File có thể bị hỏng hoặc rỗng

### Đóng góp và Phản hồi

Nếu gặp lỗi hoặc có đề xuất cải tiến:
1. Tạo Issue trên GitHub
2. Hoặc email: [Thêm email của bạn]

### Bản quyền

MIT License - Tự do sử dụng cho mục đích học tập và nghiên cứu.

---

**Tác giả:** HaiSGU  
**Repository:** https://github.com/HaiSGU/medical-image-processing  
**Năm:** 2025

# Otsu method
mask = seg.otsu_segmentation(image)

# Region growing
mask = seg.region_growing(image, seed_point=(128, 128, 64))
```

### CT Reconstruction

```python
from src.reconstruction.ct_reconstruction import CTReconstructor

recon = CTReconstructor()

# Filtered backprojection
image = recon.fbp_reconstruction(sinogram, filter_type='ram-lak')

# SART
image = recon.sart_reconstruction(
    sinogram, 
    iterations=10,
    relaxation=0.15
)

# Evaluate quality
psnr = recon.calculate_psnr(original, reconstructed)
ssim = recon.calculate_ssim(original, reconstructed)
```

### MRI Reconstruction

```python
from src.reconstruction.mri_reconstruction import MRIReconstructor

recon = MRIReconstructor()

# K-space to image
image = recon.kspace_to_image(kspace_data)

# Extract magnitude and phase
magnitude = recon.get_magnitude(image)
phase = recon.get_phase(image)

# Partial Fourier
image = recon.partial_fourier_reconstruction(kspace_data, factor=0.75)
```

### Image Preprocessing

```python
from src.preprocessing.image_transforms import ImageTransforms

transforms = ImageTransforms()

# Normalize
normalized = transforms.normalize_intensity(image, method='minmax')

# Resize
resized = transforms.resize(image, output_shape=(256, 256, 128))

# Denoise
denoised = transforms.gaussian_filter(image, sigma=1.0)

# Enhance contrast
enhanced = transforms.enhance_contrast(image, method='clahe')
```

## Project Structure

```
medical-image-processing/
│
├── app.py                      # Streamlit web application entry point
├── requirements.txt            # Python dependencies
├── README.md                   # This file
├── LICENSE                     # MIT License
│
├── pages/                      # Streamlit multi-page app
│   ├── 1_Anonymization.py      # DICOM anonymization page
│   ├── 2_Segmentation.py       # Brain segmentation page
│   ├── 3_CT_Reconstruction.py  # CT reconstruction page
│   ├── 4_MRI_Reconstruction.py # MRI reconstruction page
│   └── 5_Preprocessing.py      # Image preprocessing page
│
├── src/                        # Core library modules
│   ├── anonymization/
│   │   └── dicom_anonymizer.py # DICOM PHI removal
│   ├── preprocessing/
│   │   ├── image_transforms.py # Normalization, transforms, augmentation
│   │   └── registration.py     # Image registration
│   ├── reconstruction/
│   │   ├── ct_reconstruction.py  # FBP, SART algorithms
│   │   └── mri_reconstruction.py # K-space processing
│   ├── segmentation/
│   │   └── brain_segmentation.py # Brain extraction methods
│   └── visualization/
│       └── slice_viewer.py     # 2D/3D visualization
│
├── utils/                      # Utility functions
│   ├── file_io.py              # Multi-format I/O
│   └── image_utils.py          # Image processing utilities
│
├── data/                       # Sample data (not in git)
│   ├── anonym/                 # DICOM samples
│   ├── medical/                # Sinogram, K-space
│   ├── mri/                    # NIfTI files
│   └── sitk/                   # SimpleITK formats
│
├── notebooks/                  # Jupyter notebooks
│   ├── AnonymizingImg.ipynb
│   ├── MedImgModal.ipynb
│   ├── MRI.ipynb
│   ├── SITK.ipynb
│   └── ...
│
└── examples/                   # Example scripts
    ├── demo_file_io.py
    ├── demo_anonymization.py
    ├── demo_segmentation.py
    └── ...
```

## Supported Formats

| Format | Extension | Read | Write | Metadata |
|--------|-----------|------|-------|----------|
| **NIfTI** | `.nii`, `.nii.gz` | ✅ | ✅ | ✅ |
| **DICOM** | `.dcm` | ✅ | ✅ | ✅ |
| **NRRD** | `.nrrd` | ✅ | ✅ | ✅ |
| **MetaImage** | `.mha`, `.mhd` | ✅ | ✅ | ✅ |
| **NumPy** | `.npy` | ✅ | ✅ | ❌ |

**Metadata includes:** Spacing, origin, direction, data type, dimensions

## Documentation

### Jupyter Notebooks

Detailed tutorials available in `notebooks/`:

- **AnonymizingImg.ipynb** - DICOM anonymization techniques
- **MedImgModal.ipynb** - CT and MRI reconstruction
- **MRI.ipynb** - MRI processing workflow
- **SITK.ipynb** - SimpleITK registration and segmentation
- **ImgforML.ipynb** - Preparing images for machine learning

### Example Scripts

Ready-to-run examples in `examples/`:

```bash
python examples/demo_file_io.py
python examples/demo_anonymization.py
python examples/demo_segmentation.py
python examples/demo_ct_reconstruction.py
python examples/demo_mri_reconstruction.py
python examples/demo_preprocessing.py
```

## Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

### Development Setup

```bash
# Clone your fork
git clone https://github.com/YOUR_USERNAME/medical-image-processing.git

# Install dev dependencies
pip install -r requirements.txt

# Run tests (if available)
pytest tests/

# Format code
black src/ utils/ pages/
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- **SimpleITK** - Medical image processing library
- **Streamlit** - Web application framework
- **scikit-image** - Image processing algorithms
- Sample data from various open medical imaging datasets

## Contact

**Developer:** HaiSGU  
**Repository:** [https://github.com/HaiSGU/medical-image-processing](https://github.com/HaiSGU/medical-image-processing)  
**Issues:** [https://github.com/HaiSGU/medical-image-processing/issues](https://github.com/HaiSGU/medical-image-processing/issues)

---

**Made with ❤️ for the medical imaging community**