# 📊 Hướng Dẫn Sử Dụng Data - Medical Image Processing

## ✅ Data Đã Sẵn Sàng!

Bạn đã có đủ data để test tất cả 5 trang của project:

---

## 📁 Cấu Trúc Data

```
data/
├── anonym/dicom_dir/          ✅ 100 DICOM files
├── sitk/                      ✅ 4 Brain MRI files
├── medical/                   ✅ Sinogram & K-space
├── synthetic_ct/              🆕 CT Reconstruction data
├── synthetic_mri/             🆕 MRI K-space data
└── synthetic_preprocessing/   🆕 Test images
```

---

## 🎯 Data Cho Từng Trang

### 1️⃣ Anonymization (Trang 1)

**📂 Sử dụng:** `data/anonym/dicom_dir/`

**Có sẵn:** 100 DICOM files với metadata đầy đủ

**File mẫu:**
- `ID_0000_AGE_0060_CONTRAST_1_CT.dcm`
- `ID_0001_AGE_0069_CONTRAST_1_CT.dcm`
- ... (98 files nữa)

**Cách test:**
1. Mở trang Anonymization
2. Upload 1 hoặc nhiều DICOM files
3. Click "Anonymize" để xóa thông tin cá nhân
4. Download kết quả

---

### 2️⃣ Segmentation (Trang 2)

**📂 Sử dụng:** `data/sitk/`

**Có sẵn:**
- `A1_grayT1.nrrd` - Brain MRI T1
- `A1_grayT2.nrrd` - Brain MRI T2
- `training_001_mr_T1.mha` - Brain MRI T1
- `training_001_ct.mha` - CT scan

**Cách test:**
1. Mở trang Segmentation
2. Upload file `.nrrd` hoặc `.mha`
3. Chọn thuật toán segmentation
4. Xem kết quả phân đoạn não

**Khuyên dùng:** `A1_grayT1.nrrd` (chất lượng tốt nhất)

---

### 3️⃣ CT Reconstruction (Trang 3)

**📂 Sử dụng:** `data/synthetic_ct/`

**Có sẵn (mới tạo):**
- `phantom_ground_truth.npy` - Ground truth image
- `sinogram_full_180angles.npy` - Full sampling (180 góc)
- `sinogram_sparse_90angles.npy` - Sparse view (90 góc)
- `sinogram_verysparse_45angles.npy` - Very sparse (45 góc)
- `sinogram_limited_120deg.npy` - Limited angle (120°)
- `angles_*.npy` - Các góc projection tương ứng

**Cách test:**
1. Mở trang CT Reconstruction
2. **Option 1:** Tạo phantom mới (built-in)
3. **Option 2:** Upload sinogram từ `data/synthetic_ct/`
4. Chọn thuật toán: FBP, ART, SART
5. So sánh kết quả reconstruction

**Khuyên dùng:**
- Full sampling để có kết quả tốt nhất
- Sparse view để test khả năng reconstruction với ít data

**Existing data:** `data/medical/Schepp_Logan_sinogram 1.npy`

---

### 4️⃣ MRI Reconstruction (Trang 4)

**📂 Sử dụng:** `data/synthetic_mri/kspace_data/`

**Có sẵn (mới tạo):**

Mỗi sample có 6 files:
```
sample_01_original_image.npy       # Ground truth
sample_01_kspace_full.npy          # Full k-space (100%)
sample_01_kspace_50percent.npy     # Undersampled (50%)
sample_01_kspace_25percent.npy     # Undersampled (25%)
sample_01_mask_50percent.npy       # Sampling mask
sample_01_mask_25percent.npy       # Sampling mask
```

**Có 5 samples:** sample_01 đến sample_05

**Cách test:**
1. Mở trang MRI Reconstruction
2. **Option 1:** Upload ảnh bất kỳ → tự tạo k-space
3. **Option 2:** Upload k-space từ `data/synthetic_mri/`
4. Chọn undersampling ratio (25%, 50%, 100%)
5. So sánh reconstruction với original

**Khuyên dùng:**
- `sample_01_kspace_50percent.npy` - cân bằng tốc độ/chất lượng
- `sample_01_kspace_25percent.npy` - test extreme undersampling

**Existing data:** `data/medical/slice_kspace.npy`

---

### 5️⃣ Preprocessing (Trang 5)

**📂 Sử dụng:** `data/synthetic_preprocessing/`

**Có sẵn (mới tạo):**
- `test_image_01.npy` đến `test_image_10.npy`
- 3 loại pattern: X-ray-like, CT-like, MRI-like
- Size: 512x512 grayscale

**Cách test:**
1. Mở trang Preprocessing
2. Upload bất kỳ medical image (DICOM, NIfTI, NRRD, etc.)
3. Áp dụng operations:
   - Normalization
   - Denoising
   - Resizing
   - Contrast Enhancement
4. So sánh before/after

**Có thể dùng:**
- Synthetic images: `data/synthetic_preprocessing/test_image_*.npy`
- Real MRI: `data/sitk/*.nrrd`, `*.mha`
- Real CT: `data/anonym/dicom_dir/*.dcm`

---

## 🚀 Quick Start

### 1. Chạy app:
```bash
streamlit run app.py
```

### 2. Test từng trang:

**Anonymization:**
```python
# Upload file từ: data/anonym/dicom_dir/ID_0000_AGE_0060_CONTRAST_1_CT.dcm
```

**Segmentation:**
```python
# Upload file: data/sitk/A1_grayT1.nrrd
```

**CT Reconstruction:**
```python
# Upload sinogram: data/synthetic_ct/sinogram_full_180angles.npy
# Upload angles: data/synthetic_ct/angles_180.npy
```

**MRI Reconstruction:**
```python
# Upload k-space: data/synthetic_mri/sample_01_kspace_50percent.npy
```

**Preprocessing:**
```python
# Upload any image từ data/synthetic_preprocessing/ hoặc data/sitk/
```

---

## 📥 Download Real Data từ Kaggle (Optional)

Nếu muốn thêm real data từ Kaggle:

### Bước 1: Setup Kaggle API
1. Tạo tài khoản Kaggle: https://www.kaggle.com/
2. Vào Settings → API → Create New API Token
3. Download `kaggle.json`
4. Đặt vào: `C:\Users\<YourUsername>\.kaggle\kaggle.json`

### Bước 2: Cài Kaggle
```bash
pip install kaggle
```

### Bước 3: Download data
```bash
python download_kaggle_data.py
```

**Xem chi tiết:** `KAGGLE_SETUP.md`

---

## 📊 Kaggle Datasets Khuyên Dùng

### Cho Segmentation:
- **LGG MRI Segmentation** (có ground truth masks)
  - https://www.kaggle.com/datasets/mateuszbuda/lgg-mri-segmentation
  - 110 patients với TIFF images

### Cho Preprocessing:
- **COVID-19 Radiography** (21,165 X-rays)
  - https://www.kaggle.com/datasets/tawsifurrahman/covid19-radiography-database
- **Chest X-Ray Pneumonia** (5,863 X-rays)
  - https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia

---

## 💡 Tips

### Tốc độ test nhanh:
1. **Anonymization:** Data có sẵn (100 DICOM) ✅
2. **Segmentation:** Data có sẵn (4 MRI) ✅
3. **CT Reconstruction:** Dùng built-in phantom (không cần upload) ✅
4. **MRI Reconstruction:** Upload ảnh bất kỳ → tự tạo k-space ✅
5. **Preprocessing:** Upload ảnh bất kỳ ✅

### Chất lượng tốt nhất:
- Segmentation: `A1_grayT1.nrrd`
- CT: `sinogram_full_180angles.npy`
- MRI: `sample_01_kspace_full.npy`
- Preprocessing: Files từ `data/sitk/`

### Test nhiều scenarios:
- CT: Test với different angles (180°, 90°, 45°, 120°)
- MRI: Test với different undersampling (100%, 50%, 25%)

---

## 🎯 Tổng Kết

✅ **Existing data:** 100 DICOM + 4 MRI + 2 files (sinogram, k-space)

🆕 **Synthetic data mới tạo:**
- 9 files cho CT Reconstruction
- 30 files cho MRI Reconstruction (5 samples × 6 files)
- 10 files cho Preprocessing testing

🎉 **Bạn có thể test toàn bộ project ngay bây giờ!**

📥 **Optional:** Download thêm real data từ Kaggle nếu cần

---

## ❓ Troubleshooting

**Q: Trang nào cần download thêm data?**  
A: KHÔNG! Tất cả đã có sẵn hoặc vừa được tạo synthetic.

**Q: Kaggle data có cần thiết không?**  
A: KHÔNG bắt buộc. Synthetic data đủ để test và demo project.

**Q: File nào tốt nhất để demo?**  
A:
- Anonymization: `ID_0000_AGE_0060_CONTRAST_1_CT.dcm`
- Segmentation: `A1_grayT1.nrrd`
- CT: Dùng built-in phantom generator
- MRI: `sample_01_kspace_50percent.npy`
- Preprocessing: `A1_grayT1.nrrd`

**Q: Làm sao biết data đã đủ?**  
A: Chạy: `python generate_synthetic_data.py` → Xem summary

---

## 📞 Support

Nếu cần thêm data hoặc gặp vấn đề:
1. Check `README.txt` trong mỗi folder
2. Xem `KAGGLE_SETUP.md` cho Kaggle instructions
3. Chạy lại `generate_synthetic_data.py` nếu cần tạo lại

**Happy Testing! 🎉**
