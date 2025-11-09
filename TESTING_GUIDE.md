# 🧪 HƯỚNG DẪN TEST TỪNG TRANG

## 🚀 Bước 1: Chạy App

Mở terminal và chạy:
```bash
streamlit run app.py
```

App sẽ mở tại: http://localhost:8501

---

## 1️⃣ TEST TRANG: Anonymization

### 📁 File để test:
```
data/anonym/dicom_dir/ID_0000_AGE_0060_CONTRAST_1_CT.dcm
```

### 📋 Các bước test:

1. **Mở trang Anonymization** (sidebar bên trái)

2. **Upload DICOM file:**
   - Click "Browse files"
   - Chọn: `data/anonym/dicom_dir/ID_0000_AGE_0060_CONTRAST_1_CT.dcm`
   - Hoặc chọn nhiều files cùng lúc

3. **Xem Metadata TRƯỚC khi anonymize:**
   - Sẽ hiển thị: Patient Name, Patient ID, Birth Date, etc.

4. **Click "Anonymize DICOM"**

5. **Kiểm tra kết quả:**
   - Metadata đã được xóa (hiển thị "Anonymous" hoặc trống)
   - Ảnh vẫn giữ nguyên
   - Download file đã anonymize

### ✅ Kết quả mong đợi:
- ✅ Patient Name → "Anonymous" hoặc bị xóa
- ✅ Patient ID → Random ID hoặc bị xóa
- ✅ Birth Date → Bị xóa
- ✅ Ảnh DICOM vẫn hiển thị bình thường

---

## 2️⃣ TEST TRANG: Segmentation

### 📁 Files để test (chọn 1):

**Khuyên dùng:**
```
data/sitk/A1_grayT1.nrrd          ⭐ BEST - Brain MRI T1
data/sitk/A1_grayT2.nrrd          - Brain MRI T2
data/sitk/training_001_mr_T1.mha  - Brain MRI
```

### 📋 Các bước test:

1. **Mở trang Segmentation**

2. **Upload MRI file:**
   - Click "Browse files"
   - Chọn: `data/sitk/A1_grayT1.nrrd` (khuyên dùng)

3. **Xem ảnh gốc:**
   - Hiển thị brain MRI 3D (có thể xem nhiều slices)

4. **Chọn thuật toán:**
   - Otsu Thresholding (nhanh)
   - Watershed
   - Region Growing
   - Active Contour

5. **Click "Run Segmentation"**

6. **Xem kết quả:**
   - Ảnh gốc vs ảnh đã segment
   - Vùng não được tách ra (màu khác)
   - Có thể overlay lên ảnh gốc

### ✅ Kết quả mong đợi:
- ✅ Vùng não được tách rõ ràng
- ✅ Xương sọ, da được loại bỏ
- ✅ Có thể thấy ranh giới não rõ ràng

---

## 3️⃣ TEST TRANG: CT Reconstruction

### 📁 Files để test:

**Option 1: Dùng Built-in Phantom (KHUYÊN DÙNG - không cần upload)**
```
✅ Chọn "Generate Phantom" trong app
```

**Option 2: Upload Sinogram có sẵn**
```
data/synthetic_ct/sinogram_full_180angles.npy     ⭐ Full sampling
data/synthetic_ct/sinogram_sparse_90angles.npy    - Sparse view (90 góc)
data/synthetic_ct/sinogram_verysparse_45angles.npy - Very sparse (45 góc)
data/synthetic_ct/sinogram_limited_120deg.npy     - Limited angle
```

**Cần upload thêm (nếu dùng Option 2):**
```
data/synthetic_ct/angles_180.npy  (với sinogram_full_180angles.npy)
data/synthetic_ct/angles_90.npy   (với sinogram_sparse_90angles.npy)
```

### 📋 Các bước test:

#### **Cách 1: Dùng Built-in Phantom (Dễ nhất)**

1. **Mở trang CT Reconstruction**

2. **Chọn "Generate New Phantom"**
   - App tự tạo Shepp-Logan phantom
   - Tự tạo sinogram

3. **Chọn số góc projection:**
   - 180 angles (full) - tốt nhất
   - 90 angles (sparse)
   - 45 angles (very sparse)

4. **Chọn thuật toán reconstruction:**
   - **Filtered Back Projection (FBP)** - Nhanh nhất ⭐
   - **ART** - Chậm hơn, chất lượng tốt hơn với sparse data
   - **SART** - Chậm nhất, chất lượng tốt nhất

5. **Click "Reconstruct"**

6. **Xem kết quả:**
   - Sinogram (projection data)
   - Reconstructed image
   - Compare với ground truth

#### **Cách 2: Upload Sinogram**

1. **Mở trang CT Reconstruction**

2. **Upload sinogram:**
   - Chọn: `data/synthetic_ct/sinogram_full_180angles.npy`

3. **Upload angles:**
   - Chọn: `data/synthetic_ct/angles_180.npy`

4. **Chọn thuật toán và Reconstruct**

### ✅ Kết quả mong đợi:
- ✅ Full sampling (180°): Ảnh rõ nét, ít artifacts
- ✅ Sparse (90°): Ảnh có một số artifacts nhưng vẫn nhận diện được
- ✅ Very sparse (45°): Nhiều artifacts, cần thuật toán tốt hơn (SART)
- ✅ FBP nhanh nhất, SART chất lượng tốt nhất

---

## 4️⃣ TEST TRANG: MRI Reconstruction

### 📁 Files để test:

**Option 1: Upload ảnh bất kỳ (App tự tạo k-space) - DỄ NHẤT**
```
data/sitk/A1_grayT1.nrrd          ⭐ Brain MRI
data/synthetic_preprocessing/test_image_01.npy
```

**Option 2: Upload K-space có sẵn**
```
data/synthetic_mri/sample_01_kspace_full.npy       - 100% sampling
data/synthetic_mri/sample_01_kspace_50percent.npy  ⭐ 50% sampling
data/synthetic_mri/sample_01_kspace_25percent.npy  - 25% sampling (extreme)
```

**K-space original (có sẵn từ trước):**
```
data/medical/slice_kspace.npy
```

### 📋 Các bước test:

#### **Cách 1: Upload ảnh thường (Dễ nhất)**

1. **Mở trang MRI Reconstruction**

2. **Upload ảnh:**
   - Click "Browse files"
   - Chọn: `data/sitk/A1_grayT1.nrrd`
   - Hoặc bất kỳ medical image nào

3. **App sẽ tự động:**
   - Convert ảnh sang k-space (2D FFT)
   - Hiển thị k-space magnitude

4. **Chọn undersampling ratio:**
   - 100% (Full k-space) - Không undersample
   - 50% (Half sampling) ⭐ - Cân bằng
   - 25% (Aggressive) - Extreme undersampling

5. **Click "Reconstruct MRI"**

6. **Xem kết quả:**
   - Original image
   - K-space visualization
   - Reconstructed image
   - Comparison (PSNR, SSIM)

#### **Cách 2: Upload K-space trực tiếp**

1. **Mở trang MRI Reconstruction**

2. **Chọn "Upload K-space data"**

3. **Upload k-space file:**
   - Chọn: `data/synthetic_mri/sample_01_kspace_50percent.npy`

4. **Click "Reconstruct"**
   - App apply inverse FFT
   - Hiển thị reconstructed image

### ✅ Kết quả mong đợi:
- ✅ 100% sampling: Ảnh giống hệt original
- ✅ 50% sampling: Ảnh vẫn rõ, có thể có một số blur nhẹ
- ✅ 25% sampling: Ảnh có artifacts rõ rệt, chất lượng giảm
- ✅ K-space visualization: Bright center (low frequencies)

---

## 5️⃣ TEST TRANG: Preprocessing

### 📁 Files để test (bất kỳ medical image):

**Khuyên dùng:**
```
data/sitk/A1_grayT1.nrrd                      ⭐ Brain MRI (best)
data/sitk/training_001_ct.mha                 - CT scan
data/anonym/dicom_dir/ID_0000_*.dcm           - DICOM X-ray/CT
data/synthetic_preprocessing/test_image_01.npy - Synthetic
```

### 📋 Các bước test:

1. **Mở trang Preprocessing**

2. **Upload ảnh:**
   - Click "Browse files"
   - Chọn: `data/sitk/A1_grayT1.nrrd` (khuyên dùng)

3. **Xem ảnh gốc:**
   - Hiển thị original image
   - Show histogram

4. **Chọn preprocessing operations:**

   **✅ Normalization:**
   - Min-Max (0-1)
   - Z-score
   - Histogram Equalization
   
   **✅ Denoising:**
   - Gaussian Filter (smooth)
   - Median Filter (remove salt-pepper noise)
   - Bilateral Filter (preserve edges)
   
   **✅ Resizing:**
   - Nhập target size (e.g., 256x256)
   
   **✅ Contrast Enhancement:**
   - CLAHE (Contrast Limited AHE)
   - Histogram Equalization
   - Gamma Correction

5. **Click "Apply Preprocessing"**

6. **Xem kết quả:**
   - Before vs After
   - Histogram comparison
   - Zoom in để thấy chi tiết

7. **Download processed image**

### ✅ Kết quả mong đợi:
- ✅ Normalization: Histogram shift về [0,1] hoặc mean=0
- ✅ Denoising: Ảnh mịn hơn, ít noise
- ✅ Resizing: Ảnh có size mới
- ✅ CLAHE: Contrast tốt hơn, chi tiết rõ hơn

---

## 📊 BẢNG TỔNG KẾT - Quick Reference

| Trang | File Test | Path | Kết Quả Mong Đợi |
|-------|-----------|------|------------------|
| **Anonymization** | `ID_0000_*.dcm` | `data/anonym/dicom_dir/` | Metadata bị xóa |
| **Segmentation** | `A1_grayT1.nrrd` ⭐ | `data/sitk/` | Não được tách rõ |
| **CT Reconstruction** | Built-in phantom ⭐ | (không cần upload) | Ảnh reconstructed rõ |
| **MRI Reconstruction** | `A1_grayT1.nrrd` ⭐ | `data/sitk/` | Auto-generate k-space |
| **Preprocessing** | `A1_grayT1.nrrd` ⭐ | `data/sitk/` | Ảnh được enhance |

---

## 🎯 Test Workflow Khuyên Dùng

### Test nhanh (5-10 phút):

```
1. Anonymization:  Upload ID_0000_*.dcm → Click Anonymize
2. Segmentation:   Upload A1_grayT1.nrrd → Run Segmentation  
3. CT Recon:       Generate Phantom → Select FBP → Reconstruct
4. MRI Recon:      Upload A1_grayT1.nrrd → 50% sampling → Reconstruct
5. Preprocessing:  Upload A1_grayT1.nrrd → Apply CLAHE → Compare
```

### Test chi tiết (20-30 phút):

```
1. Anonymization:  
   - Test với 5 DICOM files khác nhau
   - Kiểm tra batch processing
   
2. Segmentation:
   - Test với A1_grayT1.nrrd, A1_grayT2.nrrd
   - So sánh các thuật toán (Otsu, Watershed, etc.)
   
3. CT Reconstruction:
   - Test với 180°, 90°, 45° angles
   - So sánh FBP vs SART
   
4. MRI Reconstruction:
   - Test với 100%, 50%, 25% sampling
   - So sánh quality metrics (PSNR, SSIM)
   
5. Preprocessing:
   - Test tất cả operations
   - So sánh before/after cho mỗi operation
```

---

## 💡 Tips & Tricks

### 🔥 Best Combinations:

**For Demo/Presentation:**
- Segmentation: `A1_grayT1.nrrd` + Otsu → Rõ nhất
- CT Recon: Built-in phantom + FBP → Nhanh nhất
- MRI Recon: `A1_grayT1.nrrd` + 50% → Cân bằng
- Preprocessing: `A1_grayT1.nrrd` + CLAHE → Hiệu quả nhất

**For Testing Algorithms:**
- CT: Test sparse data (45°) với SART
- MRI: Test extreme undersampling (25%)
- Preprocessing: Test denoising trên ảnh có noise

**For Speed:**
- CT: FBP algorithm
- MRI: 50% sampling
- Segmentation: Otsu thresholding

### ⚠️ Common Issues:

**Issue 1: File không load được**
- ✅ Check file extension (.dcm, .nrrd, .npy)
- ✅ Check file path đúng
- ✅ Try với file khác

**Issue 2: Reconstruction lâu**
- ✅ Dùng FBP thay vì SART
- ✅ Reduce image size
- ✅ Reduce number of angles

**Issue 3: Kết quả không tốt**
- ✅ CT: Tăng số angles lên 180°
- ✅ MRI: Tăng sampling ratio lên 50-100%
- ✅ Preprocessing: Try different operations

---

## 🎬 Demo Script (Cho Presentation)

### Thời gian: 10 phút

**Minute 1-2: Anonymization**
```
"Đây là DICOM file với thông tin bệnh nhân. 
Click Anonymize → Thông tin đã được xóa để bảo vệ privacy."
```

**Minute 3-4: Segmentation**
```
"Upload brain MRI → Run Segmentation
Thuật toán tự động tách vùng não ra khỏi xương sọ và da."
```

**Minute 5-6: CT Reconstruction**
```
"CT scanner thu thập projections từ nhiều góc.
Generate phantom → Reconstruct → Tạo lại ảnh từ projections.
So sánh 180 góc vs 45 góc → Ảnh quality khác nhau."
```

**Minute 7-8: MRI Reconstruction**
```
"MRI thu thập data trong frequency domain (k-space).
Upload ảnh → Auto generate k-space
Test 50% sampling → Vẫn reconstruct được ảnh tốt.
Giảm thời gian scan từ 20 phút xuống 10 phút."
```

**Minute 9-10: Preprocessing**
```
"Preprocessing cải thiện chất lượng ảnh trước khi analyze.
Apply CLAHE → Contrast tốt hơn, chi tiết rõ hơn.
Ready cho ML models hoặc clinical diagnosis."
```

---

## 🏆 Challenge Mode

Thử test các scenarios khó hơn:

**CT Reconstruction:**
- ⭐⭐⭐ Reconstruct với 45° → Compare với ground truth
- ⭐⭐⭐⭐ Limited angle 120° → Handle incomplete data
- ⭐⭐⭐⭐⭐ Thêm noise vào sinogram → Robust reconstruction

**MRI Reconstruction:**
- ⭐⭐⭐ 25% sampling → Compare quality
- ⭐⭐⭐⭐ Random undersampling pattern → Test flexibility
- ⭐⭐⭐⭐⭐ Combine với denoising → Improve quality

**Preprocessing:**
- ⭐⭐⭐ Chain multiple operations
- ⭐⭐⭐⭐ Optimize parameters (filter size, threshold)
- ⭐⭐⭐⭐⭐ Custom preprocessing pipeline

---

## 📞 Help & Support

**Nếu gặp lỗi:**
1. Check terminal console output
2. Check file path đúng chưa
3. Check file format đúng chưa
4. Restart app: `Ctrl+C` → `streamlit run app.py`

**Files quan trọng:**
- `DATA_GUIDE.md` - Chi tiết về data
- `KAGGLE_SETUP.md` - Setup Kaggle (optional)
- `README.md` - Project overview

**Scripts hữu ích:**
- `generate_synthetic_data.py` - Tạo lại data nếu bị xóa
- `check_kaggle_setup.py` - Check Kaggle API

---

## ✅ Checklist Test Hoàn Chỉnh

- [ ] Anonymization: Upload và anonymize 1 DICOM file
- [ ] Segmentation: Segment 1 brain MRI
- [ ] CT Reconstruction: Reconstruct từ phantom
- [ ] MRI Reconstruction: Reconstruct từ k-space
- [ ] Preprocessing: Apply 3+ operations

**Khi hoàn thành tất cả → Project test PASS! 🎉**

---

**Happy Testing! 🚀**
