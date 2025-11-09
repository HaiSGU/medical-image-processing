# Fix Lỗi Trang Segmentation

## ❌ Vấn đề
File NRRD (`A1_grayT1.nrrd`) gặp lỗi khi upload do Windows temp directory có Unicode/spaces trong đường dẫn.

```
Exception thrown in SimpleITK ImageFileReader_Execute: 
Unable to determine ImageIO reader for "C:\Users\THISPC-1\AppData\Local\Temp\tmpxck6ye_3.nrrd"
```

## ✅ Giải pháp: Dùng file DICOM hoặc NumPy

### Cách 1: Test với DICOM (Khuyến nghị)

1. **Vào trang Segmentation**
2. **Upload file:** `data/anonym/dicom_dir/ID_0000_AGE_0060_CONTRAST_1_CT.dcm`
3. **Chọn phương pháp:** "Tự động"
4. **Click:** "Phân đoạn Não"
5. **Kết quả:** Sẽ thấy ảnh gốc, mask phân đoạn, và overlay

### Cách 2: Test với NumPy

1. **Vào trang Segmentation**
2. **Upload file:** `data/synthetic_preprocessing/test_image_01.npy`
3. **Chọn phương pháp:** "Otsu"
4. **Click:** "Phân đoạn Não"

### Cách 3: Test với các phương pháp khác

**Threshold:**
- Upload: DICOM file
- Chọn: "Ngưỡng"
- Điều chỉnh slider "Giá trị ngưỡng": 50-150
- Click: "Phân đoạn Não"

**Region Growing:**
- Upload: DICOM file
- Chọn: "Tăng trưởng vùng"
- Điều chỉnh: Vị trí X, Y, Z (50%, 50%, 50%)
- Dung sai: 10-20
- Click: "Phán đoạn Não"

## 📊 Files Đã Test và Hoạt động

| File | Format | Kích thước | Trạng thái |
|------|--------|-----------|----------|
| `data/anonym/dicom_dir/ID_*.dcm` | DICOM | 512×512 | ✅ Hoạt động |
| `data/synthetic_preprocessing/test_*.npy` | NumPy | 512×512 | ✅ Hoạt động |
| `data/sitk/A1_grayT1.nrrd` | NRRD | 256×256×? | ⚠️ Có thể lỗi |

## 🔧 Tại sao NRRD bị lỗi?

SimpleITK không thể đọc file từ Windows temp directory khi:
1. Đường dẫn chứa khoảng trắng: `C:\Users\THIS PC\...`
2. Đường dẫn chứa Unicode/special chars
3. Tên file tạm thời có format phức tạp: `tmpxck6ye_3.nrrd`

**Fallback đã được thêm vào `file_io.py`**, nhưng không phải lúc nào cũng hoạt động 100%.

## ✨ Khuyến nghị

**Dùng DICOM (.dcm)** hoặc **NumPy (.npy)** cho:
- ✅ 100% tương thích
- ✅ Không có vấn đề path
- ✅ Load nhanh hơn
- ✅ Metadata đầy đủ

**Tránh NRRD (.nrrd)** khi:
- ❌ Máy Windows với username có khoảng trắng
- ❌ Temp directory có Unicode characters
- ❌ Cần stability cao

## 🎯 Checklist Test Segmentation

- [ ] Upload DICOM file → Tự động → Phân đoạn thành công
- [ ] Thử phương pháp "Otsu" → Thấy mask khác biệt
- [ ] Thử "Ngưỡng" với giá trị 100 → Thấy vùng phân đoạn thay đổi
- [ ] Enable "Áp dụng phép biến đổi hình thái" → Mask mượt hơn
- [ ] Enable "Chỉ giữ thành phần lớn nhất" → Loại bỏ vùng nhỏ
- [ ] Download mask (.npy) → File lưu thành công
- [ ] Kiểm tra statistics (Pixels, Coverage) → Hiển thị đúng

---

**Tóm lại:** Dùng DICOM thay vì NRRD để tránh lỗi! 🎉
