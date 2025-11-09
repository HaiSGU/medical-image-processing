# 🔑 Hướng dẫn Setup Kaggle API

## Bước 1: Lấy Kaggle API Token

1. Đăng nhập vào Kaggle: https://www.kaggle.com/
2. Click vào avatar (góc trên bên phải) → **Settings**
3. Scroll xuống phần **API**
4. Click **"Create New API Token"**
5. File `kaggle.json` sẽ được download về máy

## Bước 2: Đặt kaggle.json vào đúng vị trí

### Windows:
```
C:\Users\<TênMáyTính>\.kaggle\kaggle.json
```

**Ví dụ:**
```
C:\Users\THIS PC\.kaggle\kaggle.json
```

### Các bước:
1. Tạo thư mục `.kaggle` trong thư mục user của bạn:
   ```
   mkdir %USERPROFILE%\.kaggle
   ```

2. Copy file `kaggle.json` vào thư mục vừa tạo:
   ```
   copy Downloads\kaggle.json %USERPROFILE%\.kaggle\
   ```

3. Hoặc làm thủ công:
   - Mở File Explorer
   - Gõ `%USERPROFILE%` vào thanh địa chỉ → Enter
   - Tạo thư mục mới tên `.kaggle`
   - Copy file `kaggle.json` vào đó

## Bước 3: Chạy script download

Sau khi setup xong, chạy:
```bash
python download_kaggle_data.py
```

## Lưu ý:

- Một số datasets từ competitions cần **accept rules** trước:
  - SIIM Pneumothorax: https://www.kaggle.com/c/siim-acr-pneumothorax-segmentation/rules
  - RSNA Pneumonia: https://www.kaggle.com/c/rsna-pneumonia-detection-challenge/rules

- Script sẽ tự động:
  ✅ Download datasets
  ✅ Generate phantom và sinogram cho CT reconstruction
  ✅ Generate k-space data cho MRI reconstruction
  ✅ Tổ chức files vào các folder phù hợp

## Kiểm tra setup:

Chạy lệnh này để kiểm tra Kaggle API đã hoạt động:
```bash
kaggle datasets list
```

Nếu thấy danh sách datasets → Setup thành công! ✅
