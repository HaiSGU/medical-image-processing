"""Test DICOM file từ Zenodo"""

import pydicom
from pathlib import Path

# Test file DICOM
dicom_path = Path("data/anonym/our_sample_dicom.dcm")

print(f"Đang test file: {dicom_path}")
print(f"File tồn tại: {dicom_path.exists()}")
print(f"Kích thước: {dicom_path.stat().st_size} bytes")

try:
    # Đọc file DICOM (force=True để đọc file không có header chuẩn)
    ds = pydicom.dcmread(str(dicom_path), force=True)

    print("\n✅ File DICOM hợp lệ!")
    print("\n📋 Thông tin bệnh nhân:")
    print(f"  - Tên: {ds.get('PatientName', 'N/A')}")
    print(f"  - ID: {ds.get('PatientID', 'N/A')}")
    print(f"  - Ngày sinh: {ds.get('PatientBirthDate', 'N/A')}")

    print("\n🏥 Thông tin nghiên cứu:")
    print(f"  - Ngày: {ds.get('StudyDate', 'N/A')}")
    print(f"  - Phương thức: {ds.get('Modality', 'N/A')}")
    print(f"  - Mô tả: {ds.get('StudyDescription', 'N/A')}")

    # Check if có pixel data
    if hasattr(ds, "pixel_array"):
        print(f"\n🖼️  Có dữ liệu ảnh: {ds.pixel_array.shape}")
    else:
        print("\n⚠️  Không có pixel data")

except Exception as e:
    print(f"\n❌ Lỗi: {e}")
