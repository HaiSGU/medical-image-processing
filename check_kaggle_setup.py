"""
Script kiểm tra và hướng dẫn setup Kaggle API
"""

import os
from pathlib import Path

print("=" * 70)
print("🔍 KAGGLE API SETUP CHECKER")
print("=" * 70)

# Check if kaggle is installed
try:
    import kaggle

    print("✅ Kaggle package đã được cài đặt")
except ImportError:
    print("❌ Kaggle chưa được cài đặt")
    print("💡 Chạy: pip install kaggle")
    exit(1)

# Check for kaggle.json
kaggle_dir = Path.home() / ".kaggle"
kaggle_json = kaggle_dir / "kaggle.json"

print(f"\n📁 Đang kiểm tra: {kaggle_json}")

if kaggle_json.exists():
    print("✅ File kaggle.json đã tồn tại!")
    print(f"   Location: {kaggle_json}")

    # Try to authenticate
    print("\n🔐 Đang thử xác thực...")
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi

        api = KaggleApi()
        api.authenticate()
        print("✅ Kaggle API authentication thành công!")

        # Test API
        print("\n🧪 Test Kaggle API...")
        datasets = api.dataset_list(page=1)
        print(f"✅ API hoạt động! Tìm thấy {len(datasets)} datasets")
        print("\n🎉 Setup hoàn tất! Bạn có thể chạy download_kaggle_data.py")

    except Exception as e:
        print(f"❌ Authentication thất bại: {str(e)}")
        print("\n💡 Thử lại:")
        print("   1. Xóa file kaggle.json cũ")
        print("   2. Download token mới từ https://www.kaggle.com/settings")
        print("   3. Copy vào:", kaggle_dir)

else:
    print("❌ File kaggle.json KHÔNG tồn tại!")
    print("\n" + "=" * 70)
    print("📋 HƯỚNG DẪN SETUP KAGGLE API")
    print("=" * 70)

    print("\n🔹 Bước 1: Lấy Kaggle API Token")
    print("   1. Truy cập: https://www.kaggle.com/")
    print("   2. Đăng nhập (hoặc tạo tài khoản nếu chưa có)")
    print("   3. Click avatar → Settings")
    print("   4. Scroll xuống phần 'API'")
    print("   5. Click 'Create New API Token'")
    print("   6. File kaggle.json sẽ được download")

    print("\n🔹 Bước 2: Tạo thư mục .kaggle")
    print(f'   Chạy lệnh: mkdir "{kaggle_dir}"')

    if not kaggle_dir.exists():
        try:
            kaggle_dir.mkdir(parents=True, exist_ok=True)
            print(f"   ✅ Đã tạo thư mục: {kaggle_dir}")
        except Exception as e:
            print(f"   ⚠️  Không thể tạo thư mục: {e}")
    else:
        print(f"   ✅ Thư mục đã tồn tại: {kaggle_dir}")

    print("\n🔹 Bước 3: Copy kaggle.json vào thư mục")
    print(f"   Copy file kaggle.json vào: {kaggle_dir}")
    print(f"   Full path: {kaggle_json}")

    print("\n🔹 Bước 4: Chạy lại script này để kiểm tra")
    print("   python check_kaggle_setup.py")

    print("\n" + "=" * 70)
    print("💡 Quick Command:")
    print("=" * 70)

    downloads_path = Path.home() / "Downloads" / "kaggle.json"
    print(f"\nNếu file kaggle.json đang ở Downloads, chạy:")
    print(f'copy "{downloads_path}" "{kaggle_json}"')

    print("\n" + "=" * 70)
