"""
Trang Streamlit để ẩn danh hóa các file DICOM.
Ứng dụng này xóa thông tin nhận dạng cá nhân khỏi file ảnh y tế
để bảo vệ quyền riêng tư bệnh nhân.
"""

# ==================== IMPORT CÁC THƯ VIỆN CẦN THIẾT ====================
import io  # Xử lý input/output với dữ liệu dạng byte stream
import sys  # Tương tác với hệ thống Python
import tempfile  # Tạo các thư mục và file tạm thời
import zipfile  # Xử lý file ZIP (nén và giải nén)
from pathlib import Path  # Xử lý đường dẫn file/folder một cách hiện đại
from typing import Dict  # Định nghĩa kiểu dữ liệu Dictionary

import pandas as pd  # Xử lý dữ liệu dạng bảng
import pydicom  # Đọc và xử lý file DICOM (định dạng ảnh y tế chuẩn)
import streamlit as st  # Framework tạo web app Python
from pydicom.dataset import Dataset  # Class đại diện cho một dataset DICOM

# ==================== CẤU HÌNH ĐƯỜNG DẪN PROJECT ====================
# Lấy đường dẫn thư mục gốc của project (thư mục cha của thư mục pages)
PROJECT_ROOT = Path(__file__).parent.parent
# Thêm đường dẫn gốc vào sys.path để có thể import các module
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Import hàm hiển thị phần giải thích kết quả từ module utils
from utils.interpretation import show_interpretation_section
from utils.image_explainer import explain_input_image


# ==================== HÀM TIỆN ÍCH ====================
def format_tag(dataset: Dataset, tag: str, label: str) -> str:
    """
    Lấy giá trị của một tag DICOM và định dạng thành chuỗi hiển thị.

    Args:
        dataset: Dataset DICOM chứa thông tin ảnh
        tag: Tên tag DICOM cần lấy (vd: "PatientName", "StudyDate")
        label: Nhãn hiển thị cho tag (vd: "Tên", "Ngày")

    Returns:
        Chuỗi đã định dạng dạng "Nhãn: Giá trị"
    """
    # Lấy giá trị, nếu không có thì trả về "N/A"
    value = dataset.get(tag, "N/A")
    return f"{label}: {value}"


def render_metadata(dataset: Dataset) -> None:
    """
    Hiển thị metadata (thông tin mô tả) của file DICOM trên giao diện web.
    Metadata được chia làm 3 cột:
    - Thông tin bệnh nhân
    - Thông tin nghiên cứu/ca chụp
    - Thông tin cơ sở y tế

    Args:
        dataset: Dataset DICOM chứa metadata cần hiển thị
    """
    # Tạo 3 cột để hiển thị thông tin
    column_patient, column_study, column_site = st.columns(3)

    # Cột 1: Thông tin bệnh nhân
    with column_patient:
        st.markdown("**Bệnh nhân**")
        st.text(format_tag(dataset, "PatientName", "Tên"))
        st.text(format_tag(dataset, "PatientID", "ID"))
        st.text(format_tag(dataset, "PatientBirthDate", "Ngày sinh"))

    # Cột 2: Thông tin nghiên cứu/ca chụp
    with column_study:
        st.markdown("**Nghiên cứu**")
        st.text(format_tag(dataset, "StudyDate", "Ngày"))
        st.text(format_tag(dataset, "StudyTime", "Giờ"))
        st.text(format_tag(dataset, "Modality", "Phương thức"))

    # Cột 3: Thông tin cơ sở y tế
    with column_site:
        st.markdown("**Cơ sở**")
        st.text(format_tag(dataset, "InstitutionName", "Tên"))
        st.text(format_tag(dataset, "StationName", "Trạm"))


def show_mapping(mapping: Dict[str, str]) -> None:
    """
    Hiển thị bảng ánh xạ giữa ID gốc và ID ẩn danh của bệnh nhân.
    Cho phép tải bảng ánh xạ dưới dạng file CSV.

    Args:
        mapping: Dictionary ánh xạ {ID_gốc: ID_ẩn_danh}
    """
    # Nếu không có dữ liệu ánh xạ thì thoát
    if not mapping:
        return

    st.subheader("Bảng ánh xạ ID")
    # Tạo DataFrame từ dictionary để hiển thị dạng bảng
    frame = pd.DataFrame(
        {
            "ID Gốc": list(mapping.keys()),
            "ID Ẩn danh": list(mapping.values()),
        }
    )
    # Hiển thị bảng trên web với chiều rộng đầy đủ
    st.dataframe(frame, use_container_width=True)

    # Chuyển DataFrame thành CSV để tải về
    csv_bytes = frame.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Tải bảng ánh xạ ID (CSV)",
        data=csv_bytes,
        file_name="bang_anh_xa_id.csv",
        mime="text/csv",
    )


def download_anonymized(output_dir: Path) -> None:
    """
    Tạo file ZIP chứa tất cả các file DICOM đã ẩn danh
    và cho phép người dùng tải về.

    Args:
        output_dir: Đường dẫn thư mục chứa các file DICOM đã ẩn danh
    """
    # Tạo buffer trong bộ nhớ để lưu file ZIP
    zip_buffer = io.BytesIO()
    # Tạo file ZIP với chế độ nén DEFLATED
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        # Duyệt qua tất cả file .dcm trong thư mục output
        for file_path in output_dir.glob("*.dcm"):
            # Thêm file vào ZIP
            archive.write(file_path, file_path.name)

    # Di chuyển con trỏ về đầu buffer để đọc
    zip_buffer.seek(0)
    # Tạo nút tải về file ZIP
    st.download_button(
        label="Tải file đã ẩn danh (ZIP)",
        data=zip_buffer,
        file_name="dicom_da_an_danh.zip",
        mime="application/zip",
        use_container_width=True,
    )


# ==================== CẤU HÌNH TRANG WEB ====================
# Cấu hình metadata của trang web: tiêu đề tab và layout rộng
st.set_page_config(page_title="Ẩn danh hóa DICOM", layout="wide")
st.title("Ẩn danh hóa DICOM")
st.markdown("Xóa thông tin bệnh nhân khỏi file DICOM để bảo mật dữ liệu y tế.")

# ==================== THÔNG TIN HƯỚNG DẪN ====================
# Tạo expander (phần có thể mở rộng/thu gọn)
# để hiển thị danh sách thông tin sẽ bị xóa
with st.expander("Những thông tin nào sẽ bị xóa?"):
    column_left, column_right = st.columns(2)
    with column_left:
        st.markdown(
            """
            **Thông tin bệnh nhân**
            - Tên và mã định danh
            - Ngày sinh, tuổi, giới tính
            - Địa chỉ và liên lạc
            """
        )
    with column_right:
        st.markdown(
            """
            **Thông tin nghiên cứu**
            - Ngày giờ nghiên cứu
            - Tên cơ sở y tế
            - Bác sĩ giới thiệu
            """
        )

# ==================== THANH SIDEBAR (CÀI ĐẶT) ====================
st.sidebar.header("Cài đặt")
# Cho phép người dùng nhập tiền tố cho ID ẩn danh (mặc định "ANON")
patient_prefix = st.sidebar.text_input(
    "Tiền tố ID ẩn danh",
    value="ANON",
    help="Tiền tố cho mã định danh được tạo tự động.",
)
st.sidebar.info(" File sẽ được ẩn danh và trả về dưới dạng file ZIP.")

# ==================== PHẦN TẢI LÊN FILE ====================
st.subheader("Tải lên file DICOM")
# Widget cho phép người dùng tải lên một hoặc nhiều file DICOM
uploads = st.file_uploader(
    "Chọn file DICOM",
    type=["dcm"],  # Chỉ chấp nhận file có đuôi .dcm
    accept_multiple_files=True,  # Cho phép tải nhiều file cùng lúc
    help="Bạn có thể tải lên một hoặc nhiều file DICOM.",
)

# ==================== XỬ LÝ KHI CÓ FILE ĐƯỢC TẢI LÊN ====================
if uploads:
    # Hiển thị thông báo số lượng file đã nhận
    st.success(f"Đã nhận {len(uploads)} file.")

    # Cố gắng đọc và hiển thị metadata của file đầu tiên để xem trước
    try:
        # Chuyển file thành BytesIO để đọc
        file_bytes = io.BytesIO(uploads[0].getvalue())
        # Đọc dataset DICOM
        preview = pydicom.dcmread(file_bytes, force=True)
        # Hiển thị metadata lên web
        render_metadata(preview)
    except Exception as exc:  # pylint: disable=broad-exception-caught
        st.warning(f"Không thể đọc metadata: {exc}")

    st.markdown("---")  # Đường kẻ ngang phân cách

    # ==================== NÚT BẮT ĐẦU ẨN DANH HÓA ====================
    # Nút bấm để bắt đầu quá trình ẩn danh hóa
    if st.button("Ẩn danh hóa file", use_container_width=True, type="primary"):
        # Hiển thị spinner (biểu tượng xoay) trong khi xử lý
        with st.spinner("Đang ẩn danh hóa file..."):
            try:
                # ===== BƯỚC 1: TẠO THỦ MỤC TẠM =====
                # Tạo thư mục tạm để lưu file input và output
                with tempfile.TemporaryDirectory() as tmp_dir:
                    tmp_root = Path(tmp_dir)
                    input_dir = tmp_root / "input"  # Thư mục chứa file gốc
                    output_dir = tmp_root / "output"  # Thư mục chứa đã ẩn
                    input_dir.mkdir()
                    output_dir.mkdir()

                    # ===== BƯỚC 2: LƯU FILE TẢI LÊN VÀO THƯ MỤC INPUT =====
                    # Tạo thanh progress để theo dõi tiến độ
                    progress = st.progress(0)
                    total = len(uploads) or 1
                    # Duyệt qua từng file tải lên và lưu vào thư mục input
                    for index, upload in enumerate(uploads, start=1):
                        target_path = input_dir / upload.name
                        target_path.write_bytes(upload.getvalue())
                        progress.progress(index / total)

                    # ===== BƯỚC 3: THỰC HIỆN ẨN DANH HÓA =====
                    # Import locally để tránh điều chỉnh đường dẫn module
                    # pylint: disable=import-outside-toplevel
                    from src.anonymization.dicom_anonymizer import (
                        DICOMAnonymizer,
                    )

                    # Tạo đối tượng anonymizer với tiền tố đã chọn
                    anonymizer = DICOMAnonymizer(prefix=patient_prefix)
                    # Thực hiện ẩn danh hóa toàn bộ thư mục
                    stats = anonymizer.anonymize_directory(
                        str(input_dir),
                        str(output_dir),
                    )
                    # Cập nhật progress bar lên 100%
                    progress.progress(1.0)

                    # ===== BƯỚC 4: LẤY KẾT QUẢ THỐNG KÊ =====
                    successes = int(stats.get("successful", 0))
                    failures = int(stats.get("failed", 0))
                    mapping = stats.get("id_mapping", {})

                    # Hiển thị thông báo kết quả
                    message = (
                        "Ẩn danh hóa hoàn tất. "
                        f"Thành công: {successes} | "
                        f"Thất bại: {failures} | "
                        f"Số bệnh nhân: {len(mapping)}"
                    )
                    st.success(message)

                    # ===== BƯỚC 5: HIỂN THỊ BẢNG ÁNH XẠ =====
                    # Hiển thị bảng ánh xạ giữa ID gốc và ID ẩn danh
                    show_mapping(mapping)
                    st.markdown("---")

                    # ===== BƯỚC 6: CHO PHÉP TẢI VỀ FILE ĐÃ ẨN DANH =====
                    st.subheader("Tải file đã ẩn danh")
                    download_anonymized(output_dir)

                    # ===== BƯỚC 7: XEM TRƯỚC METADATA ĐÃ ẨN DANH =====
                    # Lấy danh sách các file đã ẩn danh
                    anonymized_files = list(output_dir.glob("*.dcm"))
                    if anonymized_files:
                        st.markdown("---")
                        st.subheader("Xem trước metadata đã ẩn danh")
                        # Đọc và hiển thị metadata của file đầu tiên
                        first_file = str(anonymized_files[0])
                        preview_dataset = pydicom.dcmread(first_file)
                        render_metadata(preview_dataset)
                        st.success("File đã không còn thông tin nhận dạng cá nhân.")

                        # ===== BƯỚC 8: PHẦN GIẢI THÍCH KẾT QUẢ =====
                        st.markdown("---")
                        st.subheader("Giải thích kết quả ẩn danh hóa")

                        # Danh sách các trường thông tin đã bị xóa
                        removed_fields = [
                            "PatientName",
                            "PatientID",
                            "PatientBirthDate",
                            "PatientAge",
                            "PatientSex",
                            "PatientAddress",
                            "ReferringPhysicianName",
                            "InstitutionName",
                            "InstitutionAddress",
                            "StationName",
                        ]

                        # Hiển thị phần giải thích chi tiết
                        show_interpretation_section(
                            task_type="anonymization",
                            metrics={},
                            image_info={
                                "num_files": successes,
                                "num_patients": len(mapping),
                                "fields_removed": removed_fields,
                                "prefix": patient_prefix,
                            },
                        )
            except Exception as exc:  # pylint: disable=broad-exception-caught
                # Bắt và hiển thị lỗi nếu có
                st.error(f"Đã xảy ra lỗi: {exc}")
else:
    # ==================== HIỂN THỊ KHI CHƯA CÓ FILE TẢI LÊN ====================
    st.info("Tải lên một hoặc nhiều file DICOM để bắt đầu.")
    st.markdown("---")
    st.subheader("Hướng dẫn nhanh")
    st.markdown(
        """
        1. Nhấn "Browse files" và chọn file DICOM.
        2. Xem trước metadata của file.
        3. Nhấn "Ẩn danh hóa file" để xử lý.
        4. Tải về file ZIP và bảng ánh xạ ID.
        """
    )

# ==================== FOOTER/LƯU Ý ====================
st.markdown("---")
st.caption(
    "Lưu ý: Giữ bảng ánh xạ ID riêng biệt với file đã ẩn danh "
    "để tuân thủ quy định bảo mật dữ liệu y tế."
)
