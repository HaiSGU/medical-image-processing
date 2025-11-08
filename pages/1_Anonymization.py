"""Trang Streamlit để ẩn danh hóa các file DICOM."""

import io
import sys
import tempfile
import zipfile
from pathlib import Path
from typing import Dict

import pandas as pd
import pydicom
import streamlit as st
from pydicom.dataset import Dataset

PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def format_tag(dataset: Dataset, tag: str, label: str) -> str:
    value = dataset.get(tag, "N/A")
    return f"{label}: {value}"


def render_metadata(dataset: Dataset) -> None:
    column_patient, column_study, column_site = st.columns(3)

    with column_patient:
        st.markdown("**Bệnh nhân**")
        st.text(format_tag(dataset, "PatientName", "Tên"))
        st.text(format_tag(dataset, "PatientID", "ID"))
        st.text(format_tag(dataset, "PatientBirthDate", "Ngày sinh"))

    with column_study:
        st.markdown("**Nghiên cứu**")
        st.text(format_tag(dataset, "StudyDate", "Ngày"))
        st.text(format_tag(dataset, "StudyTime", "Giờ"))
        st.text(format_tag(dataset, "Modality", "Phương thức"))

    with column_site:
        st.markdown("**Cơ sở**")
        st.text(format_tag(dataset, "InstitutionName", "Tên"))
        st.text(format_tag(dataset, "StationName", "Trạm"))


def show_mapping(mapping: Dict[str, str]) -> None:
    if not mapping:
        return

    st.subheader("🔑 Bảng ánh xạ ID")
    frame = pd.DataFrame(
        {
            "ID Gốc": list(mapping.keys()),
            "ID Ẩn danh": list(mapping.values()),
        }
    )
    st.dataframe(frame, use_container_width=True)

    csv_bytes = frame.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="📥 Tải bảng ánh xạ ID (CSV)",
        data=csv_bytes,
        file_name="bang_anh_xa_id.csv",
        mime="text/csv",
    )


def download_anonymized(output_dir: Path) -> None:
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for file_path in output_dir.glob("*.dcm"):
            archive.write(file_path, file_path.name)

    zip_buffer.seek(0)
    st.download_button(
        label="📥 Tải file đã ẩn danh (ZIP)",
        data=zip_buffer,
        file_name="dicom_da_an_danh.zip",
        mime="application/zip",
        use_container_width=True,
    )


st.set_page_config(page_title="🔒 Ẩn danh hóa DICOM", layout="wide")
st.title("🔒 Ẩn danh hóa DICOM")
st.markdown("Xóa thông tin bệnh nhân khỏi file DICOM để bảo mật dữ liệu y tế.")

with st.expander("📋 Những thông tin nào sẽ bị xóa?"):
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

st.sidebar.header("⚙️ Cài đặt")
patient_prefix = st.sidebar.text_input(
    "Tiền tố ID ẩn danh",
    value="ANON",
    help="Tiền tố cho mã định danh được tạo tự động.",
)
st.sidebar.info("File sẽ được ẩn danh và trả về dưới dạng file ZIP.")

st.subheader("📤 Tải lên file DICOM")
uploads = st.file_uploader(
    "Chọn file DICOM",
    type=["dcm"],
    accept_multiple_files=True,
    help="Bạn có thể tải lên một hoặc nhiều file DICOM.",
)

if uploads:
    st.success(f"✅ Đã nhận {len(uploads)} file.")

    try:
        preview = pydicom.dcmread(io.BytesIO(uploads[0].getvalue()), force=True)
        render_metadata(preview)
    except Exception as exc:  # pylint: disable=broad-exception-caught
        st.warning(f"⚠️ Không thể đọc metadata: {exc}")

    st.markdown("---")

    if st.button("🔒 Ẩn danh hóa file", use_container_width=True, type="primary"):
        with st.spinner("Đang ẩn danh hóa file..."):
            try:
                with tempfile.TemporaryDirectory() as tmp_dir:
                    tmp_root = Path(tmp_dir)
                    input_dir = tmp_root / "input"
                    output_dir = tmp_root / "output"
                    input_dir.mkdir()
                    output_dir.mkdir()

                    progress = st.progress(0)
                    total = len(uploads) or 1
                    for index, upload in enumerate(uploads, start=1):
                        target_path = input_dir / upload.name
                        target_path.write_bytes(upload.getvalue())
                        progress.progress(index / total)

                    # Import locally to avoid module-level path adjustments.
                    # pylint: disable=import-outside-toplevel
                    from src.anonymization.dicom_anonymizer import (
                        DICOMAnonymizer,
                    )

                    anonymizer = DICOMAnonymizer(prefix=patient_prefix)
                    stats = anonymizer.anonymize_directory(
                        str(input_dir),
                        str(output_dir),
                    )
                    progress.progress(1.0)

                    successes = int(stats.get("successful", 0))
                    failures = int(stats.get("failed", 0))
                    mapping = stats.get("id_mapping", {})

                    message = (
                        "Ẩn danh hóa hoàn tất. Thành công: {} | Thất bại: {} | "
                        "Số bệnh nhân: {}"
                    ).format(successes, failures, len(mapping))
                    st.success(message)

                    show_mapping(mapping)
                    st.markdown("---")
                    st.subheader("📥 Tải file đã ẩn danh")
                    download_anonymized(output_dir)

                    anonymized_files = list(output_dir.glob("*.dcm"))
                    if anonymized_files:
                        st.markdown("---")
                        st.subheader("👁️ Xem trước metadata đã ẩn danh")
                        preview_dataset = pydicom.dcmread(str(anonymized_files[0]))
                        render_metadata(preview_dataset)
                        st.success("✅ File đã không còn thông tin nhận dạng cá nhân.")
            except Exception as exc:  # pylint: disable=broad-exception-caught
                st.error(f"❌ Đã xảy ra lỗi: {exc}")
else:
    st.info("👆 Tải lên một hoặc nhiều file DICOM để bắt đầu.")
    st.markdown("---")
    st.subheader("📖 Hướng dẫn nhanh")
    st.markdown(
        """
        1. Nhấn "Browse files" và chọn file DICOM.
        2. Xem trước metadata của file.
        3. Nhấn "Ẩn danh hóa file" để xử lý.
        4. Tải về file ZIP và bảng ánh xạ ID.
        """
    )

st.markdown("---")
st.caption(
    "💡 Lưu ý: Giữ bảng ánh xạ ID riêng biệt với file đã ẩn danh để tuân thủ quy định."
)
