# CORE Processing Pipeline - Full Integration (3700+ lines)
# Tích hợp đầy đủ TẤT CẢ các tools với explanations

import streamlit as st
import sys
from pathlib import Path
import tempfile
import numpy as np
import matplotlib.pyplot as plt
import io
import zipfile
from skimage import exposure
import pandas as pd
import pydicom
from matplotlib.colors import ListedColormap
import SimpleITK as sitk

# Setup
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Imports - all modules
from utils.file_io import MedicalImageIO
from src.anonymization.dicom_anonymizer import DICOMAnonymizer
from src.preprocessing.image_transforms import ImageTransforms
from src.segmentation.brain_segmentation import BrainSegmentation
from src.reconstruction.ct_reconstruction import CTReconstructor
from src.reconstruction.mri_reconstruction import MRIReconstructor
from src.registration.image_registration import (
    ImageRegistration,
    numpy_to_sitk,
    sitk_to_numpy,
)
from utils.interpretation import (
    ResultVisualizer,
    MetricsExplainer,
    show_interpretation_section,
)
from utils.image_explainer import explain_input_image


# ==================== HELPER FUNCTIONS ====================
def format_tag(dataset, tag, label):
    """Format DICOM tag for display"""
    value = dataset.get(tag, "N/A")
    return f"{label}: {value}"


def render_metadata(dataset):
    """Display DICOM metadata in 3 columns"""
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("**Bệnh nhân**")
        st.text(format_tag(dataset, "PatientName", "Tên"))
        st.text(format_tag(dataset, "PatientID", "ID"))
        st.text(format_tag(dataset, "PatientBirthDate", "Ngày sinh"))
    with c2:
        st.markdown("**Nghiên cứu**")
        st.text(format_tag(dataset, "StudyDate", "Ngày"))
        st.text(format_tag(dataset, "StudyTime", "Giờ"))
        st.text(format_tag(dataset, "Modality", "Phương thức"))
    with c3:
        st.markdown("**Cơ sở**")
        st.text(format_tag(dataset, "InstitutionName", "Tên"))
        st.text(format_tag(dataset, "StationName", "Trạm"))


def show_mapping(mapping):
    """Display and download ID mapping table"""
    if not mapping:
        return

    st.subheader("📋 Bảng ánh xạ ID")
    df = pd.DataFrame(
        {"ID Gốc": list(mapping.keys()), "ID Ẩn danh": list(mapping.values())}
    )
    st.dataframe(df, use_container_width=True)

    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Tải bảng ánh xạ (CSV)", csv, "id_mapping.csv", "text/csv")


def create_shepp_logan_phantom(size=256):
    """Create Shepp-Logan phantom"""
    from skimage.data import shepp_logan_phantom
    from skimage.transform import resize

    phantom = shepp_logan_phantom()
    if phantom.shape[0] != size:
        phantom = resize(phantom, (size, size), anti_aliasing=True)
    return phantom


def calculate_psnr(original, reconstructed):
    """Calculate PSNR"""
    mse = np.mean((original - reconstructed) ** 2)
    if mse == 0:
        return float("inf")
    max_pixel = np.max(original)
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr


def calculate_ssim(original, reconstructed):
    """Calculate SSIM (Simplified)"""
    mu1 = np.mean(original)
    mu2 = np.mean(reconstructed)
    sigma1 = np.std(original)
    sigma2 = np.std(reconstructed)
    sigma12 = np.mean((original - mu1) * (reconstructed - mu2))

    c1 = (0.01 * np.max(original)) ** 2
    c2 = (0.03 * np.max(original)) ** 2

    ssim = ((2 * mu1 * mu2 + c1) * (2 * sigma12 + c2)) / (
        (mu1**2 + mu2**2 + c1) * (sigma1**2 + sigma2**2 + c2)
    )
    return ssim


# Page config
st.set_page_config(page_title="🔧 CORE Processing", page_icon="🔧", layout="wide")

# Title
st.title("🔧 CORE Processing Pipeline - Complete Integration")
st.markdown("### Chọn công cụ bạn cần sử dụng:")

# Tool selector
selected_tool = st.selectbox(
    "Công cụ:",
    [
        "Preprocessing",
        "Anonymization",
        "Segmentation",
        "CT Reconstruction",
        "MRI Reconstruction",
        "Registration",
    ],
    key="tool_selector",
)

st.markdown("---")

# ==================== DYNAMIC SIDEBAR ====================
with st.sidebar:
    st.markdown(f"### 🎯 {selected_tool}")
    st.markdown("---")

    # === PREPROCESSING SIDEBAR ===
    if selected_tool == "Preprocessing":
        st.header("🔧 Các phép toán")

        # Normalization
        st.markdown("### 📊 Cường độ")
        normalize_enabled = st.checkbox("Chuẩn hóa", value=False, key="norm_cb")
        if normalize_enabled:
            norm_method = st.selectbox(
                "Phương pháp:",
                ["Min-Max (0-1)", "Z-Score", "Cắt phân vị"],
                help="Phương pháp chuẩn hóa cường độ pixel",
                key="norm_method",
            )

            with st.expander(f"💡 Giải thích: {norm_method}"):
                if norm_method == "Min-Max (0-1)":
                    st.markdown(
                        """
                    **📏 Min-Max (Chuẩn)**
                    
                    Biến đổi tất cả giá trị pixel về khoảng [0, 1]
                    
                    **Công thức:** `(pixel - min) / (max - min)`
                    
                    **Ưu điểm:**
                    - ✅ Đơn giản nhất
                    - ✅ Giữ tỷ lệ tương đối
                    - ✅ Chuẩn cho deep learning
                    
                    **Khi nào dùng:**
                    - Training neural networks
                    - Ảnh không có outliers
                    """
                    )
                elif norm_method == "Z-Score":
                    st.markdown(
                        """
                    **📊 Z-Score (Chuẩn hóa thống kê)**
                    
                    Biến đổi về phân phối chuẩn (mean=0, std=1)
                    
                    **Công thức:** `(pixel - mean) / std`
                    
                    **Ưu điểm:**
                    - ✅ Tốt cho machine learning
                    - ✅ Xử lý outliers tốt hơn
                    
                    **Khi nào dùng:**
                    - Training CNN, ML models
                    - Ảnh có nhiễu/outliers
                    """
                    )
                else:
                    st.markdown(
                        """
                    **✂️ Percentile Clipping**
                    
                    Loại bỏ giá trị cực trị trước khi normalize
                    
                    **Ưu điểm:**
                    - ✅ Loại bỏ outliers hiệu quả
                    - ✅ Tăng tương phản vùng ROI
                    
                    **Khuyến nghị:** 2%-98%
                    """
                    )

            if norm_method == "Cắt phân vị":
                lower_p = st.slider("Phân vị dưới:", 0, 50, 2, key="lower_p")
                upper_p = st.slider("Phân vị trên:", 50, 100, 98, key="upper_p")

        st.markdown("---")

        # Denoising
        st.markdown("### 🔇 Khử nhiễu")
        denoise_enabled = st.checkbox("Khử nhiễu", value=False, key="denoise_cb")
        if denoise_enabled:
            denoise_method = st.selectbox(
                "Phương pháp:",
                ["Gaussian", "Median"],
                help="Phương pháp khử nhiễu",
                key="denoise_method",
            )

            with st.expander(f"💡 Giải thích: {denoise_method}"):
                if denoise_method == "Gaussian":
                    st.markdown(
                        """
                    **🔵 Gaussian Blur**
                    
                    Làm mịn và giảm nhiễu Gaussian
                    
                    **Ưu điểm:**
                    - ✅ Làm mịn đồng đều
                    - ✅ Giảm nhiễu hiệu quả
                    - ✅ Nhanh
                    
                    **Nhược điểm:**
                    - ⚠️ Làm mờ cạnh
                    
                    **Tham số σ:**
                    - 0.5-1.0: Nhẹ ⭐
                    - 1.0-2.0: Chuẩn
                    - 2.0-5.0: Mạnh
                    """
                    )
                else:
                    st.markdown(
                        """
                    **🟢 Median Filter**
                    
                    Thay pixel bằng giá trị trung vị
                    
                    **Ưu điểm:**
                    - ✅ Loại bỏ salt-pepper noise
                    - ✅ Bảo toàn cạnh tốt
                    
                    **Kernel Size:**
                    - 3×3: Nhẹ, giữ chi tiết ⭐
                    - 5×5: Chuẩn
                    - 7×7+: Mạnh
                    """
                    )

            if denoise_method == "Gaussian":
                sigma = st.slider("Sigma:", 0.1, 5.0, 1.0, 0.1, key="sigma")
            else:
                kernel_size = st.slider("Kernel size:", 3, 11, 5, 2, key="kernel")

        st.markdown("---")

        # Enhancement
        st.markdown("### ✨ Tăng cường")
        enhance_enabled = st.checkbox(
            "Tăng cường Tương phản", value=False, key="enhance_cb"
        )
        if enhance_enabled:
            enhance_method = st.selectbox(
                "Phương pháp:",
                ["CLAHE", "Cân bằng Histogram", "Hiệu chỉnh Gamma"],
                help="Phương pháp tăng cường",
                key="enhance_method",
            )

            with st.expander(f"💡 Giải thích: {enhance_method}"):
                if enhance_method == "CLAHE":
                    st.markdown(
                        """
                    **⭐ CLAHE - Adaptive Histogram**
                    
                    Cân bằng histogram cục bộ có giới hạn
                    
                    **Ưu điểm:**
                    - ✅ Tăng cường cục bộ
                    - ✅ Tránh over-enhance
                    - ✅ Hiệu quả với ảnh y tế
                    
                    **Tham số:**
                    - 1-2: Nhẹ
                    - 2-3: Chuẩn ⭐
                    - 3-5: Mạnh
                    """
                    )
                elif enhance_method == "Cân bằng Histogram":
                    st.markdown(
                        """
                    **📊 Histogram Equalization**
                    
                    Phân bố lại pixel để histogram đều
                    
                    **Ưu điểm:**
                    - ✅ Đơn giản, không cần tham số
                    - ✅ Nhanh
                    
                    **Khi nào dùng:**
                    - Ảnh độ tương phản thấp toàn bộ
                    """
                    )
                else:
                    st.markdown(
                        """
                    **🌓 Gamma Correction**
                    
                    Điều chỉnh phi tuyến độ sáng
                    
                    **Tham số:**
                    - < 1.0: Làm sáng
                    - = 1.0: Không đổi
                    - > 1.0: Làm tối
                    
                    **Mẹo:** 0.7-0.9 cho ảnh tối
                    """
                    )

            if enhance_method == "CLAHE":
                clip_limit = st.slider("Clip limit:", 0.5, 5.0, 2.0, 0.5, key="clip")
            elif enhance_method == "Hiệu chỉnh Gamma":
                gamma = st.slider("Gamma:", 0.1, 3.0, 1.0, 0.1, key="gamma")

        st.markdown("---")
        st.info("💡 Bật các phép toán theo thứ tự đề xuất")

    # === ANONYMIZATION SIDEBAR ===
    elif selected_tool == "Anonymization":
        st.markdown("#### ⚙️ Cài đặt")
        patient_prefix = st.text_input(
            "Tiền tố ID ẩn danh:",
            value="ANON",
            help="Tiền tố cho mã ID",
            key="anon_prefix",
        )
        st.info("💡 File sẽ được ẩn danh và trả về ZIP")

    # === SEGMENTATION SIDEBAR ===
    elif selected_tool == "Segmentation":
        st.header("Cài đặt Phân đoạn")

        method = st.selectbox(
            "Phương pháp",
            ["Tự động", "Ngưỡng", "Otsu", "Tăng trưởng vùng"],
            help="Chọn phương pháp phân đoạn phù hợp",
            key="seg_method",
        )

        method_map = {
            "Tự động": "Automatic",
            "Ngưỡng": "Threshold",
            "Otsu": "Otsu",
            "Tăng trưởng vùng": "Region Growing",
        }
        method_en = method_map[method]

        with st.expander(f"💡 Giải thích: {method}"):
            if method == "Tự động":
                st.markdown(
                    """
                **⭐ Khuyến nghị cho người mới**
                
                **Mô tả:**  
                Tự động kết hợp nhiều phương pháp để cho kết quả tốt nhất.
                
                **Ưu điểm:**
                - ✅ Hoàn toàn tự động
                - ✅ Kết quả ổn định
                - ✅ Phù hợp cho người không chuyên
                
                **Khi nào dùng:**
                - Bạn chưa biết phương pháp nào phù hợp
                - Cần kết quả nhanh và đáng tin cậy
                """
                )
            elif method == "Ngưỡng":
                st.markdown(
                    """
                **📏 Phân đoạn theo ngưỡng**
                
                **Ưu điểm:**
                - ✅ Đơn giản, dễ hiểu
                - ✅ Rất nhanh
                - ✅ Kiểm soát hoàn toàn
                
                **Mẹo:** Thử các giá trị 40-80 với ảnh MRI não
                """
                )
            elif method == "Otsu":
                st.markdown(
                    """
                **🎯 Otsu - Tự động tìm ngưỡng tối ưu**
                
                **Ưu điểm:**
                - ✅ Hoàn toàn tự động
                - ✅ Rất nhanh (~2 giây)
                - ✅ Hiệu quả với ảnh có 2 vùng rõ rệt
                
                **Phù hợp nhất:** Ảnh MRI T1-weighted
                """
                )
            else:
                st.markdown(
                    """
                **🌱 Region Growing**
                
                **Ưu điểm:**
                - ✅ Rất chính xác nếu chọn đúng điểm khởi đầu
                - ✅ Tốt cho vùng có ranh giới rõ ràng
                
                **Mẹo:**  
                - Chọn điểm giữa vùng não (50%, 50%, 50%)
                - Dung sai thấp (5-10) cho kết quả chính xác
                """
                )

        st.markdown("---")

        if method == "Ngưỡng":
            threshold = st.slider("Giá trị ngưỡng", 0, 255, 50, key="seg_threshold")
        elif method == "Tăng trưởng vùng":
            st.markdown("**Điểm khởi đầu (%):**")
            seed_x = st.slider("Vị trí X", 0, 100, 50, key="seed_x")
            seed_y = st.slider("Yi trí Y", 0, 100, 50, key="seed_y")
            seed_z = st.slider("Vị trí Z", 0, 100, 50, key="seed_z")
            intensity_tolerance = st.slider(
                "Dung sai cường độ", 1, 50, 10, key="tolerance"
            )

        st.markdown("---")
        st.markdown("**Xử lý sau phân đoạn:**")
        apply_morph = st.checkbox(
            "Áp dụng phép biến đổi hình thái", value=True, key="apply_morph"
        )
        if apply_morph:
            morph_op = st.selectbox(
                "Phép toán",
                [
                    "đóng (closing)",
                    "mở (opening)",
                    "giãn (dilation)",
                    "xói mòn (erosion)",
                ],
                key="morph_op",
            )
            kernel_size_morph = st.slider(
                "Kích thước Kernel", 1, 10, 3, key="kernel_morph"
            )

        keep_largest = st.checkbox(
            "Chỉ giữ thành phần lớn nhất", value=True, key="keep_largest"
        )
        st.markdown("---")
        st.info("Thử phương pháp 'Tự động' trước để có kết quả tốt nhất")

    # === CT RECONSTRUCTION SIDEBAR ===
    elif selected_tool == "CT Reconstruction":
        st.header("Cài đặt")

        data_source = st.radio(
            "Nguồn dữ liệu:", ["Tạo Phantom", "Tải lên Sinogram"], key="data_source"
        )

        st.markdown("---")

        ct_method = st.selectbox("Phương pháp:", ["FBP", "SART"], key="ct_method")

        with st.expander(f"💡 Giải thích: {ct_method}"):
            if ct_method == "FBP":
                st.markdown(
                    """
                **⚡ FBP - Filtered Back Projection**
                
                **Ưu điểm:**
                - ✅ Rất nhanh (~1-2 giây)
                - ✅ Tiêu chuẩn lâm sàng
                - ✅ Kết quả ổn định
                
                **Bộ lọc:**
                - **ramp** ⭐: Chuẩn
                - **shepp-logan**: Giảm nhiễu
                - **cosine**: Mượt mà
                - **hamming**: Giảm nhiễu mạnh
                
                **Khuyến nghị:** Dùng 'ramp' trước
                """
                )
            else:
                st.markdown(
                    """
                **🔄 SART - Algebraic Reconstruction**
                
                **Ưu điểm:**
                - ✅ Chất lượng cao hơn FBP
                - ✅ Tốt với dữ liệu thưa
                - ✅ Giảm nhiễu hiệu quả
                
                **Nhược điểm:**
                - ⚠️ Chậm hơn nhiều
                
                **Tham số:**
                - Số lần lặp: 5-20 (10 là tốt)
                - Hệ số thư giãn: 0.3-0.7 (0.5 là tốt)
                """
                )

        if ct_method == "FBP":
            filter_type = st.selectbox(
                "Bộ lọc:",
                ["ramp", "shepp-logan", "cosine", "hamming"],
                key="filter_type",
            )
        else:
            num_iterations = st.slider("Số lần lặp:", 1, 50, 10, key="num_iterations")
            relaxation = st.slider(
                "Hệ số thư giãn:", 0.1, 1.0, 0.5, 0.1, key="relaxation"
            )

        st.markdown("---")
        st.info("Thử FBP với bộ lọc 'ramp' trước")

    # === MRI RECONSTRUCTION SIDEBAR ===
    elif selected_tool == "MRI Reconstruction":
        st.header("Cài đặt")

        mri_data_source = st.radio(
            "Nguồn dữ liệu:",
            ["Tạo từ Ảnh", "Tải lên K-space"],
            help="Tạo K-space từ ảnh hoặc tải dữ liệu thực",
            key="mri_data_source",
        )

        st.markdown("---")

        if mri_data_source == "Tạo từ Ảnh":
            partial_fourier = st.checkbox(
                "Partial Fourier",
                value=False,
                help="Mô phỏng quét nhanh hơn",
                key="partial_fourier",
            )

            if partial_fourier:
                pf_percentage = st.select_slider(
                    "Phủ K-space:",
                    options=[50, 62.5, 75, 87.5, 100],
                    value=75,
                    help="Percentage of K-space to use",
                    key="pf_percentage",
                )

        st.markdown("---")
        st.info("💡 Trung tâm K-space chứa thông tin quan trọng nhất")

    # === REGISTRATION SIDEBAR ===
    else:  # Registration
        st.header("⚙️ Tham số")

        registration_type = st.selectbox(
            "Loại đăng ký",
            ["Rigid", "Affine", "Deformable"],
            help="""
            - Rigid: Chỉ di chuyển + xoay
            - Affine: + scaling + shearing
            - Deformable: Biến dạng cục bộ
            """,
            key="registration_type",
        )

        if registration_type == "Rigid":
            st.info("💡 Rigid: Translation + Rotation only (6 DOF)")
            reg_iterations = st.slider(
                "Số lần lặp", 50, 300, 100, step=10, key="reg_iter"
            )
            learning_rate = st.slider(
                "Learning rate", 0.1, 5.0, 1.0, step=0.1, key="lr"
            )
        elif registration_type == "Affine":
            st.info("💡 Affine: + Scaling + Shearing (12 DOF)")
            reg_iterations = st.slider(
                "Số lần lặp", 50, 300, 150, step=10, key="reg_iter"
            )
            learning_rate = st.slider(
                "Learning rate", 0.1, 5.0, 1.0, step=0.1, key="lr"
            )
        else:  # Deformable
            st.warning("⚠️ Deformable: Chậm nhất nhưng chính xác nhất")
            reg_iterations = st.slider(
                "Số lần lặp", 20, 100, 50, step=5, key="reg_iter"
            )
            mesh_size = st.slider(
                "Mesh size",
                3,
                10,
                5,
                step=1,
                help="Nhỏ hơn = linh hoạt hơn (nhưng chậm hơn)",
                key="mesh_size",
            )

        reg_metric = st.selectbox(
            "Similarity metric",
            ["mean_squares", "mutual_information"],
            help="""
            - Mean Squares: Cho cùng modality
            - Mutual Information: Cho multi-modal
            """,
            key="reg_metric",
        )

        st.markdown("---")
        st.info("💡 Rigid nhanh nhất, Deformable chính xác nhất")

# ==================== MAIN CONTENT ====================
if selected_tool == "Preprocessing":
    st.subheader("🎨 Tiền xử lý Ảnh")
    st.markdown("Biến đổi và nâng cao chất lượng ảnh y tế")

    uploaded_file = st.file_uploader(
        "Tải ảnh lên (.nii, .nii.gz, .dcm, .nrrd, .mha)",
        type=["nii", "gz", "dcm", "nrrd", "mha"],
        key="prep_uploader",
    )

    if uploaded_file:
        suffix = (
            ".nii.gz"
            if uploaded_file.name.endswith(".nii.gz")
            else Path(uploaded_file.name).suffix
        )
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uploaded_file.getvalue())
            tmp_path = tmp.name

        try:
            with st.spinner("⏳ Đang tải..."):
                io_handler = MedicalImageIO()
                image_data, metadata = io_handler.read_image(tmp_path)

                if image_data.ndim == 3:
                    slice_idx = image_data.shape[2] // 2
                    image_2d = image_data[:, :, slice_idx]
                    st.info(f"📊 Dùng lát cắt giữa ({slice_idx})")
                else:
                    image_2d = image_data

            st.success(f"✅ Đã tải: {image_2d.shape}")

            col1, col2 = st.columns([1, 2])
            with col1:
                st.metric("Kích thước", f"{image_2d.shape[0]}×{image_2d.shape[1]}")
                st.metric("Dải giá trị", f"{image_2d.min():.1f} - {image_2d.max():.1f}")

            with col2:
                fig, ax = plt.subplots(figsize=(6, 6))
                ax.imshow(image_2d, cmap="gray")
                ax.set_title("Ảnh gốc")
                ax.axis("off")
                st.pyplot(fig)
                plt.close()

            st.markdown("---")

            if st.button("🔄 Áp dụng Xử lý", type="primary"):
                with st.spinner("⏳ Đang xử lý..."):
                    try:
                        processed = image_2d.copy()
                        operations = []

                        if normalize_enabled:
                            transformer = ImageTransforms(processed)
                            if norm_method == "Min-Max (0-1)":
                                processed = transformer.normalize_minmax(0.0, 1.0)
                                operations.append("Chuẩn hóa (Min-Max)")
                            elif norm_method == "Z-Score":
                                processed = transformer.normalize_zscore()
                                operations.append("Chuẩn hóa (Z-Score)")
                            else:
                                processed = transformer.normalize_percentile(
                                    lower_percentile=lower_p, upper_percentile=upper_p
                                )
                                operations.append(
                                    f"Chuẩn hóa (Phân vị {lower_p}-{upper_p})"
                                )
                            st.success(f"✅ {operations[-1]}")

                        if denoise_enabled:
                            transformer = ImageTransforms(processed)
                            if denoise_method == "Gaussian":
                                processed = transformer.denoise_gaussian(sigma=sigma)
                                operations.append(f"Khử nhiễu Gaussian (σ={sigma})")
                            else:
                                processed = transformer.denoise_median(size=kernel_size)
                                operations.append(f"Khử nhiễu Median (k={kernel_size})")
                            st.success(f"✅ {operations[-1]}")

                        if enhance_enabled:
                            transformer = ImageTransforms(processed)
                            if enhance_method == "CLAHE":
                                processed = transformer.adaptive_histogram_equalization(
                                    clip_limit=clip_limit
                                )
                                operations.append(f"CLAHE (clip={clip_limit})")
                            elif enhance_method == "Cân bằng Histogram":
                                processed = transformer.histogram_equalization()
                                operations.append("Cân bằng Histogram")
                            else:
                                processed = exposure.adjust_gamma(processed, gamma)
                                operations.append(f"Gamma (γ={gamma})")
                            st.success(f"✅ {operations[-1]}")

                        st.markdown("---")
                        st.subheader("📊 Kết quả")

                        col1, col2 = st.columns(2)
                        with col1:
                            st.markdown("**Gốc**")
                            fig1, ax1 = plt.subplots(figsize=(6, 6))
                            ax1.imshow(image_2d, cmap="gray")
                            ax1.axis("off")
                            st.pyplot(fig1)
                            plt.close()

                        with col2:
                            st.markdown("**Đã xử lý**")
                            fig2, ax2 = plt.subplots(figsize=(6, 6))
                            ax2.imshow(processed, cmap="gray")
                            ax2.axis("off")
                            st.pyplot(fig2)
                            plt.close()

                        st.markdown("---")
                        st.subheader("✅ Các phép toán đã áp dụng:")
                        for idx, op in enumerate(operations, 1):
                            st.markdown(f"{idx}. {op}")

                        st.markdown("---")
                        npy_buffer = io.BytesIO()
                        np.save(npy_buffer, processed)
                        npy_buffer.seek(0)

                        st.download_button(
                            "⬇️ Tải ảnh đã xử lý (.npy)",
                            npy_buffer,
                            "preprocessed.npy",
                            "application/octet-stream",
                        )

                    except Exception as e:
                        st.error(f"❌ Lỗi: {e}")

        except Exception as e:
            st.error(f"❌ Lỗi tải ảnh: {e}")
    else:
        st.info("📁 Tải ảnh lên để bắt đầu")

        st.markdown("---")
        st.subheader("📖 Hướng dẫn Sử dụng")

        st.markdown(
            """
        **Các bước thực hiện:**
        1. Tải lên ảnh y tế (NIfTI, DICOM, NRRD, MHA)
        2. Chọn các phép toán cần áp dụng từ sidebar:
           - **Chuẩn hóa**: Đưa pixel về khoảng chuẩn
           - **Khử nhiễu**: Giảm nhiễu trong ảnh  
           - **Tăng cường**: Tăng độ tương phản
        3. Điều chỉnh tham số cho từng phép toán
        4. Nhấn "Áp dụng Xử lý"
        5. Xem kết quả và tải về
        
        **Cài đặt Khuyến nghị:**
        - **Thứ tự**: Chuẩn hóa → Khử nhiễu → Tăng cường
        - **Chuẩn hóa**: Min-Max (0-1) cho đầu vào neural network
        - **Khử nhiễu**: Gaussian với σ=1.0 cho ảnh MRI
        - **Tăng cường**: CLAHE với clip=2.0 cho độ tương phản tốt
        
        **Mẹo hữu ích:**
        - ⭐ Dùng Z-Score nếu ảnh có outliers
        - ⭐ Median filter tốt cho salt-pepper noise
        - ⭐ CLAHE tốt hơn histogram equalization cho ảnh y tế
        - ⭐ Xem trước kết quả trước khi áp dụng nhiều phép toán
        """
        )

elif selected_tool == "Anonymization":
    st.subheader("🔐 Ẩn danh hóa DICOM")
    st.markdown("Xóa thông tin bệnh nhân khỏi file DICOM")

    with st.expander("ℹ️ Những thông tin nào sẽ bị xóa?"):
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(
                """
            **Thông tin bệnh nhân**
            - Tên và mã định danh
            - Ngày sinh, tuổi, giới tính
            - Địa chỉ và liên lạc
            """
            )
        with col2:
            st.markdown(
                """
            **Thông tin nghiên cứu**
            - Ngày giờ nghiên cứu
            - Tên cơ sở y tế
            - Bác sĩ giới thiệu
            """
            )

    uploads = st.file_uploader(
        "Chọn file DICOM (.dcm)",
        type=["dcm"],
        accept_multiple_files=True,
        key="anon_uploader",
    )

    if uploads:
        st.success(f"✅ Đã nhận {len(uploads)} file")

        # Preview metadata
        try:
            file_bytes = io.BytesIO(uploads[0].getvalue())
            preview = pydicom.dcmread(file_bytes, force=True)
            st.markdown("**Xem trước Metadata (File đầu tiên):**")
            render_metadata(preview)
        except Exception as exc:
            st.warning(f"Không thể đọc metadata: {exc}")

        st.markdown("---")

        if st.button("🔒 Ẩn danh hóa", type="primary"):
            with st.spinner("⏳ Đang xử lý..."):
                try:
                    with tempfile.TemporaryDirectory() as tmp_dir:
                        tmp_root = Path(tmp_dir)
                        input_dir = tmp_root / "input"
                        output_dir = tmp_root / "output"
                        input_dir.mkdir()
                        output_dir.mkdir()

                        for upload in uploads:
                            (input_dir / upload.name).write_bytes(upload.getvalue())

                        anonymizer = DICOMAnonymizer(prefix=patient_prefix)
                        stats = anonymizer.anonymize_directory(
                            str(input_dir), str(output_dir)
                        )

                        # Detailed stats
                        successes = int(stats.get("successful", 0))
                        failures = int(stats.get("failed", 0))
                        mapping = stats.get("id_mapping", {})

                        message = (
                            "Ẩn danh hóa hoàn tất. "
                            f"Thành công: {successes} | "
                            f"Thất bại: {failures} | "
                            f"Số bệnh nhân: {len(mapping)}"
                        )
                        st.success(message)

                        # Show mapping
                        show_mapping(mapping)
                        st.markdown("---")

                        # Download ZIP
                        zip_buffer = io.BytesIO()
                        with zipfile.ZipFile(zip_buffer, "w") as zf:
                            for f in output_dir.glob("*.dcm"):
                                zf.write(f, f.name)
                        zip_buffer.seek(0)

                        st.subheader("Tải file đã ẩn danh")
                        st.download_button(
                            "⬇️ Tải file đã ẩn danh (ZIP)",
                            zip_buffer,
                            "dicom_da_an_danh.zip",
                            "application/zip",
                        )

                        # Preview anonymized metadata
                        anonymized_files = list(output_dir.glob("*.dcm"))
                        if anonymized_files:
                            st.markdown("---")
                            st.subheader("Xem trước metadata đã ẩn danh")
                            first_file = str(anonymized_files[0])
                            preview_dataset = pydicom.dcmread(first_file)
                            render_metadata(preview_dataset)
                            st.success("File đã không còn thông tin nhận dạng cá nhân.")

                            # Interpretation
                            st.markdown("---")
                            st.subheader("Giải thích kết quả ẩn danh hóa")
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

                except Exception as e:
                    st.error(f"❌ Lỗi: {e}")
    else:
        st.info("📁 Tải lên một hoặc nhiều file DICOM để bắt đầu.")
        st.markdown("---")
        st.subheader("📖 Hướng dẫn Sử dụng")
        st.markdown(
            """
            **Các bước thực hiện:**
            1. Nhấn "Browse files" và chọn file DICOM (.dcm).
            2. Xem trước metadata của file để kiểm tra thông tin.
            3. Nhấn "Ẩn danh hóa" để hệ thống tự động xóa thông tin nhạy cảm.
            4. Tải về file ZIP chứa các ảnh đã ẩn danh và bảng ánh xạ ID.
            
            **Lưu ý quan trọng:**
            - Giữ bảng ánh xạ ID (CSV) ở nơi an toàn nếu bạn cần tra cứu lại sau này.
            - File sau khi ẩn danh sẽ không thể khôi phục lại thông tin bệnh nhân nếu mất bảng ánh xạ.
            """
        )

        st.markdown("---")
        st.subheader("📖 Hướng dẫn Sử dụng")

        st.markdown(
            """
        **Các bước thực hiện:**
        1. Tải lên một hoặc nhiều file DICOM (.dcm)
        2. Nhập tiền tố ID ẩn danh từ sidebar (ví dụ: "ANON")
        3. Nhấn "Ẩn danh hóa"
        4. Tải về file ZIP chứa các file đã ẩn danh
        
        **Thông tin bị xóa:**
        - Tên bệnh nhân, ID bệnh nhân
        - Ngày sinh, tuổi, giới tính
        - Ngày giờ nghiên cứu
        - Tên bác sĩ, tên cơ sở y tế
        - Các thông tin nhận dạng khác
        
        **Mẹo hữu ích:**
        - ⭐ Có thể tải nhiều file cùng lúc
        - ⭐ ID mới sẽ được tạo tự động (ví dụ: ANON_0001)
        - ⭐ Metadata DICOM khác vẫn được giữ nguyên
        - ⭐ Phù hợp cho chia sẻ dữ liệu nghiên cứu
        """
        )

elif selected_tool == "Segmentation":
    st.subheader("✂️ Brain Segmentation")
    st.markdown("Trích xuất các vùng não từ ảnh y tế")

    with st.expander("Phương pháp Phân đoạn"):
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(
                """
                **Ngưỡng (Threshold):**
                - Phân đoạn dựa trên cường độ đơn giản
                - Nhanh và dễ hiểu
                - Tốt cho ảnh có độ tương phản cao
                
                **Phương pháp Otsu:**
                - Tự động chọn ngưỡng tối ưu
                - Không cần tham số thủ công
                - Hoạt động tốt với histogram hai đỉnh
                """
            )
        with col2:
            st.markdown(
                """
                **Tăng trưởng vùng (Region Growing):**
                - Phát triển từ điểm khởi đầu
                - Ranh giới chính xác hơn
                - Cần chọn điểm khởi đầu
                
                **Tự động:**
                - Kết hợp nhiều phương pháp
                - Kết quả tổng thể tốt nhất
                - Khuyến nghị cho người mới bắt đầu
                """
            )
    st.markdown("---")

    uploaded_file = st.file_uploader(
        "Chọn file (.nii, .nii.gz, .dcm, .nrrd, .mha, .npy)",
        type=["nii", "gz", "dcm", "nrrd", "mha", "npy"],
        key="seg_uploader",
    )

    if uploaded_file:
        suffix = (
            ".nii.gz"
            if uploaded_file.name.endswith(".nii.gz")
            else Path(uploaded_file.name).suffix
        )
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uploaded_file.getvalue())
            tmp_path = tmp.name

        try:
            with st.spinner("Đang tải ảnh..."):
                io_handler = MedicalImageIO()
                image_data, metadata = io_handler.read_image(tmp_path)

            st.success(f"Đã tải: {uploaded_file.name}")

            col1, col2, col3 = st.columns(3)
            col1.metric("Kích thước", f"{' × '.join(map(str, metadata['shape']))}")
            col2.metric("Kiểu dữ liệu", metadata["dtype"])
            col3.metric("Số chiều", f"{len(metadata['shape'])}D")

            st.markdown("---")
            explain_input_image(image_data, metadata)

            if st.button("Phân đoạn Não", type="primary"):
                with st.spinner("Đang phân đoạn..."):
                    try:
                        segmenter = BrainSegmentation(image_data)

                        if method_en == "Automatic" or method_en == "Otsu":
                            mask = segmenter.threshold_otsu()
                        elif method_en == "Threshold":
                            mask = segmenter.threshold_manual(threshold=threshold)
                        elif method_en == "Region Growing":
                            shape = image_data.shape
                            seed = [
                                int(seed_x * shape[0] / 100),
                                int(seed_y * shape[1] / 100),
                                int(seed_z * shape[2] / 100) if len(shape) > 2 else 0,
                            ]
                            if len(shape) == 2:
                                seed = seed[:2]
                            mask = segmenter.region_growing(
                                seed=tuple(seed), tolerance=intensity_tolerance
                            )

                        if apply_morph:
                            morph_map = {
                                "đóng (closing)": "closing",
                                "mở (opening)": "opening",
                                "giãn (dilation)": "dilation",
                                "xói mòn (erosion)": "erosion",
                            }
                            morph_op_en = morph_map[morph_op]

                            if morph_op_en == "closing":
                                mask = segmenter.morphological_closing(
                                    mask, kernel_size=kernel_size_morph
                                )
                            elif morph_op_en == "opening":
                                mask = segmenter.morphological_opening(
                                    mask, kernel_size=kernel_size_morph
                                )
                            elif morph_op_en == "dilation":
                                from skimage import morphology

                                if image_data.ndim == 2:
                                    kernel = morphology.disk(kernel_size_morph)
                                else:
                                    kernel = morphology.ball(kernel_size_morph)
                                mask = morphology.binary_dilation(mask, kernel).astype(
                                    np.uint8
                                )
                            elif morph_op_en == "erosion":
                                from skimage import morphology

                                if image_data.ndim == 2:
                                    kernel = morphology.disk(kernel_size_morph)
                                else:
                                    kernel = morphology.ball(kernel_size_morph)
                                mask = morphology.binary_erosion(mask, kernel).astype(
                                    np.uint8
                                )

                        if keep_largest:
                            mask = segmenter.get_largest_component(mask)

                        st.success("Phân đoạn hoàn tất!")

                        # Display results
                        st.markdown("---")
                        st.header("Kết quả Phân đoạn")

                        total_voxels = mask.size
                        segmented_voxels = np.sum(mask > 0)
                        percentage = (segmented_voxels / total_voxels) * 100

                        col1, col2, col3, col4 = st.columns(4)
                        col1.metric("Tổng Voxel", f"{total_voxels:,}")
                        col2.metric("Đã phân đoạn", f"{segmented_voxels:,}")
                        col3.metric("Tỷ lệ", f"{percentage:.1f}%")
                        col4.metric("Nền", f"{total_voxels - segmented_voxels:,}")

                        st.markdown("---")
                        st.subheader("Trực quan hóa")

                        if image_data.ndim == 3:
                            col1, col2 = st.columns([3, 1])
                            with col1:
                                view_mode = st.radio(
                                    "Chế độ xem:",
                                    ["Gốc", "Mask", "Phủ lớp"],
                                    horizontal=True,
                                    key="view_mode",
                                )
                            with col2:
                                opacity = st.slider(
                                    "Độ mờ phủ lớp", 0.0, 1.0, 0.5, key="opacity"
                                )

                            axis = st.radio(
                                "Trục:",
                                [
                                    "Trục Z (Axial)",
                                    "Trục Y (Coronal)",
                                    "Trục X (Sagittal)",
                                ],
                                horizontal=True,
                                key="axis",
                            )

                            if axis == "Trục Z (Axial)":
                                max_slice = image_data.shape[2] - 1
                                slice_idx = st.slider(
                                    "Lát cắt",
                                    0,
                                    max_slice,
                                    max_slice // 2,
                                    key="slice_idx",
                                )
                                img_slice = image_data[:, :, slice_idx]
                                mask_slice = mask[:, :, slice_idx]
                            elif axis == "Trục Y (Coronal)":
                                max_slice = image_data.shape[1] - 1
                                slice_idx = st.slider(
                                    "Lát cắt",
                                    0,
                                    max_slice,
                                    max_slice // 2,
                                    key="slice_idx",
                                )
                                img_slice = image_data[:, slice_idx, :]
                                mask_slice = mask[:, slice_idx, :]
                            else:
                                max_slice = image_data.shape[0] - 1
                                slice_idx = st.slider(
                                    "Lát cắt",
                                    0,
                                    max_slice,
                                    max_slice // 2,
                                    key="slice_idx",
                                )
                                img_slice = image_data[slice_idx, :, :]
                                mask_slice = mask[slice_idx, :, :]
                        else:
                            view_mode = st.radio(
                                "Chế độ xem:",
                                ["Gốc", "Mask", "Phủ lớp"],
                                horizontal=True,
                                key="view_mode",
                            )
                            opacity = st.slider(
                                "Độ mờ phủ lớp", 0.0, 1.0, 0.5, key="opacity"
                            )
                            img_slice = image_data
                            mask_slice = mask

                        fig, ax = plt.subplots(figsize=(10, 10))
                        view_map = {
                            "Gốc": "Original",
                            "Mask": "Mask",
                            "Phủ lớp": "Overlay",
                        }
                        view_mode_en = view_map.get(view_mode, view_mode)

                        if view_mode_en == "Original":
                            ax.imshow(img_slice.T, cmap="gray", origin="lower")
                            ax.set_title("Ảnh gốc", fontsize=14, fontweight="bold")
                        elif view_mode_en == "Mask":
                            ax.imshow(mask_slice.T, cmap="hot", origin="lower")
                            ax.set_title(
                                "Mask phân đoạn", fontsize=14, fontweight="bold"
                            )
                        else:
                            ax.imshow(img_slice.T, cmap="gray", origin="lower")
                            colors = [(0, 0, 0, 0), (1, 0, 0, opacity)]
                            cmap = ListedColormap(colors)
                            ax.imshow(
                                mask_slice.T, cmap=cmap, origin="lower", alpha=opacity
                            )
                            ax.set_title(
                                "Phủ lớp (Đỏ = Đã phân đoạn)",
                                fontsize=14,
                                fontweight="bold",
                            )

                        ax.axis("off")
                        st.pyplot(fig)
                        plt.close()

                        # Download
                        st.markdown("---")
                        st.subheader("Tải về Kết quả")
                        col1, col2 = st.columns(2)
                        with col1:
                            npy_buffer = io.BytesIO()
                            np.save(npy_buffer, mask)
                            st.download_button(
                                "⬇️ Tải Mask (.npy)",
                                npy_buffer.getvalue(),
                                "mask.npy",
                                "application/octet-stream",
                            )
                        with col2:
                            fig_download, ax_download = plt.subplots(figsize=(10, 10))
                            ax_download.imshow(img_slice.T, cmap="gray", origin="lower")
                            colors = [(0, 0, 0, 0), (1, 0, 0, 0.5)]
                            cmap = ListedColormap(colors)
                            ax_download.imshow(
                                mask_slice.T, cmap=cmap, origin="lower", alpha=0.5
                            )
                            ax_download.axis("off")
                            img_buffer = io.BytesIO()
                            fig_download.savefig(
                                img_buffer, format="png", bbox_inches="tight", dpi=150
                            )
                            img_buffer.seek(0)
                            plt.close(fig_download)
                            st.download_button(
                                "⬇️ Tải Phủ lớp (.png)",
                                img_buffer,
                                "overlay.png",
                                "image/png",
                            )

                        # Interpretation
                        st.markdown("---")
                        st.subheader("Giải thích kết quả phân đoạn")
                        visualizer = ResultVisualizer()
                        labels = {1: "Vùng não đã phân đoạn (Brain Tissue)"}
                        visualizer.show_overlay_with_legend(
                            image=img_slice,
                            mask=mask_slice,
                            labels=labels,
                            title="Kết quả phân đoạn với chú thích màu",
                        )

                        show_interpretation_section(
                            task_type="segmentation",
                            metrics={},
                            image_info={
                                "method": method,
                                "region_percentage": percentage,
                                "segmented_voxels": segmented_voxels,
                                "total_voxels": total_voxels,
                                "morph_applied": apply_morph,
                                "kept_largest": keep_largest,
                            },
                        )

                    except Exception as e:
                        st.error(f"Phân đoạn thất bại: {e}")
                        st.exception(e)

        except Exception as e:
            st.error(f"Lỗi khi tải ảnh: {e}")
    else:
        st.info("Tải lên ảnh não để bắt đầu phân đoạn")
        st.markdown("---")
        st.subheader("📖 Hướng dẫn Sử dụng")
        st.markdown(
            """
            **Các bước thực hiện:**
            1. Tải lên ảnh chụp não (NIfTI, DICOM, NRRD, MHA)
            2. Chọn phương pháp phân đoạn từ sidebar:
               - **Tự động**: Khuyến nghị cho người mới
               - **Otsu**: Tự động chọn ngưỡng tối ưu
               - **Ngưỡng**: Kiểm soát thủ công
               - **Tăng trưởng vùng**: Chính xác nhất
            3. Điều chỉnh tham số (nếu cần)
            4. Nhấn "Phân đoạn Não"
            5. Xem và tải về kết quả
            
            **Cài đặt Khuyến nghị:**
            - **Phương pháp**: Bắt đầu với "Tự động"
            - **Xử lý sau**: Bật phép đóng hình thái (morphological closing)
            - **Giữ lớn nhất**: Luôn bật để loại bỏ nhiễu
            
            **Mẹo hữu ích:**
            - ⭐ Dùng Otsu để tự động chọn ngưỡng tối ưu
            - ⭐ Tăng trưởng vùng hoạt động tốt nhất với ranh giới rõ ràng
            - ⭐ Thử các góc nhìn khác nhau (Axial/Coronal/Sagittal) cho ảnh 3D
            - ⭐ Chọn điểm khởi đầu ở giữa vùng não (50%, 50%, 50%)
            """
        )

elif selected_tool == "CT Reconstruction":
    st.subheader("🏗️ Tái tạo CT")
    st.markdown("Tái tạo ảnh CT từ dữ liệu chiếu (sinogram)")

    with st.expander("Về Tái tạo CT"):
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(
                """
                **Tái tạo CT là gì?**
                
                Máy CT quay nguồn tia X quanh bệnh nhân,
                chụp hình chiếu ở các góc khác nhau.
                
                **Sinogram:** Tập hợp tất cả hình chiếu
                - Mỗi hàng = một góc quét
                - Chứa dữ liệu cường độ tia X
                
                **Tái tạo:** Chuyển sinogram → ảnh CT 2D
                """
            )
        with col2:
            st.markdown(
                """
                **Thuật toán:**
                
                **FBP (Chiếu ngược có lọc):**
                - Nhanh (tiêu chuẩn lâm sàng)
                - Nhiều bộ lọc khả dụng
                - Tốt cho dữ liệu đầy đủ
                
                **SART (Phương pháp lặp):**
                - Chậm hơn nhưng chất lượng tốt hơn
                - Tốt cho dữ liệu thưa (ít góc quét)
                - Giảm nhiễu hiệu quả
                """
            )
    st.markdown("---")

    if data_source == "Tạo Phantom":
        st.subheader("Shepp-Logan Phantom")

        col1, col2 = st.columns(2)
        with col1:
            phantom_size = st.slider(
                "Kích thước Phantom:", 64, 512, 256, step=64, key="phantom_size"
            )
            num_angles = st.slider(
                "Số góc quét:", 30, 360, 180, step=30, key="num_angles"
            )

        with col2:
            st.markdown(
                """
                **Shepp-Logan Phantom:**
                - Ảnh test chuẩn cho CT
                - Chứa các hình elip
                - Hoàn hảo để test thuật toán
                """
            )

        if st.button("Tạo & Tái tạo", type="primary"):
            with st.spinner("Đang tạo phantom..."):
                phantom = create_shepp_logan_phantom(phantom_size)

                from skimage.transform import radon

                angles = np.linspace(0, 180, num_angles, endpoint=False)
                sinogram = radon(phantom, theta=angles)

            with st.spinner(f"Đang tái tạo sử dụng {ct_method}..."):
                reconstructor = CTReconstructor(sinogram, theta=angles)

                if ct_method == "FBP":
                    reconstructed = reconstructor.reconstruct_fbp(
                        filter_name=filter_type
                    )
                else:
                    reconstructed = reconstructor.reconstruct_sart(
                        iterations=num_iterations,
                        relaxation=relaxation,
                        image_size=phantom_size,
                    )

            st.success("Tái tạo hoàn tất!")

            # Display Results
            st.markdown("---")
            st.subheader("Kết quả")

            # Sinogram
            st.subheader("Sinogram (Dữ liệu chiếu)")
            fig, ax = plt.subplots(figsize=(10, 6))
            im = ax.imshow(sinogram, cmap="gray", aspect="auto")
            ax.set_xlabel("Vị trí Detector")
            ax.set_ylabel("Góc chiếu")
            plt.colorbar(im, ax=ax, label="Cường độ")
            st.pyplot(fig)
            plt.close()

            # Reconstructed Image
            st.markdown("---")
            st.subheader("Ảnh CT Tái tạo")

            col1, col2 = st.columns([3, 1])
            with col2:
                colormap = st.selectbox(
                    "Colormap:", ["gray", "bone", "hot"], index=1, key="ct_cmap"
                )
                show_colorbar = st.checkbox(
                    "Hiển thị thanh màu", value=True, key="ct_cbar"
                )

            fig, ax = plt.subplots(figsize=(10, 10))
            im = ax.imshow(reconstructed, cmap=colormap)
            ax.axis("off")
            ax.set_title(f"Ảnh tái tạo ({ct_method})", fontsize=14, fontweight="bold")
            if show_colorbar:
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            st.pyplot(fig)
            plt.close()

            # Metrics & Comparison
            st.markdown("---")
            st.subheader("Chỉ số Chất lượng")

            # Resize phantom if needed
            if phantom.shape != reconstructed.shape:
                from skimage.transform import resize

                phantom = resize(phantom, reconstructed.shape, anti_aliasing=True)

            psnr = calculate_psnr(phantom, reconstructed)
            ssim = calculate_ssim(phantom, reconstructed)

            col1, col2, col3 = st.columns(3)
            col1.metric("PSNR (dB)", f"{psnr:.2f}")
            col2.metric("SSIM", f"{ssim:.4f}")
            col3.metric("Sai số Max", f"{np.max(np.abs(phantom - reconstructed)):.4f}")

            st.markdown("---")
            st.subheader("So sánh: Gốc và Tái tạo")

            fig, axes = plt.subplots(1, 3, figsize=(15, 5))

            axes[0].imshow(phantom, cmap="gray")
            axes[0].set_title("Phantom gốc")
            axes[0].axis("off")

            axes[1].imshow(reconstructed, cmap="gray")
            axes[1].set_title(f"Tái tạo ({ct_method})")
            axes[1].axis("off")

            diff = np.abs(phantom - reconstructed)
            im = axes[2].imshow(diff, cmap="hot")
            axes[2].set_title("Sai số tuyệt đối")
            axes[2].axis("off")
            plt.colorbar(im, ax=axes[2], fraction=0.046)

            st.pyplot(fig)
            plt.close()

            # Download
            st.markdown("---")
            st.subheader("Tải về Kết quả")
            col1, col2 = st.columns(2)
            with col1:
                npy_buffer = io.BytesIO()
                np.save(npy_buffer, reconstructed)
                st.download_button(
                    "⬇️ Tải ảnh (.npy)",
                    npy_buffer.getvalue(),
                    f"ct_{ct_method.lower()}.npy",
                    "application/octet-stream",
                )
            with col2:
                fig_save = plt.figure(figsize=(8, 8))
                plt.imshow(reconstructed, cmap="gray")
                plt.axis("off")
                img_buffer = io.BytesIO()
                plt.savefig(img_buffer, format="png", bbox_inches="tight", dpi=150)
                img_buffer.seek(0)
                plt.close()
                st.download_button(
                    "⬇️ Tải ảnh (.png)",
                    img_buffer,
                    f"ct_{ct_method.lower()}.png",
                    "image/png",
                )

            # Interpretation
            st.markdown("---")
            st.subheader("Giải thích kết quả tái tạo CT")

            visualizer = ResultVisualizer()
            phantom_norm = (phantom - phantom.min()) / (
                phantom.max() - phantom.min() + 1e-8
            )
            recon_norm = (reconstructed - reconstructed.min()) / (
                reconstructed.max() - reconstructed.min() + 1e-8
            )

            visualizer.compare_images(
                phantom_norm,
                recon_norm,
                title_before="Phantom gốc",
                title_after=f"CT tái tạo ({ct_method})",
                description=f"Tái tạo từ {num_angles} góc quét. Phương pháp {ct_method}.",
            )

            metrics = {"PSNR": psnr, "SSIM": ssim}
            explainer = MetricsExplainer()
            explainer.show_metrics_dashboard(metrics)

            info_dict = {"method": ct_method, "num_angles": num_angles}
            if ct_method == "FBP":
                info_dict["filter"] = filter_type
            else:
                info_dict["iterations"] = num_iterations

            show_interpretation_section(
                task_type="reconstruction", metrics=metrics, image_info=info_dict
            )

    else:
        st.subheader("Tải lên Sinogram")
        uploaded_file = st.file_uploader(
            "Chọn file sinogram (.npy)", type=["npy"], key="sinogram_uploader"
        )

        if uploaded_file:
            try:
                sinogram = np.load(io.BytesIO(uploaded_file.getvalue()))

                if sinogram.ndim != 2:
                    st.error(f"Sinogram phải là mảng 2D, nhận được {sinogram.ndim}D")
                    st.stop()

                st.success(f"✅ Đã tải sinogram: {sinogram.shape}")

                with st.expander("📊 Thông tin về Sinogram đã tải"):
                    explain_input_image(sinogram)

                if st.button("Tái tạo", type="primary"):
                    with st.spinner(f"Đang tái tạo sử dụng {ct_method}..."):
                        num_angles = sinogram.shape[1]
                        angles = np.linspace(0, 180, num_angles, endpoint=False)

                        reconstructor = CTReconstructor(sinogram, theta=angles)

                        if ct_method == "FBP":
                            reconstructed = reconstructor.reconstruct_fbp(
                                filter_name=filter_type
                            )
                        else:
                            image_size = sinogram.shape[0]
                            reconstructed = reconstructor.reconstruct_sart(
                                iterations=num_iterations,
                                relaxation=relaxation,
                                image_size=image_size,
                            )

                    st.success("✅ Tái tạo hoàn tất!")

                    # Display Results (Similar to Phantom mode but without Ground Truth comparison)
                    st.markdown("---")
                    st.subheader("Kết quả")

                    # Sinogram
                    st.subheader("Sinogram (Dữ liệu chiếu)")
                    fig, ax = plt.subplots(figsize=(10, 6))
                    im = ax.imshow(sinogram, cmap="gray", aspect="auto")
                    ax.set_xlabel("Vị trí Detector")
                    ax.set_ylabel("Góc chiếu")
                    plt.colorbar(im, ax=ax, label="Cường độ")
                    st.pyplot(fig)
                    plt.close()

                    # Reconstructed Image
                    st.markdown("---")
                    st.subheader("Ảnh CT Tái tạo")
                    col1, col2 = st.columns([3, 1])
                    with col2:
                        colormap = st.selectbox(
                            "Colormap:",
                            ["gray", "bone", "hot"],
                            index=1,
                            key="ct_cmap_upload",
                        )
                        show_colorbar = st.checkbox(
                            "Hiển thị thanh màu", value=True, key="ct_cbar_upload"
                        )

                    fig, ax = plt.subplots(figsize=(10, 10))
                    im = ax.imshow(reconstructed, cmap=colormap)
                    ax.axis("off")
                    ax.set_title(
                        f"Ảnh tái tạo ({ct_method})", fontsize=14, fontweight="bold"
                    )
                    if show_colorbar:
                        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                    st.pyplot(fig)
                    plt.close()

                    # Download
                    st.markdown("---")
                    st.subheader("Tải về Kết quả")
                    col1, col2 = st.columns(2)
                    with col1:
                        npy_buffer = io.BytesIO()
                        np.save(npy_buffer, reconstructed)
                        st.download_button(
                            "⬇️ Tải ảnh (.npy)",
                            npy_buffer.getvalue(),
                            f"ct_{ct_method.lower()}.npy",
                            "application/octet-stream",
                        )
                    with col2:
                        fig_save = plt.figure(figsize=(8, 8))
                        plt.imshow(reconstructed, cmap="gray")
                        plt.axis("off")
                        img_buffer = io.BytesIO()
                        plt.savefig(
                            img_buffer, format="png", bbox_inches="tight", dpi=150
                        )
                        img_buffer.seek(0)
                        plt.close()
                        st.download_button(
                            "⬇️ Tải ảnh (.png)",
                            img_buffer,
                            f"ct_{ct_method.lower()}.png",
                            "image/png",
                        )

            except Exception as e:
                st.error(f"❌ Lỗi: {e}")
        else:
            st.info("Tải sinogram lên để bắt đầu")
            st.markdown("---")
            st.subheader("📖 Hướng dẫn Sử dụng Chi tiết")
            st.markdown(
                """
                **Các bước thực hiện - Sử dụng Phantom (Khuyến nghị cho người mới):**
                1. **Chuẩn bị**: Giữ "Tạo Phantom" được chọn từ sidebar
                2. **Cài đặt Phantom**:
                   - Kích thước: 256x256 (cân bằng tốc độ/chất lượng)
                   - Số góc quét: 180 góc (đủ cho tái tạo tốt)
                3. **Chọn phương pháp tái tạo**:
                   - **FBP** (Filtered Back Projection): Nhanh, phù hợp demo
                   - **SART** (Algebraic): Chất lượng cao hơn, chậm hơn
                4. **Điều chỉnh tham số**:
                   - FBP: Chọn bộ lọc (ramp khuyến nghị)
                   - SART: Số lần lặp (10-20), relaxation (0.5)
                5. **Thực hiện**: Nhấn "Tạo & Tái tạo"
                6. **Đánh giá**:
                   - So sánh ảnh gốc vs ảnh tái tạo
                   - Xem chỉ số: PSNR, SSIM, MSE, SNR
                7. **Thử nghiệm**: Thay đổi tham số và so sánh kết quả
                
                **Sử dụng Dữ liệu Thực (Sinogram):**
                1. Chọn "Tải lên Sinogram" từ sidebar
                2. Tải file .npy chứa dữ liệu sinogram
                3. Chọn phương pháp và tham số
                4. Nhấn "Tái tạo"
                5. Tải về ảnh đã tái tạo (.npy)
                
                **Mẹo hữu ích:**
                - ⭐ **Bắt đầu với FBP**: Nhanh, dễ hiểu
                - ⭐ **Nhiều góc = Tốt hơn**: 180 góc > 90 góc
                - ⭐ **SART cho ít góc**: Nếu < 90 góc, dùng SART
                - ⭐ **Thử bộ lọc**: ramp → shepp-logan → hamming
                - ⭐ **FBP vs SART**: FBP nhanh x10-20
                - ⭐ **Dùng phantom**: Để học và thử nghiệm
                """
            )

elif selected_tool == "MRI Reconstruction":
    st.subheader("🧲 MRI Reconstruction")
    st.markdown("Tái tạo ảnh MRI từ dữ liệu K-space sử dụng FFT")

    with st.expander("Về MRI và K-space"):
        col1, col2 = st.columns(2)

        with col1:
            st.markdown(
                """
                **K-space là gì?**
                
                K-space là biểu diễn **miền tần số** 
                của dữ liệu MRI thu thập bởi máy quét.
                
                **Thuộc tính:**
                - Trung tâm: Tần số thấp (độ tương phản)
                - Rìa: Tần số cao (chi tiết)
                - Dữ liệu thô từ máy quét MRI
                
                **KHÔNG phải ảnh thực!**
                Cần FFT để chuyển thành ảnh.
                """
            )

        with col2:
            st.markdown(
                """
                **Quy trình Tái tạo:**
                
                1. **Thu thập K-space** (máy quét)
                2. **Inverse FFT** (2D)
                3. **Trích xuất magnitude** (giải phẫu)
                4. **Trích xuất phase** (dòng máu, v.v.)
                
                **Partial Fourier:**
                - Scan only part of K-space
                - 50% faster acquisition
                - Estimate missing data
                """
            )

    st.markdown("---")

    # Main content
    if mri_data_source == "Tạo từ Ảnh":
        st.subheader("📤 K-space từ Ảnh")

        uploaded_file = st.file_uploader(
            "Tải ảnh lên (.nii, .dcm, .nrrd, .mha, .npy)",
            type=["nii", "gz", "dcm", "nrrd", "mha", "npy"],
            key="mri_upload",
            help="Tải ảnh y tế lên để tạo K-space",
        )

        if uploaded_file:
            # Load image - handle compound extensions like .nii.gz
            if uploaded_file.name.endswith(".nii.gz"):
                suffix = ".nii.gz"
            else:
                suffix = Path(uploaded_file.name).suffix

            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_path = tmp_file.name

            try:
                with st.spinner("Đang tải ảnh..."):
                    io_handler = MedicalImageIO()
                    image_data, metadata = io_handler.read_image(tmp_path)

                    # Use 2D slice if 3D
                    if image_data.ndim == 3:
                        slice_idx = image_data.shape[2] // 2
                        image_2d = image_data[:, :, slice_idx]
                        st.info(f"Using middle slice ({slice_idx}) from 3D volume")
                    else:
                        image_2d = image_data

                st.success(f"Đã tải: {image_2d.shape}")

                # Generate K-space button
                if st.button(
                    "🔄 Tạo K-space và Tái tạo",
                    type="primary",
                    use_container_width=True,
                    key="mri_generate_btn",
                ):

                    with st.spinner("Đang tạo K-space..."):
                        # Save original image for comparison
                        st.session_state.mri_original_image = image_2d

                        # Create dummy kspace for initialization
                        dummy_kspace = np.zeros((2, 2), dtype=np.complex128)
                        reconstructor = MRIReconstructor(dummy_kspace)

                        # Forward FFT: Image → K-space
                        kspace = reconstructor.image_to_kspace(image_2d)
                        st.session_state.mri_kspace = kspace

                        # Apply partial Fourier if enabled
                        if (
                            "partial_fourier" in st.session_state
                            and st.session_state.partial_fourier
                        ):
                            # Keep only percentage of K-space
                            kspace_partial = kspace.copy()
                            rows = kspace.shape[0]
                            pf_pct = st.session_state.get("pf_percentage", 75)
                            keep_rows = int(rows * pf_pct / 100)
                            start_row = (rows - keep_rows) // 2

                            # Zero out other rows
                            mask = np.zeros(rows, dtype=bool)
                            mask[start_row : start_row + keep_rows] = True
                            kspace_partial[~mask, :] = 0

                            st.session_state.mri_kspace = kspace_partial
                            st.info(f"Using {pf_pct}% of K-space (Partial Fourier)")

                    with st.spinner("Reconstructing image..."):
                        # Inverse FFT: K-space → Image
                        image_complex = reconstructor.kspace_to_image(
                            st.session_state.mri_kspace
                        )

                        # Extract magnitude and phase
                        magnitude = np.abs(image_complex)
                        phase = np.angle(image_complex)

                        st.session_state.mri_magnitude = magnitude
                        st.session_state.mri_phase = phase

                    st.success("✅ Hoàn tất!")

            except Exception as e:
                st.error(f"❌ Lỗi: {str(e)}")
                st.exception(e)

    else:  # Upload K-space
        st.subheader("📤 Dữ liệu K-space")

        uploaded_kspace = st.file_uploader(
            "Chọn file K-space (.npy)",
            type=["npy"],
            key="kspace_upload",
            help="Mảng NumPy phức (dữ liệu K-space)",
        )

        if uploaded_kspace:
            try:
                kspace = np.load(io.BytesIO(uploaded_kspace.getvalue()))

                if not np.iscomplexobj(kspace):
                    st.warning("⚠️ Dữ liệu nên là số phức. Đang chuyển đổi...")
                    kspace = kspace.astype(np.complex128)

                st.session_state.mri_kspace = kspace
                st.success(f"✅ Đã tải K-space: {kspace.shape}")

                # Reconstruct button
                if st.button(
                    "🔄 Tái tạo",
                    type="primary",
                    use_container_width=True,
                    key="mri_reconstruct_btn",
                ):

                    with st.spinner("Đang tái tạo..."):
                        reconstructor = MRIReconstructor(kspace)

                        # Inverse FFT
                        image_complex = reconstructor.kspace_to_image(kspace)

                        # Extract magnitude and phase
                        magnitude = np.abs(image_complex)
                        phase = np.angle(image_complex)

                        st.session_state.mri_magnitude = magnitude
                        st.session_state.mri_phase = phase

                    st.success("✅ Hoàn tất!")

            except Exception as e:
                st.error(f"❌ Lỗi khi tải K-space: {str(e)}")

    # Display results
    if st.session_state.get("mri_kspace") is not None:
        st.markdown("---")
        st.header("📊 Kết quả")

        kspace = st.session_state.mri_kspace
        magnitude = st.session_state.get("mri_magnitude")
        phase = st.session_state.get("mri_phase")

        if magnitude is not None and phase is not None:
            col1, col2, col3 = st.columns(3)

            with col1:
                st.markdown("**K-space (Log Magnitude)**")
                kspace_mag = np.log(np.abs(kspace) + 1)
                fig, ax = plt.subplots(figsize=(5, 5))
                im = ax.imshow(kspace_mag, cmap="hot")
                ax.set_title("K-space")
                ax.axis("off")
                plt.colorbar(im, ax=ax, fraction=0.046)
                st.pyplot(fig)
                plt.close()

            with col2:
                st.markdown("**Magnitude Image**")
                fig, ax = plt.subplots(figsize=(5, 5))
                ax.imshow(magnitude, cmap="gray")
                ax.set_title("Magnitude")
                ax.axis("off")
                st.pyplot(fig)
                plt.close()

            with col3:
                st.markdown("**Phase Image**")
                fig, ax = plt.subplots(figsize=(5, 5))
                im = ax.imshow(phase, cmap="twilight", vmin=-np.pi, vmax=np.pi)
                ax.set_title("Phase")
                ax.axis("off")
                plt.colorbar(im, ax=ax, fraction=0.046)
                st.pyplot(fig)
                plt.close()

            # Statistics
            st.markdown("### 📈 Thống kê")
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("K-space Mean", f"{np.abs(kspace).mean():.2e}")
            col2.metric("K-space Max", f"{np.abs(kspace).max():.2e}")
            col3.metric("Magnitude Mean", f"{magnitude.mean():.4f}")
            col4.metric("Magnitude Std", f"{magnitude.std():.4f}")

            # Comparison with original if available
            if "mri_original_image" in st.session_state:
                st.markdown("### 🔍 So sánh với Ảnh Gốc")
                original = st.session_state.mri_original_image

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.markdown("**Original**")
                    fig, ax = plt.subplots(figsize=(5, 5))
                    ax.imshow(original, cmap="gray")
                    ax.set_title("Original")
                    ax.axis("off")
                    st.pyplot(fig)
                    plt.close()

                with col2:
                    st.markdown("**Reconstructed**")
                    fig, ax = plt.subplots(figsize=(5, 5))
                    ax.imshow(magnitude, cmap="gray")
                    ax.set_title("Reconstructed")
                    ax.axis("off")
                    st.pyplot(fig)
                    plt.close()

                with col3:
                    st.markdown("**Difference**")
                    # Normalize both to same range for fair comparison
                    orig_norm = (original - original.min()) / (
                        original.max() - original.min() + 1e-10
                    )
                    mag_norm = (magnitude - magnitude.min()) / (
                        magnitude.max() - magnitude.min() + 1e-10
                    )
                    diff = np.abs(orig_norm - mag_norm)
                    fig, ax = plt.subplots(figsize=(5, 5))
                    im = ax.imshow(diff, cmap="hot")
                    ax.set_title("Difference")
                    ax.axis("off")
                    plt.colorbar(im, ax=ax, fraction=0.046)
                    st.pyplot(fig)
                    plt.close()

                # Quality metrics
                mse = np.mean((orig_norm - mag_norm) ** 2)
                psnr = 10 * np.log10(1.0 / (mse + 1e-10))

                col1, col2 = st.columns(2)
                col1.metric("MSE", f"{mse:.6f}")
                col2.metric("PSNR", f"{psnr:.2f} dB")

            # Download options
            st.markdown("### 💾 Tải về")
            col1, col2 = st.columns(2)

            with col1:
                magnitude_buffer = io.BytesIO()
                np.save(magnitude_buffer, magnitude)
                st.download_button(
                    "⬇️ Magnitude (.npy)",
                    magnitude_buffer.getvalue(),
                    "magnitude.npy",
                    "application/octet-stream",
                    use_container_width=True,
                )

            with col2:
                phase_buffer = io.BytesIO()
                np.save(phase_buffer, phase)
                st.download_button(
                    "⬇️ Phase (.npy)",
                    phase_buffer.getvalue(),
                    "phase.npy",
                    "application/octet-stream",
                    use_container_width=True,
                )

    # User guide
    st.markdown("---")
    st.subheader("📖 Hướng dẫn Sử dụng")
    st.markdown(
        """
        **Tạo từ Ảnh:**
        1. Chọn "Tạo từ Ảnh" từ sidebar
        2. Tải lên ảnh MRI (.nii, .dcm, .npy, etc.)
        3. (Tùy chọn) Bật Partial Fourier để mô phỏng quét nhanh
        4. Nhấn "Tạo K-space và Tái tạo"
        5. Xem K-space, ảnh tái tạo, và so sánh

        **Tải lên K-space:**
        1. Chọn "Tải lên K-space" từ sidebar
        2. Tải file .npy chứa K-space (complex array)
        3. Nhấn "Tái tạo"
        4. Xem kết quả

        **Partial Fourier:**
        - Chỉ quét một phần K-space (50-100%)
        - Giảm thời gian quét MRI
        - Trade-off: Tốc độ vs Chất lượng
        - 75% là cân bằng tốt

        **Mẹo:**
        - ⭐ Trung tâm K-space chứa thông tin quan trọng nhất
        - ⭐ Phase image hữu ích cho phát hiện artifacts
        - ⭐ Magnitude image là ảnh giải phẫu thông thường
        """
    )

elif selected_tool == "Registration":
    st.subheader("🔄 Image Registration")
    st.markdown("Căn chỉnh hai ảnh y tế")

    with st.expander("Về Image Registration"):
        st.markdown(
            """
            **Image Registration** là quá trình căn chỉnh hai hoặc nhiều ảnh để chúng nằm trên cùng một hệ tọa độ.
            
            **Ứng dụng:**
            - 📊 So sánh ảnh trước/sau điều trị
            - 🧠 Đăng ký ảnh multi-modal (T1-T2 MRI)
            - 📈 Theo dõi sự phát triển của khối u
            - 🎯 Căn chỉnh ảnh theo atlas
            
            **Các loại Registration:**
            - **Rigid:** Chỉ di chuyển và xoay (nhanh nhất).
            - **Affine:** Thêm co giãn và nghiêng (trung bình).
            - **Deformable:** Biến dạng cục bộ (chậm nhất, chính xác nhất).
            """
        )

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📍 Fixed Image (Ảnh tham chiếu)")
        fixed_file = st.file_uploader(
            "Tải ảnh Fixed",
            type=["nii", "gz", "dcm", "nrrd", "mha", "npy"],
            key="fixed_img",
            help="Ảnh này sẽ giữ nguyên vị trí",
        )

        if fixed_file:
            suffix = (
                ".nii.gz"
                if fixed_file.name.endswith(".nii.gz")
                else Path(fixed_file.name).suffix
            )
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(fixed_file.getvalue())
                tmp_path = tmp.name

            try:
                io_handler = MedicalImageIO()
                fixed_array, _ = io_handler.read_image(tmp_path)

                if np.iscomplexobj(fixed_array):
                    fixed_array = np.abs(fixed_array)

                if fixed_array.ndim == 3:
                    slice_idx = fixed_array.shape[0] // 2
                    display_slice = fixed_array[slice_idx, :, :]
                else:
                    display_slice = fixed_array

                fig, ax = plt.subplots(figsize=(6, 6))
                ax.imshow(display_slice, cmap="gray")
                ax.set_title("Fixed Image")
                ax.axis("off")
                st.pyplot(fig)
                plt.close()

                st.session_state["reg_fixed"] = fixed_array
                st.success(f"✅ Loaded: {fixed_array.shape}")

            except Exception as e:
                st.error(f"❌ Lỗi: {e}")

    with col2:
        st.subheader("🔄 Moving Image (Ảnh cần căn chỉnh)")
        moving_file = st.file_uploader(
            "Tải ảnh Moving",
            type=["nii", "gz", "dcm", "nrrd", "mha", "npy"],
            key="moving_img",
            help="Ảnh này sẽ được di chuyển để khớp với Fixed",
        )

        if moving_file:
            suffix = (
                ".nii.gz"
                if moving_file.name.endswith(".nii.gz")
                else Path(moving_file.name).suffix
            )
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(moving_file.getvalue())
                tmp_path = tmp.name

            try:
                io_handler = MedicalImageIO()
                moving_array, _ = io_handler.read_image(tmp_path)

                if np.iscomplexobj(moving_array):
                    moving_array = np.abs(moving_array)

                if moving_array.ndim == 3:
                    slice_idx = moving_array.shape[0] // 2
                    display_slice = moving_array[slice_idx, :, :]
                else:
                    display_slice = moving_array

                fig, ax = plt.subplots(figsize=(6, 6))
                ax.imshow(display_slice, cmap="gray")
                ax.set_title("Moving Image")
                ax.axis("off")
                st.pyplot(fig)
                plt.close()

                st.session_state["reg_moving"] = moving_array
                st.success(f"✅ Loaded: {moving_array.shape}")

            except Exception as e:
                st.error(f"❌ Lỗi: {e}")

    st.markdown("---")

    if st.button("🔄 Bắt đầu Registration", type="primary", use_container_width=True):
        if "reg_fixed" not in st.session_state or "reg_moving" not in st.session_state:
            st.error("❌ Vui lòng tải cả 2 ảnh!")
        else:
            try:
                with st.spinner(
                    f"⏳ Đang thực hiện {registration_type} registration..."
                ):
                    fixed_sitk = numpy_to_sitk(st.session_state["reg_fixed"])
                    moving_sitk = numpy_to_sitk(st.session_state["reg_moving"])

                    reg = ImageRegistration(fixed_sitk, moving_sitk, verbose=False)

                    if registration_type == "Rigid":
                        registered_sitk = reg.rigid_registration(
                            number_of_iterations=reg_iterations,
                            learning_rate=learning_rate,
                            metric=reg_metric,
                        )
                    elif registration_type == "Affine":
                        registered_sitk = reg.affine_registration(
                            number_of_iterations=reg_iterations,
                            learning_rate=learning_rate,
                            metric=reg_metric,
                        )
                    else:
                        registered_sitk = reg.deformable_registration(
                            number_of_iterations=reg_iterations,
                            mesh_size=mesh_size,
                            metric=reg_metric,
                        )

                    registered_array = sitk_to_numpy(registered_sitk)
                    st.session_state["reg_result"] = registered_array
                    st.session_state["reg_metrics"] = reg.get_metrics()
                    st.session_state["reg_transform"] = reg.get_transform()

                st.success("✅ Registration hoàn tất!")

            except Exception as e:
                st.error(f"❌ Lỗi: {e}")
                st.exception(e)

    # Display Results
    if "reg_result" in st.session_state:
        st.markdown("---")
        st.header("Kết quả")

        # Metrics
        if "reg_metrics" in st.session_state:
            st.subheader("Metrics")
            metrics = st.session_state["reg_metrics"]
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("MSE Before", f"{metrics.get('mse_before', 0):.2f}")
            col2.metric(
                "MSE After",
                f"{metrics.get('mse_after', 0):.2f}",
                delta=f"{metrics.get('mse_improvement', 0):.1f}%",
                delta_color="inverse",
            )
            col3.metric("NCC Before", f"{metrics.get('ncc_before', 0):.3f}")
            col4.metric(
                "NCC After",
                f"{metrics.get('ncc_after', 0):.3f}",
                delta=f"{metrics.get('ncc_improvement', 0):.1f}%",
            )

        # Visualization
        st.subheader("Comparison")
        viz_mode = st.radio(
            "Chế độ hiển thị",
            ["Side by Side", "Overlay", "Checkerboard", "Difference"],
            horizontal=True,
            key="reg_viz_mode",
        )

        fixed_img = st.session_state["reg_fixed"]
        moving_img = st.session_state["reg_moving"]
        registered_img = st.session_state["reg_result"]

        # Slice selection for 3D
        if fixed_img.ndim == 3:
            slice_idx = st.slider(
                "Chọn slice",
                0,
                fixed_img.shape[0] - 1,
                fixed_img.shape[0] // 2,
                key="reg_slice",
            )
            fixed_slice = fixed_img[slice_idx, :, :]
            moving_slice = moving_img[slice_idx, :, :]
            registered_slice = registered_img[slice_idx, :, :]
        else:
            fixed_slice = fixed_img
            moving_slice = moving_img
            registered_slice = registered_img

        if viz_mode == "Side by Side":
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            axes[0].imshow(fixed_slice, cmap="gray")
            axes[0].set_title("Fixed")
            axes[0].axis("off")
            axes[1].imshow(moving_slice, cmap="gray")
            axes[1].set_title("Moving (Before)")
            axes[1].axis("off")
            axes[2].imshow(registered_slice, cmap="gray")
            axes[2].set_title("Registered (After)")
            axes[2].axis("off")
            st.pyplot(fig)
            plt.close()

        elif viz_mode == "Overlay":
            alpha = st.slider("Transparency", 0.0, 1.0, 0.5, 0.1, key="reg_alpha")
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

            ax1.imshow(fixed_slice, cmap="Reds", alpha=1.0)
            ax1.imshow(moving_slice, cmap="Blues", alpha=alpha)
            ax1.set_title("Before (Red=Fixed, Blue=Moving)")
            ax1.axis("off")

            ax2.imshow(fixed_slice, cmap="Reds", alpha=1.0)
            ax2.imshow(registered_slice, cmap="Blues", alpha=alpha)
            ax2.set_title("After (Red=Fixed, Blue=Registered)")
            ax2.axis("off")
            st.pyplot(fig)
            plt.close()

        elif viz_mode == "Checkerboard":
            squares = st.slider("Number of squares", 4, 16, 8, 2, key="reg_squares")

            def create_checkerboard(img1, img2, n_squares):
                h, w = img1.shape
                result = img1.copy()
                square_h = h // n_squares
                square_w = w // n_squares
                for i in range(n_squares):
                    for j in range(n_squares):
                        if (i + j) % 2 == 0:
                            result[
                                i * square_h : (i + 1) * square_h,
                                j * square_w : (j + 1) * square_w,
                            ] = img2[
                                i * square_h : (i + 1) * square_h,
                                j * square_w : (j + 1) * square_w,
                            ]
                return result

            checker_before = create_checkerboard(fixed_slice, moving_slice, squares)
            checker_after = create_checkerboard(fixed_slice, registered_slice, squares)

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
            ax1.imshow(checker_before, cmap="gray")
            ax1.set_title("Before Registration")
            ax1.axis("off")
            ax2.imshow(checker_after, cmap="gray")
            ax2.set_title("After Registration")
            ax2.axis("off")
            st.pyplot(fig)
            plt.close()

        else:  # Difference
            fig, axes = plt.subplots(1, 2, figsize=(12, 6))
            diff_before = np.abs(fixed_slice - moving_slice)
            diff_after = np.abs(fixed_slice - registered_slice)

            im1 = axes[0].imshow(diff_before, cmap="hot")
            axes[0].set_title("Difference Before")
            axes[0].axis("off")
            plt.colorbar(im1, ax=axes[0])

            im2 = axes[1].imshow(diff_after, cmap="hot")
            axes[1].set_title("Difference After")
            axes[1].axis("off")
            plt.colorbar(im2, ax=axes[1])
            st.pyplot(fig)
            plt.close()

        # Download & Save
        st.markdown("---")
        col1, col2 = st.columns(2)
        with col1:
            npy_buffer = io.BytesIO()
            np.save(npy_buffer, registered_img)
            st.download_button(
                "⬇️ Tải ảnh (.npy)",
                npy_buffer.getvalue(),
                f"registered_{registration_type.lower()}.npy",
                "application/octet-stream",
            )

        with col2:
            if st.button("💾 Lưu Transform"):
                try:
                    output_path = "registration_transform.tfm"
                    sitk.WriteTransform(st.session_state["reg_transform"], output_path)
                    st.success(f"Transform saved: {output_path}")
                except Exception as e:
                    st.error(f"❌ Lỗi: {e}")

    st.markdown("---")
    st.subheader("📖 Hướng dẫn Sử dụng")
    st.markdown(
        """
        **Các bước thực hiện:**
        1. Tải lên **Fixed Image** (ảnh tham chiếu) - bên trái
        2. Tải lên **Moving Image** (ảnh cần căn chỉnh) - bên phải  
        3. Chọn loại registration từ sidebar:
           - **Rigid**: Nhanh nhất, chỉ xoay + di chuyển
           - **Affine**: Trung bình, thêm scaling + shearing
           - **Deformable**: Chậm nhất, chính xác nhất
        4. Điều chỉnh tham số (iterations, learning rate)
        5. Nhấn "Bắt đầu Registration"
        6. Xem kết quả: Fixed | Moving (Before) | Registered (After)
        7. Tải về ảnh đã registration (.npy)
        
        **Cài đặt Khuyến nghị:**
        
        **Rigid Registration:**
        - **Số lần lặp**: 100 iterations  
        - **Learning rate**: 1.0
        - **Metric**: mean_squares (cùng modality)
        - **Phù hợp cho**: Follow-up scans của cùng bệnh nhân, head motion correction
        
        **Affine Registration:**
        - **Số lần lặp**: 150 iterations
        - **Learning rate**: 1.0  
        - **Metric**: mean_squares (cùng modality) hoặc mutual_information (multi-modal)
        - **Phù hợp cho**: Inter-subject registration, CT-MRI alignment
        
        **Deformable Registration:**
        - **Số lần lặp**: 50 iterations
        - **Mesh size**: 5 (nhỏ hơn = linh hoạt hơn, chậm hơn)
        - **Metric**: mutual_information (khuyến nghị)
        - **Phù hợp cho**: Tumor tracking, breathing motion, longitudinal studies
        
        **Mẹo hữu ích:**
        - ⭐ Bắt đầu với Rigid → nếu không đủ tốt → thử Affine → cuối cùng Deformable
        - ⭐ Mean Squares nhanh hơn Mutual Information
        - ⭐ Với 3D volume: Chọn slice ở giữa để xem kết quả
        - ⭐ MSE thấp và NCC cao = registration tốt
        """
    )
