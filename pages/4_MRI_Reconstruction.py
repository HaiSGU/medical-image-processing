"""
Trang Tái tạo MRI

Tái tạo ảnh MRI từ dữ liệu K-space sử dụng FFT.

Tác giả: HaiSGU
Ngày: 2025-10-28
"""

import streamlit as st
from pathlib import Path
import sys
import numpy as np
import matplotlib.pyplot as plt
import io

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import from src/ modules
from src.reconstruction.mri_reconstruction import MRIReconstructor
from utils.file_io import MedicalImageIO
from utils.interpretation import (
    ResultVisualizer,
    MetricsExplainer,
    show_interpretation_section,
)

# Page config
st.set_page_config(page_title="Tái tạo MRI", layout="wide")

# Initialize session state
if "mri_kspace" not in st.session_state:
    st.session_state.mri_kspace = None
if "mri_magnitude" not in st.session_state:
    st.session_state.mri_magnitude = None
if "mri_phase" not in st.session_state:
    st.session_state.mri_phase = None

# Header
st.title("Tái tạo MRI")
st.markdown("Tái tạo ảnh MRI từ dữ liệu K-space sử dụng FFT")

# Info
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

# Sidebar controls
with st.sidebar:
    st.header("Cài đặt")

    # Data source
    data_source = st.radio(
        "Nguồn dữ liệu:",
        ["Tạo từ Ảnh", "Tải lên K-space"],
        help="Tạo K-space từ ảnh hoặc tải dữ liệu thực",
    )

    st.markdown("---")

    # Reconstruction options
    if data_source == "Tạo từ Ảnh":
        partial_fourier = st.checkbox(
            "Partial Fourier", value=False, help="Mô phỏng quét nhanh hơn"
        )

        if partial_fourier:
            pf_percentage = st.select_slider(
                "Phủ K-space:",
                options=[50, 62.5, 75, 87.5, 100],
                value=75,
                help="Percentage of K-space to use",
            )

    st.markdown("---")
    st.info("of K-space contains most important information")

# Main content
if data_source == "Tạo từ Ảnh":
    st.subheader("K-space từ Ảnh")

    uploaded_file = st.file_uploader(
        "Tải ảnh lên (.nii, .dcm, .nrrd, .mha, .npy)",
        type=["nii", "gz", "dcm", "nrrd", "mha", "npy"],
        help="Tải ảnh y tế lên để tạo K-space",
    )

    if uploaded_file:
        # Load image - handle compound extensions like .nii.gz
        import tempfile

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
                "Tạo K-space và Tái tạo",
                type="primary",
                use_container_width=True,
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
                    if partial_fourier:
                        # Keep only percentage of K-space
                        kspace_partial = kspace.copy()
                        rows = kspace.shape[0]
                        keep_rows = int(rows * pf_percentage / 100)
                        start_row = (rows - keep_rows) // 2

                        # Zero out other rows
                        mask = np.zeros(rows, dtype=bool)
                        mask[start_row : start_row + keep_rows] = True
                        kspace_partial[~mask, :] = 0

                        st.session_state.mri_kspace = kspace_partial
                        st.info(f"Using {pf_percentage}% of K-space (Partial Fourier)")

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

                st.success("complete!")

        except Exception as e:
            st.error(f" Error: {str(e)}")
            st.exception(e)

else:  # Upload K-space
    st.subheader("Dữ liệu K-space")

    uploaded_kspace = st.file_uploader(
        "Chọn file K-space (.npy)",
        type=["npy"],
        help="Mảng NumPy phức (dữ liệu K-space)",
    )

    if uploaded_kspace:
        try:
            kspace = np.load(io.BytesIO(uploaded_kspace.getvalue()))

            if not np.iscomplexobj(kspace):
                st.warning("Dữ liệu nên là số phức. Đang chuyển đổi...")
                kspace = kspace.astype(np.complex128)

            st.session_state.mri_kspace = kspace
            st.success(f"Đã tải K-space: {kspace.shape}")

            # Reconstruct button
            if st.button("Tái tạo", type="primary", use_container_width=True):

                with st.spinner("Đang tái tạo..."):
                    reconstructor = MRIReconstructor(kspace)

                    # Inverse FFT
                    image_complex = reconstructor.kspace_to_image(kspace)

                    # Extract magnitude and phase
                    magnitude = np.abs(image_complex)
                    phase = np.angle(image_complex)

                    st.session_state.mri_magnitude = magnitude
                    st.session_state.mri_phase = phase

                st.success("Hoàn tất!")

        except Exception as e:
            st.error(f"Lỗi khi tải K-space: {str(e)}")

# Display results
if st.session_state.mri_kspace is not None:
    st.markdown("---")
    st.header("Kết quả")

    kspace = st.session_state.mri_kspace

    # Show K-space
    st.subheader("K-space (Miền tần số)")

    # Log magnitude for better visualization
    kspace_log = np.log(np.abs(kspace) + 1)

    fig, ax = plt.subplots(figsize=(10, 10))
    im = ax.imshow(kspace_log, cmap="gray")
    ax.set_title("K-space (Log Magnitude)", fontsize=14, fontweight="bold")
    ax.axis("off")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    st.pyplot(fig)
    plt.close()

    st.caption(f"Shape: {kspace.shape} | Data type: {kspace.dtype}")
    st.caption(
        "Bright center = low frequencies (contrast), Dark edges = high frequencies (details)"
    )

    # Show reconstructed images
    if st.session_state.mri_magnitude is not None:
        st.markdown("---")
        st.subheader("Hình ảnh")

        magnitude = st.session_state.mri_magnitude
        phase = st.session_state.mri_phase

        # Display side by side
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Magnitude Image (Anatomy)**")

            fig, ax = plt.subplots(figsize=(8, 8))
            ax.imshow(magnitude, cmap="gray")
            ax.set_title("Magnitude", fontsize=14, fontweight="bold")
            ax.axis("off")
            st.pyplot(fig)
            plt.close()

            st.caption("What we see: Tissue contrast and structure")

        with col2:
            st.markdown("**Phase Image**")

            fig, ax = plt.subplots(figsize=(8, 8))
            im = ax.imshow(phase, cmap="twilight")
            ax.set_title("Phase", fontsize=14, fontweight="bold")
            ax.axis("off")
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            st.pyplot(fig)
            plt.close()

            st.caption("Phase information: Used for blood flow, temperature, etc.")

        # Statistics
        st.markdown("---")
        st.subheader("Thống kê")

        col1, col2, col3, col4 = st.columns(4)

        col1.metric("Magnitude Range", f"{magnitude.min():.2f} - {magnitude.max():.2f}")
        col2.metric("Magnitude Mean", f"{magnitude.mean():.2f}")
        col3.metric("Phase Range", f"{phase.min():.2f} - {phase.max():.2f}")
        col4.metric("Phase Mean", f"{phase.mean():.2f}")

        # Download
        st.markdown("---")
        st.subheader("Tải về")

        col1, col2, col3 = st.columns(3)

        with col1:
            # Download magnitude
            npy_buffer = io.BytesIO()
            np.save(npy_buffer, magnitude)
            npy_bytes = npy_buffer.getvalue()

            st.download_button(
                label=" Download Magnitude (.npy)",
                data=npy_bytes,
                file_name="mri_magnitude.npy",
                mime="application/octet-stream",
            )

        with col2:
            # Download phase
            npy_buffer = io.BytesIO()
            np.save(npy_buffer, phase)
            npy_bytes = npy_buffer.getvalue()

            st.download_button(
                label=" Download Phase (.npy)",
                data=npy_bytes,
                file_name="mri_phase.npy",
                mime="application/octet-stream",
            )

        with col3:
            # Download magnitude as PNG
            fig_save = plt.figure(figsize=(8, 8))
            plt.imshow(magnitude, cmap="gray")
            plt.axis("off")

            img_buffer = io.BytesIO()
            plt.savefig(img_buffer, format="png", bbox_inches="tight", dpi=150)
            img_buffer.seek(0)
            plt.close()

            st.download_button(
                label=" Download Magnitude (.png)",
                data=img_buffer,
                file_name="mri_magnitude.png",
                mime="image/png",
            )

        # Interpretation section
        st.markdown("---")
        st.subheader("Giải thích kết quả tái tạo MRI")

        # Check if we have original image for comparison
        if data_source == "Generate from Image" and hasattr(
            st.session_state, "mri_original_image"
        ):
            visualizer = ResultVisualizer()

            original = st.session_state.mri_original_image

            # Normalize images to [0, 1] for visualization and comparison
            orig_norm = (original - original.min()) / (
                original.max() - original.min() + 1e-8
            )
            mag_norm = (magnitude - magnitude.min()) / (
                magnitude.max() - magnitude.min() + 1e-8
            )

            # Compare original with magnitude
            pf_info = ""
            if partial_fourier:
                pf_info = f" với Partial Fourier ({pf_percentage}%)"

            visualizer.compare_images(
                orig_norm,
                mag_norm,
                title_before="MRI gốc",
                title_after="MRI tái tạo",
                description=(
                    f"Tái tạo từ K-space{pf_info}. "
                    "Magnitude hiển thị cấu trúc giải phẫu. "
                    "Phase chứa thông tin về dòng chảy và nhiệt độ."
                ),
            )

            # Calculate quality metrics
            from skimage.metrics import (
                peak_signal_noise_ratio,
                structural_similarity,
                mean_squared_error,
            )

            psnr = peak_signal_noise_ratio(orig_norm, mag_norm, data_range=1.0)
            ssim = structural_similarity(orig_norm, mag_norm, data_range=1.0)
            mse = mean_squared_error(orig_norm, mag_norm)
            snr = psnr - 10  # Approximation

            metrics = {"PSNR": psnr, "SSIM": ssim, "MSE": mse, "SNR": snr}

            # Show metrics dashboard
            explainer = MetricsExplainer()
            explainer.show_metrics_dashboard(metrics)

            # Show interpretation
            info_dict = {
                "method": "Inverse FFT with K-space",
                "partial_fourier": partial_fourier,
            }
            if partial_fourier:
                info_dict["sampling_rate"] = pf_percentage

            show_interpretation_section(
                task_type="reconstruction", metrics=metrics, image_info=info_dict
            )
        else:
            # No comparison possible, just explain the results
            st.info(
                "**Giải thích kết quả:**\n\n"
                "- **Magnitude (Biên độ):** Hiển thị cấu trúc giải phẫu như xương, mô, dịch.\n"
                "- **Phase (Pha):** Chứa thông tin về dòng máu, nhiệt độ, và chuyển động.\n"
                "- **K-space:** Miền tần số chứa dữ liệu thô từ máy MRI.\n"
                "- **FFT:** Chuyển đổi từ K-space sang ảnh có thể nhìn thấy.\n\n"
                "Đây là công cụ hỗ trợ, không thay thế chẩn đoán y khoa chuyên nghiệp."
            )

else:
    st.info("Tạo K-space hoặc tải dữ liệu lên để bắt đầu")

    st.markdown("---")
    st.subheader("Hướng dẫn")

    st.markdown(
        """
    **Tạo từ Ảnh (Demo):**
    1. Tải lên ảnh y tế (NIfTI, DICOM, v.v.)
    2. Tùy chọn bật Partial Fourier
    3. Nhấn "Tạo K-space và Tái tạo"
    4. Xem K-space và ảnh đã tái tạo
    
    **Tải lên K-space (Dữ liệu Thực):**
    1. Chọn "Upload K-space"
    2. Tải file .npy (mảng số phức)
    3. Nhấn "Reconstruct"
    4. Tải về ảnh magnitude/phase
    
    **Hiểu Kết quả:**
    - **K-space:** Dữ liệu tần số thô từ máy quét MRI
    - **Magnitude:** Ảnh giải phẫu (những gì ta nhìn thấy)
    - **Phase:** Thông tin bổ sung (dòng máu, v.v.)
    
    **Partial Fourier:**
    - Mô phỏng quét MRI nhanh hơn
    - 75% = nhanh hơn 25% thời gian quét
    - 50% = nhanh hơn 50% (nhưng chất lượng thấp hơn)
    """
    )

# Footer
st.markdown("---")
st.caption("Mẹo: Thử Partial Fourier để thấy sự đánh đổi giữa tốc độ và chất lượng")
