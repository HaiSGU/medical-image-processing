"""
Trang Tái tạo CT

Tái tạo ảnh CT từ sinogram sử dụng thuật toán FBP và SART.

Tác giả: HaiSGU
Ngày: 2025-10-28
"""

import io
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.reconstruction.ct_reconstruction import CTReconstructor
from utils.interpretation import (
    ResultVisualizer,
    MetricsExplainer,
    show_interpretation_section,
)

# Page config
st.set_page_config(page_title=" Tái tạo CT", layout="wide")


# Helper functions
def create_shepp_logan_phantom(size=256):
    """Create synthetic Shepp-Logan phantom for testing."""
    from skimage.data import shepp_logan_phantom
    from skimage.transform import resize

    phantom = shepp_logan_phantom()
    if phantom.shape[0] != size:
        phantom = resize(phantom, (size, size), anti_aliasing=True)

    return phantom


def calculate_psnr(original, reconstructed):
    """Calculate Peak Signal-to-Noise Ratio."""
    mse = np.mean((original - reconstructed) ** 2)
    if mse == 0:
        return float("inf")
    max_pixel = np.max(original)
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr


def calculate_ssim(original, reconstructed):
    """Calculate Structural Similarity Index (simplified)."""
    # Simplified SSIM calculation
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


# Initialize session state
if "ct_sinogram" not in st.session_state:
    st.session_state.ct_sinogram = None
if "ct_phantom" not in st.session_state:
    st.session_state.ct_phantom = None
if "ct_reconstructed" not in st.session_state:
    st.session_state.ct_reconstructed = None

# Header
st.title("Tái tạo CT")
st.markdown("Tái tạo ảnh CT từ dữ liệu chiếu (sinogram)")

# Info
with st.expander(" Về Tái tạo CT"):
    col1, col2 = st.columns(2)

    with col1:
        st.markdown(
            """
        **Tái tạo CT là gì?**
        
        Máy CT quay nguồn tia X quanh bệnh nhân,
        chụp hình chiếu ở các góc khác nhau.
        
        **Sinogram:** Tập hợp tất cả hình chiếu
        - Mỗi hàng = một góc
        - Chứa dữ liệu chiếu
        
        **Tái tạo:** Chuyển sinogram → ảnh CT
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
        
        **SART (Lặp):**
        - Chậm hơn nhưng chất lượng tốt hơn
        - Tốt cho dữ liệu thưa
        - Giảm nhiễu
        """
        )

st.markdown("---")

# Sidebar controls
with st.sidebar:
    st.header("Cài đặt")

    # Data source
    data_source = st.radio(
        "Nguồn dữ liệu:",
        ["Tạo Phantom", "Tải lên Sinogram"],
        help="Dùng phantom cho demo hoặc tải lên dữ liệu thật",
    )

    st.markdown("---")

    # Reconstruction method
    method = st.selectbox("Phương pháp:", ["FBP", "SART"], help="Thuật toán tái tạo")

    if method == "FBP":
        filter_type = st.selectbox(
            "Bộ lọc:",
            ["ramp", "shepp-logan", "cosine", "hamming"],
            help="Bộ lọc cho tái tạo FBP",
        )
    else:
        num_iterations = st.slider(
            "Số lần lặp:",
            min_value=1,
            max_value=50,
            value=10,
            help="Số lần lặp SART",
        )

        relaxation = st.slider(
            "Hệ số thư giãn:",
            min_value=0.1,
            max_value=1.0,
            value=0.5,
            step=0.1,
            help="Hệ số thư giãn SART",
        )

    st.markdown("---")
    st.info("Thử FBP với bộ lọc 'ramp' trước để có kết quả tốt nhất")

# Main content
if data_source == "Tạo Phantom":
    st.subheader("Shepp-Logan Phantom")

    col1, col2 = st.columns(2)

    with col1:
        phantom_size = st.slider("Kích thước Phantom:", 64, 512, 256, step=64)
        num_angles = st.slider("Số góc:", 30, 360, 180, step=30)

    with col2:
        st.markdown(
            """
        **Shepp-Logan Phantom:**
        - Ảnh test chuẩn cho CT
        - Chứa các hình elip có mật độ khác nhau
        - Hoàn hảo để test thuật toán
        """
        )

    if st.button("Tạo & Tái tạo", type="primary", use_container_width=True):

        with st.spinner("Đang tạo phantom và sinogram..."):
            # Create phantom
            phantom = create_shepp_logan_phantom(phantom_size)
            st.session_state.ct_phantom = phantom

            # Generate sinogram using radon transform
            from skimage.transform import radon

            angles = np.linspace(0, 180, num_angles, endpoint=False)
            sinogram_raw = radon(phantom, theta=angles)
            # radon returns (num_detectors, num_angles)
            # CTReconstructor expects (num_angles, num_detectors)
            sinogram = sinogram_raw.T
            st.session_state.ct_sinogram = sinogram

        with st.spinner(f"Đang tái tạo sử dụng {method}..."):
            # Create reconstructor with sinogram
            reconstructor = CTReconstructor(sinogram, theta=angles)

            # Reconstruct
            if method == "FBP":
                reconstructed = reconstructor.reconstruct_fbp(filter_name=filter_type)
            else:  # SART
                reconstructed = reconstructor.reconstruct_sart(
                    iterations=num_iterations,
                    relaxation=relaxation,
                    image_size=phantom_size,
                )

            st.session_state.ct_reconstructed = reconstructed

        st.success("Tái tạo hoàn tất!")

else:  # Upload Sinogram
    st.subheader("Tải lên Sinogram")

    uploaded_file = st.file_uploader(
        "Chọn file sinogram (.npy)",
        type=["npy"],
        help="Mảng NumPy chứa dữ liệu chiếu",
    )

    if uploaded_file:
        try:
            sinogram = np.load(io.BytesIO(uploaded_file.getvalue()))
            st.session_state.ct_sinogram = sinogram

            st.success(f"Đã tải sinogram: {sinogram.shape}")

            # Reconstruct button
            if st.button("Tái tạo", type="primary", use_container_width=True):

                with st.spinner(f"Đang tái tạo sử dụng {method}..."):
                    # Create angles for reconstruction
                    num_angles = sinogram.shape[0]
                    angles = np.linspace(0, 180, num_angles, endpoint=False)

                    # Create reconstructor with sinogram
                    reconstructor = CTReconstructor(sinogram, theta=angles)

                    if method == "FBP":
                        reconstructed = reconstructor.reconstruct_fbp(
                            filter_name=filter_type
                        )
                    else:  # SART
                        image_size = sinogram.shape[1]
                        reconstructed = reconstructor.reconstruct_sart(
                            iterations=num_iterations,
                            relaxation=relaxation,
                            image_size=image_size,
                        )

                    st.session_state.ct_reconstructed = reconstructed

                st.success("Tái tạo hoàn tất!")

        except Exception as e:
            st.error(f"Lỗi khi tải sinogram: {str(e)}")

# Display results
if st.session_state.ct_sinogram is not None:
    st.markdown("---")
    st.header("Kết quả")

    sinogram = st.session_state.ct_sinogram

    # Show sinogram
    st.subheader("Sinogram (Dữ liệu chiếu)")

    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(sinogram, cmap="gray", aspect="auto")
    ax.set_xlabel("Detector Position", fontsize=12)
    ax.set_ylabel("Projection Angle", fontsize=12)
    ax.set_title("Sinogram", fontsize=14, fontweight="bold")
    plt.colorbar(im, ax=ax, label="Intensity")
    st.pyplot(fig)
    plt.close()

    st.caption(f"Shape: {sinogram.shape[0]} angles × {sinogram.shape[1]} detectors")

    # Show reconstruction if available
    if st.session_state.ct_reconstructed is not None:
        st.markdown("---")
        st.subheader("Ảnh CT")

        reconstructed = st.session_state.ct_reconstructed

        # Display controls
        col1, col2 = st.columns([3, 1])

        with col2:
            colormap = st.selectbox("Colormap:", ["gray", "bone", "hot"], index=1)
            show_colorbar = st.checkbox("Colorbar", value=True)

        # Plot reconstruction
        fig, ax = plt.subplots(figsize=(10, 10))
        im = ax.imshow(reconstructed, cmap=colormap)
        ax.axis("off")
        ax.set_title(f"Reconstructed Image ({method})", fontsize=14, fontweight="bold")

        if show_colorbar:
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        st.pyplot(fig)
        plt.close()

        # Quality metrics (if phantom available)
        if st.session_state.ct_phantom is not None:
            st.markdown("---")
            st.subheader("Chỉ số")

            phantom = st.session_state.ct_phantom

            # Ensure same size
            if phantom.shape != reconstructed.shape:
                from skimage.transform import resize

                phantom = resize(phantom, reconstructed.shape, anti_aliasing=True)

            # Calculate metrics
            psnr = calculate_psnr(phantom, reconstructed)
            ssim = calculate_ssim(phantom, reconstructed)

            col1, col2, col3 = st.columns(3)

            col1.metric("PSNR (dB)", f"{psnr:.2f}")
            col2.metric("SSIM", f"{ssim:.4f}")
            col3.metric("Max Error", f"{np.max(np.abs(phantom - reconstructed)):.4f}")

            # Comparison plot
            st.markdown("---")
            st.subheader("So sánh: Gốc và Tái tạo")

            fig, axes = plt.subplots(1, 3, figsize=(15, 5))

            # Original
            axes[0].imshow(phantom, cmap="gray")
            axes[0].set_title("Original Phantom", fontweight="bold")
            axes[0].axis("off")

            # Reconstructed
            axes[1].imshow(reconstructed, cmap="gray")
            axes[1].set_title(f"Reconstructed ({method})", fontweight="bold")
            axes[1].axis("off")

            # Difference
            diff = np.abs(phantom - reconstructed)
            im = axes[2].imshow(diff, cmap="hot")
            axes[2].set_title("Absolute Difference", fontweight="bold")
            axes[2].axis("off")
            plt.colorbar(im, ax=axes[2], fraction=0.046)

            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

        # Download
        st.markdown("---")
        st.subheader("Tải về")

        col1, col2 = st.columns(2)

        with col1:
            # Download as NumPy
            npy_buffer = io.BytesIO()
            np.save(npy_buffer, reconstructed)
            npy_bytes = npy_buffer.getvalue()

            st.download_button(
                label=" Download Image (.npy)",
                data=npy_bytes,
                file_name=f"ct_reconstructed_{method.lower()}.npy",
                mime="application/octet-stream",
            )

        with col2:
            # Download as PNG
            fig_save = plt.figure(figsize=(8, 8))
            plt.imshow(reconstructed, cmap="gray")
            plt.axis("off")

            img_buffer = io.BytesIO()
            plt.savefig(img_buffer, format="png", bbox_inches="tight", dpi=150)
            img_buffer.seek(0)
            plt.close()

            st.download_button(
                label=" Download Image (.png)",
                data=img_buffer,
                file_name=f"ct_reconstructed_{method.lower()}.png",
                mime="image/png",
            )

        # Interpretation section
        st.markdown("---")
        st.subheader("Giải thích kết quả tái tạo CT")

        # Compare with phantom if available
        if st.session_state.ct_phantom is not None:
            visualizer = ResultVisualizer()

            phantom = st.session_state.ct_phantom

            # Resize if needed
            if phantom.shape != reconstructed.shape:
                from skimage.transform import resize

                phantom = resize(phantom, reconstructed.shape, anti_aliasing=True)

            # Normalize images to [0, 1] for visualization
            phantom_norm = (phantom - phantom.min()) / (
                phantom.max() - phantom.min() + 1e-8
            )
            recon_norm = (reconstructed - reconstructed.min()) / (
                reconstructed.max() - reconstructed.min() + 1e-8
            )

            # Show comparison
            visualizer.compare_images(
                phantom_norm,
                recon_norm,
                title_before="Phantom gốc",
                title_after=f"CT tái tạo ({method})",
                description=(
                    f"Tái tạo từ {sinogram.shape[0]} góc quét. "
                    f"Phương pháp {method}: "
                    f"{'Nhanh nhưng có thể có artifacts' if method == 'FBP' else 'Chất lượng cao hơn nhưng chậm hơn'}."
                ),
            )

            # Calculate quality metrics using normalized data
            from skimage.metrics import (
                peak_signal_noise_ratio,
                structural_similarity,
                mean_squared_error,
            )

            psnr = peak_signal_noise_ratio(phantom_norm, recon_norm, data_range=1.0)
            ssim = structural_similarity(phantom_norm, recon_norm, data_range=1.0)
            mse = mean_squared_error(phantom_norm, recon_norm)
            snr = psnr - 10  # Approximation

            metrics = {"PSNR": psnr, "SSIM": ssim, "MSE": mse, "SNR": snr}

            # Show metrics dashboard
            explainer = MetricsExplainer()
            explainer.show_metrics_dashboard(metrics)

            # Show interpretation
            info_dict = {"method": method, "num_angles": sinogram.shape[0]}

            # Add method-specific parameters
            if method == "FBP" and "filter_type" in locals():
                info_dict["filter"] = filter_type
            elif method == "SART" and "num_iterations" in locals():
                info_dict["iterations"] = num_iterations

            show_interpretation_section(
                task_type="reconstruction", metrics=metrics, image_info=info_dict
            )

else:
    st.info("Tạo phantom hoặc tải sinogram lên để bắt đầu")

    st.markdown("---")
    st.subheader("Hướng dẫn")

    st.markdown(
        """
    **Sử dụng Phantom (Demo):**
    1. Giữ "Tạo Phantom" được chọn
    2. Điều chỉnh kích thước phantom và góc quét
    3. Nhấn "Generate & Reconstruct"
    4. So sánh phương pháp FBP và SART
    
    **Sử dụng Dữ liệu Thực:**
    1. Chọn "Tải lên Sinogram"
    2. Tải file .npy chứa sinogram
    3. Chọn phương pháp tái tạo
    4. Nhấn "Reconstruct"
    
    **Mẹo:**
    - FBP nhanh hơn, SART chất lượng tốt hơn
    - Nhiều góc quét = tái tạo tốt hơn
    - Thử các bộ lọc khác nhau cho FBP
    - SART thường cần 10-20 lần lặp
    """
    )

# Footer
st.markdown("---")
st.caption("Mẹo: Dùng Shepp-Logan phantom để thử nghiệm các tham số tái tạo khác nhau")
