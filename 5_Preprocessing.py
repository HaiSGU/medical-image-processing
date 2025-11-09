"""
Trang Tiền xử lý Ảnh

Áp dụng các phép tiền xử lý cho ảnh y tế.

Tác giả: HaiSGU
Ngày: 2025-10-28
"""

import streamlit as st
import tempfile
from pathlib import Path
import sys
import numpy as np
import matplotlib.pyplot as plt
import io
from skimage import exposure, filters
from scipy import ndimage

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import from src/ modules
from src.preprocessing.image_transforms import ImageTransforms
from utils.file_io import MedicalImageIO

# Page config
st.set_page_config(page_title=" Tiền xử lý Ảnh", layout="wide")

# Initialize session state
if "prep_image" not in st.session_state:
    st.session_state.prep_image = None
if "prep_processed" not in st.session_state:
    st.session_state.prep_processed = None
if "prep_operations" not in st.session_state:
    st.session_state.prep_operations = []

# Header
st.title(" Tiền xử lý Ảnh")
st.markdown("Biến đổi và nâng cao chất lượng ảnh y tế để phân tích")

# Info
with st.expander("About Preprocessing"):
    col1, col2 = st.columns(2)

    with col1:
        st.markdown(
            """
        **Why Preprocessing?**
        
        Raw medical images need preparation:
        - Different intensity ranges
        - Varying sizes
        - Scanner noise
        - Low contrast
        
        **Operations:**
        - Normalization
        - Denoising
        - Resizing
        - Contrast enhancement
        """
        )

    with col2:
        st.markdown(
            """
        **Recommended Order:**
        
        1. **Normalize** intensities first
        2. **Denoise** to remove noise
        3. **Resize** to target size
        4. **Enhance** contrast last
        
        **Tips:**
        - Apply operations one at a time
        - Check preview before download
        - Save pipeline for reuse
        """
        )

st.markdown("---")

# Sidebar
with st.sidebar:
    st.header(" Các phép toán")

    st.markdown("### Cường độ")

    normalize_enabled = st.checkbox("Chuẩn hóa", value=False)
    if normalize_enabled:
        norm_method = st.selectbox(
            "Phương pháp:",
            ["Min-Max (0-1)", "Z-Score", "Cắt phân vị"],
            help="Phương pháp chuẩn hóa",
        )

        if norm_method == "Cắt phân vị":
            lower_p = st.slider("Phân vị dưới", 0, 50, 2)
            upper_p = st.slider("Phân vị trên", 50, 100, 98)

    enhance_enabled = st.checkbox("Tăng cường Tương phản", value=False)
    if enhance_enabled:
        enhance_method = st.selectbox(
            "Phương pháp:",
            ["Cân bằng Histogram", "CLAHE", "Hiệu chỉnh Gamma"],
            help="Phương pháp tăng cường tương phản",
        )

        if enhance_method == "CLAHE":
            clip_limit = st.slider("Giới hạn cắt", 0.5, 5.0, 2.0, step=0.5)
        elif enhance_method == "Hiệu chỉnh Gamma":
            gamma = st.slider("Gamma", 0.1, 3.0, 1.0, step=0.1)

    st.markdown("---")
    st.markdown("### Không gian")

    resize_enabled = st.checkbox("Thay đổi kích thước", value=False)
    if resize_enabled:
        target_size = st.slider("Kích thước đích", 64, 512, 256, step=64)

    crop_enabled = st.checkbox("Cắt theo Nội dung", value=False)

    st.markdown("---")
    st.markdown("### Khử nhiễu")

    denoise_enabled = st.checkbox("Khử nhiễu", value=False)
    if denoise_enabled:
        denoise_method = st.selectbox(
            "Phương pháp:", ["Gaussian", "Median"], help="Phương pháp khử nhiễu"
        )

        if denoise_method == "Gaussian":
            sigma = st.slider("Sigma", 0.1, 5.0, 1.0, step=0.1)
        else:
            kernel_size = st.slider("Kích thước Kernel", 3, 11, 5, step=2)

    st.markdown("---")
    st.info(" Bật các phép toán theo thứ tự khuyến nghị")

# Upload
st.subheader(" Tải lên Ảnh")

uploaded_file = st.file_uploader(
    "Chọn file ảnh (.nii, .dcm, .nrrd, .mha, .npy)",
    type=["nii", "gz", "dcm", "nrrd", "mha", "npy"],
    help="Tải lên ảnh y tế để tiền xử lý",
)

if uploaded_file:
    # Load image
    with tempfile.NamedTemporaryFile(
        delete=False, suffix=Path(uploaded_file.name).suffix
    ) as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_path = tmp_file.name

    try:
        with st.spinner("Loading image..."):
            io_handler = MedicalImageIO()
            image_data, metadata = io_handler.read_image(tmp_path)

            # Use 2D slice if 3D
            if image_data.ndim == 3:
                slice_idx = image_data.shape[2] // 2
                image_2d = image_data[:, :, slice_idx]
                st.info(f"Using middle slice ({slice_idx}) from 3D volume")
            else:
                image_2d = image_data

            st.session_state.prep_image = image_2d

        st.success(f" Loaded: {image_2d.shape}")

        # Show original
        col1, col2, col3 = st.columns(3)
        col1.metric("Shape", f"{image_2d.shape[0]}×{image_2d.shape[1]}")
        col2.metric("Range", f"{image_2d.min():.1f} - {image_2d.max():.1f}")
        col3.metric("Mean", f"{image_2d.mean():.1f}")

    except Exception as e:
        st.error(f" Error loading image: {str(e)}")
        st.stop()

    st.markdown("---")

    # Apply preprocessing button
    if st.button(" Apply Preprocessing", type="primary", use_container_width=True):

        with st.spinner("Processing..."):
            try:
                processed = image_2d.copy()
                operations = []
                transformer = ImageTransforms()

                # 1. Normalize
                if normalize_enabled:
                    if norm_method == "Min-Max (0-1)":
                        processed = transformer.normalize_intensity(
                            processed, method="minmax"
                        )
                        operations.append("Normalize (Min-Max)")

                    elif norm_method == "Z-Score":
                        processed = transformer.normalize_intensity(
                            processed, method="zscore"
                        )
                        operations.append("Normalize (Z-Score)")

                    else:  # Percentile Clipping
                        processed = transformer.normalize_intensity(
                            processed,
                            method="percentile",
                            percentile_range=(lower_p, upper_p),
                        )
                        operations.append(f"Normalize (Percentile {lower_p}-{upper_p})")

                # 2. Denoise
                if denoise_enabled:
                    if denoise_method == "Gaussian":
                        processed = filters.gaussian(processed, sigma=sigma)
                        operations.append(f"Gaussian Blur (σ={sigma})")
                    else:  # Median
                        processed = filters.median(
                            processed, footprint=np.ones((kernel_size, kernel_size))
                        )
                        operations.append(f"Median Filter (k={kernel_size})")

                # 3. Resize
                if resize_enabled:
                    from skimage.transform import resize

                    processed = resize(
                        processed,
                        (target_size, target_size),
                        anti_aliasing=True,
                        preserve_range=True,
                    )
                    operations.append(f"Resize ({target_size}×{target_size})")

                # 4. Crop to content
                if crop_enabled:
                    # Find non-zero bounding box
                    coords = np.argwhere(processed > np.percentile(processed, 5))
                    if len(coords) > 0:
                        y_min, x_min = coords.min(axis=0)
                        y_max, x_max = coords.max(axis=0)
                        processed = processed[y_min : y_max + 1, x_min : x_max + 1]
                        operations.append("Crop to Content")

                # 5. Enhance contrast
                if enhance_enabled:
                    if enhance_method == "Histogram Equalization":
                        processed = exposure.equalize_hist(processed)
                        operations.append("Histogram Equalization")

                    elif enhance_method == "CLAHE":
                        processed = exposure.equalize_adapthist(
                            processed, clip_limit=clip_limit
                        )
                        operations.append(f"CLAHE (clip={clip_limit})")

                    else:  # Gamma
                        processed = exposure.adjust_gamma(processed, gamma)
                        operations.append(f"Gamma Correction (γ={gamma})")

                # Store results
                st.session_state.prep_processed = processed
                st.session_state.prep_operations = operations

                st.success("Preprocessing complete!")

            except Exception as e:
                st.error(f" Error during preprocessing: {str(e)}")
                st.exception(e)

    # Display results
    if st.session_state.prep_processed is not None:
        st.markdown("---")
        st.header("Results")

        original = st.session_state.prep_image
        processed = st.session_state.prep_processed

        # Statistics comparison
        st.subheader("Statistics Comparison")

        col1, col2, col3, col4 = st.columns(4)

        col1.metric(
            "Shape",
            f"{processed.shape[0]}×{processed.shape[1]}",
            delta=f"Original: {original.shape[0]}×{original.shape[1]}",
        )

        col2.metric(
            "Min Value",
            f"{processed.min():.2f}",
            delta=f"{processed.min() - original.min():.2f}",
        )

        col3.metric(
            "Max Value",
            f"{processed.max():.2f}",
            delta=f"{processed.max() - original.max():.2f}",
        )

        col4.metric(
            "Mean",
            f"{processed.mean():.2f}",
            delta=f"{processed.mean() - original.mean():.2f}",
        )

        # Before/After visualization
        st.markdown("---")
        st.subheader(" Before/After Comparison")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Original Image**")

            fig, ax = plt.subplots(figsize=(8, 8))
            ax.imshow(original, cmap="gray")
            ax.set_title("Original", fontsize=14, fontweight="bold")
            ax.axis("off")
            st.pyplot(fig)
            plt.close()

            st.caption(
                f"Shape: {original.shape} | "
                f"Range: [{original.min():.1f}, {original.max():.1f}]"
            )

        with col2:
            st.markdown("**Processed Image**")

            fig, ax = plt.subplots(figsize=(8, 8))
            ax.imshow(processed, cmap="gray")
            ax.set_title("Processed", fontsize=14, fontweight="bold")
            ax.axis("off")
            st.pyplot(fig)
            plt.close()

            st.caption(
                f"Shape: {processed.shape} | "
                f"Range: [{processed.min():.1f}, {processed.max():.1f}]"
            )

        # Histogram comparison
        st.markdown("---")
        st.subheader("Intensity Distribution")

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4))

        # Original histogram
        ax1.hist(original.flatten(), bins=50, color="steelblue", alpha=0.7)
        ax1.set_title("Original Histogram", fontweight="bold")
        ax1.set_xlabel("Intensity")
        ax1.set_ylabel("Frequency")
        ax1.grid(alpha=0.3)

        # Processed histogram
        ax2.hist(processed.flatten(), bins=50, color="green", alpha=0.7)
        ax2.set_title("Processed Histogram", fontweight="bold")
        ax2.set_xlabel("Intensity")
        ax2.set_ylabel("Frequency")
        ax2.grid(alpha=0.3)

        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        # Pipeline summary
        st.markdown("---")
        st.subheader("📋 Pipeline Summary")

        if st.session_state.prep_operations:
            for i, op in enumerate(st.session_state.prep_operations, 1):
                st.markdown(f"{i}. {op}")
        else:
            st.info("No operations applied")

        # Download
        st.markdown("---")
        st.subheader("Download")

        col1, col2, col3 = st.columns(3)

        with col1:
            # Download as NumPy
            npy_buffer = io.BytesIO()
            np.save(npy_buffer, processed)
            npy_bytes = npy_buffer.getvalue()

            st.download_button(
                label="📥 Download Image (.npy)",
                data=npy_bytes,
                file_name="preprocessed_image.npy",
                mime="application/octet-stream",
            )

        with col2:
            # Download as PNG
            fig_save = plt.figure(figsize=(8, 8))
            plt.imshow(processed, cmap="gray")
            plt.axis("off")

            img_buffer = io.BytesIO()
            plt.savefig(img_buffer, format="png", bbox_inches="tight", dpi=150)
            img_buffer.seek(0)
            plt.close()

            st.download_button(
                label="📥 Download Image (.png)",
                data=img_buffer,
                file_name="preprocessed_image.png",
                mime="image/png",
            )

        with col3:
            # Download pipeline config
            import json

            config = {"operations": st.session_state.prep_operations, "parameters": {}}

            if normalize_enabled:
                config["parameters"]["normalize"] = {"method": norm_method}
                if norm_method == "Percentile Clipping":
                    config["parameters"]["normalize"]["percentiles"] = (
                        lower_p,
                        upper_p,
                    )

            if denoise_enabled:
                config["parameters"]["denoise"] = {"method": denoise_method}
                if denoise_method == "Gaussian":
                    config["parameters"]["denoise"]["sigma"] = sigma
                else:
                    config["parameters"]["denoise"]["kernel_size"] = kernel_size

            if resize_enabled:
                config["parameters"]["resize"] = {"size": target_size}

            if enhance_enabled:
                config["parameters"]["enhance"] = {"method": enhance_method}
                if enhance_method == "CLAHE":
                    config["parameters"]["enhance"]["clip_limit"] = clip_limit
                elif enhance_method == "Gamma Correction":
                    config["parameters"]["enhance"]["gamma"] = gamma

            json_str = json.dumps(config, indent=2)

            st.download_button(
                label="📥 Download Config (.json)",
                data=json_str,
                file_name="pipeline_config.json",
                mime="application/json",
            )

else:
    st.info("👆 Upload an image to start preprocessing")

    st.markdown("---")
    st.subheader("Quick Guide")

    st.markdown(
        """
    **How to use:**
    1. Upload medical image
    2. Enable desired operations (sidebar)
    3. Adjust parameters for each operation
    4. Click "Apply Preprocessing"
    5. Compare before/after results
    6. Download processed image
    
    **Recommended workflow:**
    - Start with **Normalize** (Min-Max)
    - Add **Denoise** if image is noisy
    - Use **Resize** to standardize size
    - Apply **CLAHE** for better contrast
    
    **Tips:**
    - Apply one operation at a time to see effect
    - Check histogram to verify normalization
    - Save pipeline config for reproducibility
    - Use Percentile Clipping for outlier-heavy images
    """
    )

# Footer
st.markdown("---")
st.caption(" Tip: Apply operations in order - Normalize → Denoise → Resize → Enhance")
