"""
Image Registration Page

Đăng ký / Căn chỉnh ảnh y tế (Medical Image Registration)

"""

import streamlit as st
import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.file_io import MedicalImageIO
from src.registration.image_registration import (
    ImageRegistration,
    numpy_to_sitk,
    sitk_to_numpy,
)

# Page config
st.set_page_config(
    page_title="Image Registration - Medical Image Processing",
    page_icon="🔄",
    layout="wide",
)

st.title("🔄 Image Registration (Đăng Ký Ảnh)")

st.markdown(
    """
**Image Registration** là quá trình căn chỉnh hai hoặc nhiều ảnh để chúng nằm trên cùng một hệ tọa độ.

**Ứng dụng:**
- 📊 So sánh ảnh trước/sau điều trị
- 🧠 Đăng ký ảnh multi-modal (T1-T2 MRI)
- 📈 Theo dõi sự phát triển của khối u
- 🎯 Căn chỉnh ảnh theo atlas
"""
)

# Initialize session state
if "fixed_image" not in st.session_state:
    st.session_state.fixed_image = None
if "moving_image" not in st.session_state:
    st.session_state.moving_image = None
if "registered_image" not in st.session_state:
    st.session_state.registered_image = None
if "registration_transform" not in st.session_state:
    st.session_state.registration_transform = None
if "registration_metrics" not in st.session_state:
    st.session_state.registration_metrics = {}

# Sidebar for parameters
st.sidebar.header("⚙️ Tham số")

registration_type = st.sidebar.selectbox(
    "Loại đăng ký",
    ["Rigid", "Affine", "Deformable"],
    help="""
    - Rigid: Chỉ di chuyển + xoay
    - Affine: + scaling + shearing
    - Deformable: Biến dạng cục bộ
    """,
)

# Type-specific parameters
if registration_type == "Rigid":
    st.sidebar.info("💡 Rigid: Translation + Rotation only (6 DOF)")
    iterations = st.sidebar.slider("Số lần lặp", 50, 300, 100, step=10)
    learning_rate = st.sidebar.slider("Learning rate", 0.1, 5.0, 1.0, step=0.1)
elif registration_type == "Affine":
    st.sidebar.info("💡 Affine: + Scaling + Shearing (12 DOF)")
    iterations = st.sidebar.slider("Số lần lặp", 50, 300, 150, step=10)
    learning_rate = st.sidebar.slider("Learning rate", 0.1, 5.0, 1.0, step=0.1)
else:  # Deformable
    st.sidebar.warning("⚠️ Deformable: Chậm nhất nhưng chính xác nhất")
    iterations = st.sidebar.slider("Số lần lặp", 20, 100, 50, step=5)
    mesh_size = st.sidebar.slider(
        "Mesh size", 3, 10, 5, step=1, help="Nhỏ hơn = linh hoạt hơn (nhưng chậm hơn)"
    )

metric = st.sidebar.selectbox(
    "Similarity metric",
    ["mean_squares", "mutual_information"],
    help="""
    - Mean Squares: Cho cùng modality
    - Mutual Information: Cho multi-modal
    """,
)

# Main content - two columns
col1, col2 = st.columns(2)

with col1:
    st.subheader("📍 Fixed Image (Ảnh tham chiếu)")
    fixed_file = st.file_uploader(
        "Tải ảnh Fixed",
        type=["nii", "gz", "dcm", "nrrd", "mha", "npy"],
        key="fixed",
        help="Ảnh này sẽ giữ nguyên vị trí",
    )

    if fixed_file:
        try:
            # Save to temp file
            import tempfile
            import os

            # Create temp file with correct extension
            suffix = Path(fixed_file.name).suffix
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(fixed_file.getvalue())
                tmp_path = tmp.name

            try:
                io = MedicalImageIO()
                fixed_array, _ = io.read_image(tmp_path)

                # Handle complex arrays
                if np.iscomplexobj(fixed_array):
                    fixed_array = np.abs(fixed_array)

                # Get middle slice for display
                if fixed_array.ndim == 3:
                    slice_idx = fixed_array.shape[0] // 2
                    display_slice = fixed_array[slice_idx, :, :]
                else:
                    display_slice = fixed_array

                # Display
                fig, ax = plt.subplots(figsize=(6, 6))
                ax.imshow(display_slice, cmap="gray")
                ax.set_title("Fixed Image")
                ax.axis("off")
                st.pyplot(fig)
                plt.close()

                # Store in session_state
                st.session_state.fixed_image = fixed_array
                st.success(f"✅ Loaded: {fixed_array.shape}")

            finally:
                # Clean up temp file
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)

        except Exception as e:
            st.error(f"❌ Lỗi: {e}")

with col2:
    st.subheader("🔄 Moving Image (Ảnh cần căn chỉnh)")
    moving_file = st.file_uploader(
        "Tải ảnh Moving",
        type=["nii", "gz", "dcm", "nrrd", "mha", "npy"],
        key="moving",
        help="Ảnh này sẽ được di chuyển để khớp với Fixed",
    )

    if moving_file:
        try:
            # Save to temp file
            import tempfile
            import os

            # Create temp file with correct extension
            suffix = Path(moving_file.name).suffix
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(moving_file.getvalue())
                tmp_path = tmp.name

            try:
                io = MedicalImageIO()
                moving_array, _ = io.read_image(tmp_path)

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

                st.session_state.moving_image = moving_array
                st.success(f"✅ Loaded: {moving_array.shape}")

            finally:
                # Clean up temp file
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)

        except Exception as e:
            st.error(f"❌ Lỗi: {e}")

# Registration button
st.markdown("---")
col_left, col_center, col_right = st.columns([1, 2, 1])

with col_center:
    if st.button(" Bắt đầu Registration", type="primary", use_container_width=True):
        if st.session_state.fixed_image is None:
            st.error(" Vui lòng tải Fixed image!")
        elif st.session_state.moving_image is None:
            st.error(" Vui lòng tải Moving image!")
        else:
            try:
                with st.spinner(
                    f"⏳ Đang thực hiện {registration_type} registration..."
                ):
                    # Convert to SimpleITK
                    fixed_sitk = numpy_to_sitk(st.session_state.fixed_image)
                    moving_sitk = numpy_to_sitk(st.session_state.moving_image)

                    # Initialize registration
                    reg = ImageRegistration(fixed_sitk, moving_sitk, verbose=False)

                    # Run registration based on type
                    if registration_type == "Rigid":
                        registered_sitk = reg.rigid_registration(
                            number_of_iterations=iterations,
                            learning_rate=learning_rate,
                            metric=metric,
                        )
                    elif registration_type == "Affine":
                        registered_sitk = reg.affine_registration(
                            number_of_iterations=iterations,
                            learning_rate=learning_rate,
                            metric=metric,
                        )
                    else:  # Deformable
                        registered_sitk = reg.deformable_registration(
                            number_of_iterations=iterations,
                            mesh_size=mesh_size,
                            metric=metric,
                        )

                    # Convert back to numpy
                    st.session_state.registered_image = sitk_to_numpy(registered_sitk)
                    st.session_state.registration_transform = reg.get_transform()
                    st.session_state.registration_metrics = reg.get_metrics()

                st.success("✅ Registration hoàn thành!")

            except Exception as e:
                st.error(f"❌ Lỗi registration: {e}")
                import traceback

                st.code(traceback.format_exc())

# Results
if st.session_state.registered_image is not None:
    st.markdown("---")
    st.header(" Kết quả")

    # Metrics
    if st.session_state.registration_metrics:
        st.subheader(" Metrics")
        metrics = st.session_state.registration_metrics

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("MSE Before", f"{metrics['mse_before']:.2f}")
        with col2:
            st.metric(
                "MSE After",
                f"{metrics['mse_after']:.2f}",
                delta=f"{metrics['mse_improvement']:.1f}%",
                delta_color="inverse",
            )
        with col3:
            st.metric("NCC Before", f"{metrics['ncc_before']:.3f}")
        with col4:
            st.metric(
                "NCC After",
                f"{metrics['ncc_after']:.3f}",
                delta=f"{metrics['ncc_improvement']:.1f}%",
            )

    # Visualization
    st.subheader(" Comparison")

    viz_mode = st.radio(
        "Chế độ hiển thị",
        ["Side by Side", "Overlay", "Checkerboard", "Difference"],
        horizontal=True,
    )

    # Get slices
    if st.session_state.fixed_image.ndim == 3:
        slice_idx = st.slider(
            "Chọn slice",
            0,
            st.session_state.fixed_image.shape[0] - 1,
            st.session_state.fixed_image.shape[0] // 2,
        )
        fixed_slice = st.session_state.fixed_image[slice_idx, :, :]
        moving_slice = st.session_state.moving_image[slice_idx, :, :]
        registered_slice = st.session_state.registered_image[slice_idx, :, :]
    else:
        fixed_slice = st.session_state.fixed_image
        moving_slice = st.session_state.moving_image
        registered_slice = st.session_state.registered_image

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
        alpha = st.slider("Transparency", 0.0, 1.0, 0.5, 0.1)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

        # Before
        ax1.imshow(fixed_slice, cmap="Reds", alpha=1.0)
        ax1.imshow(moving_slice, cmap="Blues", alpha=alpha)
        ax1.set_title("Before Registration (Red=Fixed, Blue=Moving)")
        ax1.axis("off")

        # After
        ax2.imshow(fixed_slice, cmap="Reds", alpha=1.0)
        ax2.imshow(registered_slice, cmap="Blues", alpha=alpha)
        ax2.set_title("After Registration (Red=Fixed, Blue=Registered)")
        ax2.axis("off")

        st.pyplot(fig)
        plt.close()

    elif viz_mode == "Checkerboard":
        # Checkerboard pattern
        def create_checkerboard(img1, img2, squares=8):
            h, w = img1.shape
            result = img1.copy()
            square_h = h // squares
            square_w = w // squares

            for i in range(squares):
                for j in range(squares):
                    if (i + j) % 2 == 0:
                        result[
                            i * square_h : (i + 1) * square_h,
                            j * square_w : (j + 1) * square_w,
                        ] = img2[
                            i * square_h : (i + 1) * square_h,
                            j * square_w : (j + 1) * square_w,
                        ]
            return result

        squares = st.slider("Number of squares", 4, 16, 8, 2)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

        checker_before = create_checkerboard(fixed_slice, moving_slice, squares)
        checker_after = create_checkerboard(fixed_slice, registered_slice, squares)

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

        # Stats
        st.info(
            f" Mean absolute difference: Before = {diff_before.mean():.2f}, After = {diff_after.mean():.2f}"
        )

# Transform save/load
if st.session_state.registration_transform is not None:
    st.markdown("---")
    st.subheader("💾 Transform")

    col1, col2 = st.columns(2)

    with col1:
        if st.button("💾 Lưu Transform"):
            try:
                output_path = "registration_transform.tfm"
                sitk.WriteTransform(
                    st.session_state.registration_transform, output_path
                )
                st.success(f" Transform saved: {output_path}")
            except Exception as e:
                st.error(f"❌ Lỗi: {e}")

    with col2:
        transform_type = type(st.session_state.registration_transform).__name__
        st.info(f" Transform type: {transform_type}")

# Usage Guide Section
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
- **Số lần lặp**: 100 iterati ons  
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

**Lựa chọn Similarity Metric:**
- **Mean Squares**: Cho cùng modality (MRI-MRI, CT-CT)  
- **Mutual Information**: Cho multi-modal (T1-T2 MRI, CT-MRI, PET-CT)

**Mẹo hữu ích:**
- ⭐ Bắt đầu với Rigid → nếu không đủ tốt → thử Affine → cuối cùng Deformable
- ⭐ Mean Squares nhanh hơn Mutual Information
- ⭐ Với 3D volume: Chọn slice ở giữa để xem kết quả
- ⭐ MSE thấp và NCC cao = registration tốt
- ⭐ Deformable có thể mất 1-2 phút cho ảnh lớn
- ⭐ Learning rate quá cao → không hội tụ, quá thấp → chậm
- ⭐ Lưu transform để áp dụng cho ảnh khác

**Hiểu Kết quả:**
- **MSE Before vs After**: Giảm = tốt (thấp hơn = khớp hơn)
- **NCC**: Cao hơn = tốt (gần 1.0 = rất khớp)
- **Visualizations**: 
  - Side by Side: So sánh trực quan
  - Overlay: Xem sự chồng lấp (đỏ=fixed, xanh=moving/registered)
  - Checkerboard: Dễ thấy sai lệch
  - Difference: Vùng sáng = sai khác lớn

**Lưu ý:**
- Ảnh Fixed và Moving nên có FOV (field of view) tương tự
- Ảnh nên được preprocessing trước (normalization, bias correction)
- Deformable registration không phù hợp cho ảnh có artifacts lớn
"""
)

# Information
with st.expander(" Thông tin"):
    st.markdown(
        """
    ### Registration Types
    
    **Rigid (Cứng nhắc):**
    -  Chỉ translation + rotation
    -  Nhanh nhất
    -  Phù hợp: Follow-up scans, head motion
    -  Không thích hợp cho: Inter-subject, tumor growth
    
    **Affine:**
    -  + Scaling + shearing
    -  Trung bình về tốc độ
    -  Phù hợp: Inter-subject, CT-MRI
    -  Không thích hợp cho: Deformation phức tạp
    
    **Deformable (Non-rigid):**
    -  Biến dạng cục bộ
    -  Chính xác nhất
    -  Phù hợp: Tumor tracking, breathing motion
    -  Chậm nhất, cần computing power cao
    
    ### Metrics
    
    **MSE (Mean Squared Error):**
    - Thấp hơn = tốt hơn
    - Phù hợp cho cùng modality
    
    **NCC (Normalized Cross Correlation):**
    - Cao hơn = tốt hơn (max = 1.0)
    - Robust với intensity changes
    
    **MI (Mutual Information):**
    - Best cho multi-modal (T1-T2, CT-MRI)
    """
    )
