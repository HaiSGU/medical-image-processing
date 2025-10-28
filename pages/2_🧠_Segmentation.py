"""
Brain Segmentation Page

Segment brain regions from medical images using multiple methods.

Author: HaiSGU
Date: 2025-10-28
"""

import streamlit as st
import tempfile
from pathlib import Path
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import io

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import from src/ modules
from src.segmentation.brain_segmentation import BrainSegmentation
from utils.file_io import MedicalImageIO

# Page config
st.set_page_config(page_title="Brain Segmentation", page_icon="🧠", layout="wide")

# Initialize session state
if "seg_image_data" not in st.session_state:
    st.session_state.seg_image_data = None
if "seg_mask" not in st.session_state:
    st.session_state.seg_mask = None
if "seg_metadata" not in st.session_state:
    st.session_state.seg_metadata = {}

# Header
st.title("🧠 Brain Segmentation")
st.markdown("Extract brain regions from medical images")

# Info
with st.expander("ℹ️ Segmentation Methods"):
    col1, col2 = st.columns(2)

    with col1:
        st.markdown(
            """
        **Threshold-based:**
        - Simple intensity thresholding
        - Fast and straightforward
        - Good for high-contrast images
        
        **Otsu Method:**
        - Automatic threshold selection
        - No manual parameters needed
        - Works well for bimodal histograms
        """
        )

    with col2:
        st.markdown(
            """
        **Region Growing:**
        - Grows from seed point
        - More precise boundaries
        - Requires seed selection
        
        **Automatic:**
        - Combines multiple methods
        - Best overall results
        - Recommended for beginners
        """
        )

st.markdown("---")

# Sidebar controls
with st.sidebar:
    st.header("⚙️ Segmentation Settings")

    method = st.selectbox(
        "Method",
        ["Automatic", "Threshold", "Otsu", "Region Growing"],
        help="Choose segmentation method",
    )

    st.markdown("---")

    # Method-specific parameters
    if method == "Threshold":
        threshold = st.slider(
            "Threshold Value",
            min_value=0,
            max_value=255,
            value=50,
            help="Pixels above this value will be kept",
        )

    elif method == "Region Growing":
        st.markdown("**Seed Point (%):**")
        seed_x = st.slider("X position", 0, 100, 50)
        seed_y = st.slider("Y position", 0, 100, 50)
        seed_z = st.slider("Z position", 0, 100, 50)

        intensity_tolerance = st.slider(
            "Intensity Tolerance",
            min_value=1,
            max_value=50,
            value=10,
            help="Max difference from seed intensity",
        )

    # Morphological operations
    st.markdown("---")
    st.markdown("**Post-processing:**")

    apply_morph = st.checkbox("Apply morphological ops", value=True)

    if apply_morph:
        morph_op = st.selectbox(
            "Operation",
            ["closing", "opening", "dilation", "erosion"],
            help="Morphological operation to clean up mask",
        )

        kernel_size = st.slider("Kernel Size", 1, 10, 3)

    keep_largest = st.checkbox(
        "Keep largest component only",
        value=True,
        help="Remove small disconnected regions",
    )

    st.markdown("---")
    st.info("💡 Try 'Automatic' method first for best results")

# File upload
st.subheader("📤 Upload Medical Image")

uploaded_file = st.file_uploader(
    "Choose a file (.nii, .nii.gz, .dcm, .nrrd, .mha, .npy)",
    type=["nii", "gz", "dcm", "nrrd", "mha", "npy"],
    help="Upload brain scan image",
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

            st.session_state.seg_image_data = image_data
            st.session_state.seg_metadata = metadata

        st.success(f"✅ Loaded: {uploaded_file.name}")

        # Show image info
        col1, col2, col3 = st.columns(3)
        col1.metric("Shape", f"{' × '.join(map(str, metadata['shape']))}")
        col2.metric("Data Type", metadata["dtype"])
        col3.metric("Dimension", f"{metadata['ndim']}D")

    except Exception as e:
        st.error(f"❌ Error loading image: {str(e)}")
        st.stop()

    st.markdown("---")

    # Segmentation button
    if st.button("🧠 Segment Brain", type="primary", use_container_width=True):

        with st.spinner("Segmenting..."):
            try:
                # Create segmenter
                segmenter = BrainSegmentation()

                # Run segmentation based on method
                if method == "Automatic":
                    mask = segmenter.segment_brain(image_data)

                elif method == "Threshold":
                    mask = segmenter.threshold_segmentation(
                        image_data, threshold=threshold
                    )

                elif method == "Otsu":
                    mask = segmenter.otsu_segmentation(image_data)

                elif method == "Region Growing":
                    # Convert percentage to actual coordinates
                    shape = image_data.shape
                    seed = [
                        int(seed_x * shape[0] / 100),
                        int(seed_y * shape[1] / 100),
                        int(seed_z * shape[2] / 100) if len(shape) > 2 else 0,
                    ]

                    # For 2D images
                    if len(shape) == 2:
                        seed = seed[:2]

                    mask = segmenter.region_growing_segmentation(
                        image_data,
                        seed_point=tuple(seed),
                        intensity_tolerance=intensity_tolerance,
                    )

                # Apply morphological operations
                if apply_morph:
                    if morph_op == "closing":
                        mask = segmenter.morphological_closing(
                            mask, kernel_size=kernel_size
                        )
                    elif morph_op == "opening":
                        mask = segmenter.morphological_opening(
                            mask, kernel_size=kernel_size
                        )
                    elif morph_op == "dilation":
                        mask = segmenter.morphological_dilation(
                            mask, kernel_size=kernel_size
                        )
                    elif morph_op == "erosion":
                        mask = segmenter.morphological_erosion(
                            mask, kernel_size=kernel_size
                        )

                # Keep largest component
                if keep_largest:
                    mask = segmenter.keep_largest_component(mask)

                # Store in session state
                st.session_state.seg_mask = mask

                st.success("✅ Segmentation complete!")

            except Exception as e:
                st.error(f"❌ Segmentation failed: {str(e)}")
                st.exception(e)
                st.stop()

    # Display results
    if st.session_state.seg_mask is not None:
        st.markdown("---")
        st.header("📊 Segmentation Results")

        image_data = st.session_state.seg_image_data
        mask = st.session_state.seg_mask

        # Statistics
        col1, col2, col3, col4 = st.columns(4)

        total_voxels = mask.size
        segmented_voxels = np.sum(mask > 0)
        percentage = (segmented_voxels / total_voxels) * 100

        col1.metric("Total Voxels", f"{total_voxels:,}")
        col2.metric("Segmented", f"{segmented_voxels:,}")
        col3.metric("Percentage", f"{percentage:.1f}%")
        col4.metric("Background", f"{total_voxels - segmented_voxels:,}")

        st.markdown("---")

        # Visualization
        st.subheader("🖼️ Visualization")

        # View controls
        if image_data.ndim == 3:
            col1, col2 = st.columns([3, 1])

            with col1:
                view_mode = st.radio(
                    "View:", ["Original", "Mask", "Overlay"], horizontal=True
                )

            with col2:
                opacity = st.slider("Overlay opacity", 0.0, 1.0, 0.5)

            # Slice navigation
            axis = st.radio(
                "Axis:", ["Axial (Z)", "Coronal (Y)", "Sagittal (X)"], horizontal=True
            )

            if axis == "Axial (Z)":
                max_slice = image_data.shape[2] - 1
                slice_idx = st.slider("Slice", 0, max_slice, max_slice // 2)
                img_slice = image_data[:, :, slice_idx]
                mask_slice = mask[:, :, slice_idx]

            elif axis == "Coronal (Y)":
                max_slice = image_data.shape[1] - 1
                slice_idx = st.slider("Slice", 0, max_slice, max_slice // 2)
                img_slice = image_data[:, slice_idx, :]
                mask_slice = mask[:, slice_idx, :]

            else:  # Sagittal
                max_slice = image_data.shape[0] - 1
                slice_idx = st.slider("Slice", 0, max_slice, max_slice // 2)
                img_slice = image_data[slice_idx, :, :]
                mask_slice = mask[slice_idx, :, :]

        else:  # 2D image
            view_mode = st.radio(
                "View:", ["Original", "Mask", "Overlay"], horizontal=True
            )
            opacity = st.slider("Overlay opacity", 0.0, 1.0, 0.5)

            img_slice = image_data
            mask_slice = mask

        # Plot
        fig, ax = plt.subplots(figsize=(10, 10))

        if view_mode == "Original":
            ax.imshow(img_slice.T, cmap="gray", origin="lower")
            ax.set_title("Original Image", fontsize=14, fontweight="bold")

        elif view_mode == "Mask":
            ax.imshow(mask_slice.T, cmap="hot", origin="lower")
            ax.set_title("Segmentation Mask", fontsize=14, fontweight="bold")

        else:  # Overlay
            ax.imshow(img_slice.T, cmap="gray", origin="lower")

            # Create transparent colormap for mask
            colors = [(0, 0, 0, 0), (1, 0, 0, opacity)]
            n_bins = 2
            cmap = ListedColormap(colors)

            ax.imshow(mask_slice.T, cmap=cmap, origin="lower", alpha=opacity)
            ax.set_title("Overlay (Red = Segmented)", fontsize=14, fontweight="bold")

        ax.axis("off")
        st.pyplot(fig)
        plt.close()

        # Download options
        st.markdown("---")
        st.subheader("💾 Download Results")

        col1, col2 = st.columns(2)

        with col1:
            # Download mask as NumPy
            npy_buffer = io.BytesIO()
            np.save(npy_buffer, mask)
            npy_bytes = npy_buffer.getvalue()

            st.download_button(
                label="📥 Download Mask (.npy)",
                data=npy_bytes,
                file_name="segmentation_mask.npy",
                mime="application/octet-stream",
            )

        with col2:
            # Download overlay image
            fig_download, ax_download = plt.subplots(figsize=(10, 10))

            if image_data.ndim == 3:
                mid_slice = image_data.shape[2] // 2
                img_slice = image_data[:, :, mid_slice]
                mask_slice = mask[:, :, mid_slice]
            else:
                img_slice = image_data
                mask_slice = mask

            ax_download.imshow(img_slice.T, cmap="gray", origin="lower")

            colors = [(0, 0, 0, 0), (1, 0, 0, 0.5)]
            cmap = ListedColormap(colors)
            ax_download.imshow(mask_slice.T, cmap=cmap, origin="lower", alpha=0.5)
            ax_download.axis("off")

            img_buffer = io.BytesIO()
            fig_download.savefig(img_buffer, format="png", bbox_inches="tight", dpi=150)
            img_buffer.seek(0)
            plt.close(fig_download)

            st.download_button(
                label="📥 Download Overlay (.png)",
                data=img_buffer,
                file_name="segmentation_overlay.png",
                mime="image/png",
            )

else:
    st.info("👆 Upload a brain image to start segmentation")

    st.markdown("---")
    st.subheader("📝 Quick Guide")

    st.markdown(
        """
    **Steps:**
    1. Upload brain scan (NIfTI, DICOM, etc.)
    2. Choose segmentation method
    3. Adjust parameters (optional)
    4. Click "Segment Brain"
    5. View and download results
    
    **Recommended Settings:**
    - **Method:** Start with "Automatic"
    - **Post-processing:** Enable morphological closing
    - **Keep largest:** Always enabled
    
    **Tips:**
    - Use Otsu for automatic threshold selection
    - Region Growing works best with clear boundaries
    - Try different views (Axial/Coronal/Sagittal) for 3D images
    """
    )

# Footer
st.markdown("---")
st.caption("💡 Tip: Try different methods and compare results for best accuracy")
