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

# Page config
st.set_page_config(page_title="🧲 Tái tạo MRI", layout="wide")

# Initialize session state
if "mri_kspace" not in st.session_state:
    st.session_state.mri_kspace = None
if "mri_magnitude" not in st.session_state:
    st.session_state.mri_magnitude = None
if "mri_phase" not in st.session_state:
    st.session_state.mri_phase = None

# Header
st.title("🧲 Tái tạo MRI")
st.markdown("Tái tạo ảnh MRI từ dữ liệu K-space sử dụng FFT")

# Info
with st.expander("About MRI & K-space"):
    col1, col2 = st.columns(2)

    with col1:
        st.markdown(
            """
        **What is K-space?**
        
        K-space is the **frequency domain** representation 
        of MRI data collected by the scanner.
        
        **Properties:**
        - Center: Low frequencies (contrast)
        - Edges: High frequencies (details)
        - Raw data from MRI scanner
        
        **NOT the actual image!**
        Need FFT to convert to image.
        """
        )

    with col2:
        st.markdown(
            """
        **Reconstruction Process:**
        
        1. **Acquire K-space** (scanner)
        2. **Inverse FFT** (2D)
        3. **Extract magnitude** (anatomy)
        4. **Extract phase** (blood flow, etc.)
        
        **Partial Fourier:**
        - Scan only part of K-space
        - 50% faster acquisition
        - Estimate missing data
        """
        )

st.markdown("---")

# Sidebar controls
with st.sidebar:
    st.header("Settings")

    # Data source
    data_source = st.radio(
        "Data Source:",
        ["Generate from Image", "Upload K-space"],
        help="Create K-space from image or upload real data",
    )

    st.markdown("---")

    # Reconstruction options
    if data_source == "Generate from Image":
        partial_fourier = st.checkbox(
            "Partial Fourier", value=False, help="Simulate faster acquisition"
        )

        if partial_fourier:
            pf_percentage = st.select_slider(
                "K-space coverage:",
                options=[50, 62.5, 75, 87.5, 100],
                value=75,
                help="Percentage of K-space to use",
            )

    st.markdown("---")
    st.info("Center of K-space contains most important information")

# Main content
if data_source == "Generate from Image":
    st.subheader("Generate K-space from Image")

    uploaded_file = st.file_uploader(
        "Upload image (.nii, .dcm, .nrrd, .mha, .npy)",
        type=["nii", "gz", "dcm", "nrrd", "mha", "npy"],
        help="Upload medical image to generate K-space",
    )

    if uploaded_file:
        # Load image
        import tempfile

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

            st.success(f"✅ Loaded: {image_2d.shape}")

            # Generate K-space button
            if st.button(
                "🧲 Generate K-space & Reconstruct",
                type="primary",
                use_container_width=True,
            ):

                with st.spinner("Generating K-space..."):
                    reconstructor = MRIReconstructor()

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
                    magnitude = reconstructor.get_magnitude_image(image_complex)
                    phase = reconstructor.get_phase_image(image_complex)

                    st.session_state.mri_magnitude = magnitude
                    st.session_state.mri_phase = phase

                st.success("Reconstruction complete!")

        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
            st.exception(e)

else:  # Upload K-space
    st.subheader("Upload K-space Data")

    uploaded_kspace = st.file_uploader(
        "Choose K-space file (.npy)",
        type=["npy"],
        help="Complex NumPy array (K-space data)",
    )

    if uploaded_kspace:
        try:
            kspace = np.load(io.BytesIO(uploaded_kspace.getvalue()))

            if not np.iscomplexobj(kspace):
                st.warning("K-space should be complex. Converting to complex...")
                kspace = kspace.astype(np.complex128)

            st.session_state.mri_kspace = kspace
            st.success(f"✅ Loaded K-space: {kspace.shape}")

            # Reconstruct button
            if st.button("🧲 Reconstruct", type="primary", use_container_width=True):

                with st.spinner("Reconstructing..."):
                    reconstructor = MRIReconstructor()

                    # Inverse FFT
                    image_complex = reconstructor.kspace_to_image(kspace)

                    # Extract magnitude and phase
                    magnitude = reconstructor.get_magnitude_image(image_complex)
                    phase = reconstructor.get_phase_image(image_complex)

                    st.session_state.mri_magnitude = magnitude
                    st.session_state.mri_phase = phase

                st.success("Reconstruction complete!")

        except Exception as e:
            st.error(f"❌ Error loading K-space: {str(e)}")

# Display results
if st.session_state.mri_kspace is not None:
    st.markdown("---")
    st.header("Results")

    kspace = st.session_state.mri_kspace

    # Show K-space
    st.subheader("K-space (Frequency Domain)")

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
        st.subheader("Reconstructed Images")

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
        st.subheader("Image Statistics")

        col1, col2, col3, col4 = st.columns(4)

        col1.metric("Magnitude Range", f"{magnitude.min():.2f} - {magnitude.max():.2f}")
        col2.metric("Magnitude Mean", f"{magnitude.mean():.2f}")
        col3.metric("Phase Range", f"{phase.min():.2f} - {phase.max():.2f}")
        col4.metric("Phase Mean", f"{phase.mean():.2f}")

        # Download
        st.markdown("---")
        st.subheader("Download")

        col1, col2, col3 = st.columns(3)

        with col1:
            # Download magnitude
            npy_buffer = io.BytesIO()
            np.save(npy_buffer, magnitude)
            npy_bytes = npy_buffer.getvalue()

            st.download_button(
                label="📥 Download Magnitude (.npy)",
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
                label="📥 Download Phase (.npy)",
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
                label="📥 Download Magnitude (.png)",
                data=img_buffer,
                file_name="mri_magnitude.png",
                mime="image/png",
            )

else:
    st.info("👆 Generate K-space or upload data to start")

    st.markdown("---")
    st.subheader("Quick Guide")

    st.markdown(
        """
    **Generate from Image (Demo):**
    1. Upload medical image (NIfTI, DICOM, etc.)
    2. Optionally enable Partial Fourier
    3. Click "Generate K-space & Reconstruct"
    4. View K-space and reconstructed images
    
    **Upload K-space (Real Data):**
    1. Select "Upload K-space"
    2. Upload .npy file (complex array)
    3. Click "Reconstruct"
    4. Download magnitude/phase images
    
    **Understanding Results:**
    - **K-space:** Raw frequency data from MRI scanner
    - **Magnitude:** Anatomical image (what we see)
    - **Phase:** Additional information (blood flow, etc.)
    
    **Partial Fourier:**
    - Simulates faster MRI acquisition
    - 75% = 25% faster scan time
    - 50% = 50% faster (but lower quality)
    """
    )

# Footer
st.markdown("---")
st.caption("💡 Tip: Try Partial Fourier to see trade-off between speed and quality")
