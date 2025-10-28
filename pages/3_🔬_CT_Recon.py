"""
CT Reconstruction Page

Reconstruct CT images from sinograms using FBP and SART algorithms.

Author: HaiSGU
Date: 2025-10-28
"""

import streamlit as st
import tempfile
from pathlib import Path
import sys
import numpy as np
import matplotlib.pyplot as plt
import io

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import from src/ modules
from src.reconstruction.ct_reconstruction import CTReconstructor

# Page config
st.set_page_config(page_title="CT Reconstruction", page_icon="🔬", layout="wide")


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
st.title("🔬 CT Reconstruction")
st.markdown("Reconstruct CT images from projection data (sinogram)")

# Info
with st.expander("ℹ️ About CT Reconstruction"):
    col1, col2 = st.columns(2)

    with col1:
        st.markdown(
            """
        **What is CT Reconstruction?**
        
        CT scanners rotate X-ray source around patient, 
        capturing projections at different angles.
        
        **Sinogram:** Collection of all projections
        - Each row = one angle
        - Contains projection data
        
        **Reconstruction:** Convert sinogram → CT image
        """
        )

    with col2:
        st.markdown(
            """
        **Algorithms:**
        
        **FBP (Filtered Back Projection):**
        - Fast (clinical standard)
        - Various filters available
        - Good for full data
        
        **SART (Iterative):**
        - Slower but better quality
        - Good for sparse data
        - Reduces noise
        """
        )

st.markdown("---")

# Sidebar controls
with st.sidebar:
    st.header("⚙️ Settings")

    # Data source
    data_source = st.radio(
        "Data Source:",
        ["Generate Phantom", "Upload Sinogram"],
        help="Use phantom for demo or upload real data",
    )

    st.markdown("---")

    # Reconstruction method
    method = st.selectbox("Method:", ["FBP", "SART"], help="Reconstruction algorithm")

    if method == "FBP":
        filter_type = st.selectbox(
            "Filter:",
            ["ramp", "shepp-logan", "cosine", "hamming"],
            help="Filter for FBP reconstruction",
        )
    else:
        num_iterations = st.slider(
            "Iterations:",
            min_value=1,
            max_value=50,
            value=10,
            help="Number of SART iterations",
        )

        relaxation = st.slider(
            "Relaxation:",
            min_value=0.1,
            max_value=1.0,
            value=0.5,
            step=0.1,
            help="SART relaxation factor",
        )

    st.markdown("---")
    st.info("💡 Try FBP with 'ramp' filter first for best results")

# Main content
if data_source == "Generate Phantom":
    st.subheader("🎯 Shepp-Logan Phantom")

    col1, col2 = st.columns(2)

    with col1:
        phantom_size = st.slider("Phantom size:", 64, 512, 256, step=64)
        num_angles = st.slider("Number of angles:", 30, 360, 180, step=30)

    with col2:
        st.markdown(
            """
        **Shepp-Logan Phantom:**
        - Standard test image for CT
        - Contains ellipses of different densities
        - Perfect for algorithm testing
        """
        )

    if st.button("🔬 Generate & Reconstruct", type="primary", use_container_width=True):

        with st.spinner("Generating phantom and sinogram..."):
            # Create phantom
            phantom = create_shepp_logan_phantom(phantom_size)
            st.session_state.ct_phantom = phantom

            # Create reconstructor
            reconstructor = CTReconstructor()

            # Generate sinogram
            angles = np.linspace(0, 180, num_angles, endpoint=False)
            sinogram = reconstructor.create_sinogram(phantom, angles)
            st.session_state.ct_sinogram = sinogram

        with st.spinner(f"Reconstructing using {method}..."):
            # Reconstruct
            if method == "FBP":
                reconstructed = reconstructor.fbp_reconstruction(
                    sinogram, filter_name=filter_type
                )
            else:  # SART
                reconstructed = reconstructor.sart_reconstruction(
                    sinogram,
                    image_size=phantom_size,
                    num_iterations=num_iterations,
                    relaxation_factor=relaxation,
                )

            st.session_state.ct_reconstructed = reconstructed

        st.success("✅ Reconstruction complete!")

else:  # Upload Sinogram
    st.subheader("📤 Upload Sinogram")

    uploaded_file = st.file_uploader(
        "Choose sinogram file (.npy)",
        type=["npy"],
        help="NumPy array containing projection data",
    )

    if uploaded_file:
        try:
            sinogram = np.load(io.BytesIO(uploaded_file.getvalue()))
            st.session_state.ct_sinogram = sinogram

            st.success(f"✅ Loaded sinogram: {sinogram.shape}")

            # Reconstruct button
            if st.button("🔬 Reconstruct", type="primary", use_container_width=True):

                with st.spinner(f"Reconstructing using {method}..."):
                    reconstructor = CTReconstructor()

                    if method == "FBP":
                        reconstructed = reconstructor.fbp_reconstruction(
                            sinogram, filter_name=filter_type
                        )
                    else:  # SART
                        image_size = sinogram.shape[1]
                        reconstructed = reconstructor.sart_reconstruction(
                            sinogram,
                            image_size=image_size,
                            num_iterations=num_iterations,
                            relaxation_factor=relaxation,
                        )

                    st.session_state.ct_reconstructed = reconstructed

                st.success("✅ Reconstruction complete!")

        except Exception as e:
            st.error(f"❌ Error loading sinogram: {str(e)}")

# Display results
if st.session_state.ct_sinogram is not None:
    st.markdown("---")
    st.header("📊 Results")

    sinogram = st.session_state.ct_sinogram

    # Show sinogram
    st.subheader("📈 Sinogram (Projection Data)")

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
        st.subheader("🔬 Reconstructed CT Image")

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
            st.subheader("📈 Quality Metrics")

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
            st.subheader("🔍 Comparison: Original vs Reconstructed")

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
        st.subheader("💾 Download")

        col1, col2 = st.columns(2)

        with col1:
            # Download as NumPy
            npy_buffer = io.BytesIO()
            np.save(npy_buffer, reconstructed)
            npy_bytes = npy_buffer.getvalue()

            st.download_button(
                label="📥 Download Image (.npy)",
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
                label="📥 Download Image (.png)",
                data=img_buffer,
                file_name=f"ct_reconstructed_{method.lower()}.png",
                mime="image/png",
            )

else:
    st.info("👆 Generate phantom or upload sinogram to start")

    st.markdown("---")
    st.subheader("📝 Quick Guide")

    st.markdown(
        """
    **Using Phantom (Demo):**
    1. Keep "Generate Phantom" selected
    2. Adjust phantom size and angles
    3. Click "Generate & Reconstruct"
    4. Compare FBP vs SART methods
    
    **Using Real Data:**
    1. Select "Upload Sinogram"
    2. Upload .npy file with sinogram
    3. Choose reconstruction method
    4. Click "Reconstruct"
    
    **Tips:**
    - FBP is faster, SART is better quality
    - More angles = better reconstruction
    - Try different filters for FBP
    - SART needs 10-20 iterations typically
    """
    )

# Footer
st.markdown("---")
st.caption(
    "💡 Tip: Use Shepp-Logan phantom to test different reconstruction parameters"
)
