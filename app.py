"""
Medical Image Processing Web Application

Simple interface for medical image viewing and processing.

Author: HaiSGU
Date: 2025-10-28
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import tempfile
from pathlib import Path
import sys

# Add src to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from utils.file_io import MedicalImageIO

# Page config
st.set_page_config(page_title="Medical Image Processing", layout="wide")

# Session state
if "image_data" not in st.session_state:
    st.session_state.image_data = None
if "metadata" not in st.session_state:
    st.session_state.metadata = {}
if "filename" not in st.session_state:
    st.session_state.filename = None

# Sidebar
with st.sidebar:
    st.title("Medical Image Processing")
    st.markdown("---")

    st.info(
        """
    **Features:**
    - Multi-format support
    - 2D/3D visualization  
    - Metadata extraction
    - Statistics analysis
    
    **Use sidebar → for other tools**
    """
    )

    if st.session_state.image_data is not None:
        st.markdown("---")
        st.subheader("Current File")
        st.text(st.session_state.filename)
        meta = st.session_state.metadata
        st.text(f"Shape: {' × '.join(map(str, meta['shape']))}")
        st.text(f"Type: {meta['dtype']}")

# Main page
st.title("File Upload & Preview")
st.markdown("Upload and view medical images")

# File uploader
uploaded_file = st.file_uploader(
    "Choose a file",
    type=["nii", "gz", "dcm", "nrrd", "mha", "mhd", "npy"],
    help="Supported: NIfTI, DICOM, NRRD, MetaImage, NumPy",
)

if uploaded_file:
    # Save temp
    with tempfile.NamedTemporaryFile(
        delete=False, suffix=Path(uploaded_file.name).suffix
    ) as tmp:
        tmp.write(uploaded_file.getvalue())
        tmp_path = tmp.name

    # Load
    try:
        with st.spinner("Loading..."):
            io_handler = MedicalImageIO()
            image_data, metadata = io_handler.read_image(tmp_path)

        st.session_state.image_data = image_data
        st.session_state.metadata = metadata
        st.session_state.filename = uploaded_file.name

        st.success(f"Loaded: {uploaded_file.name}")

    except Exception as e:
        st.error(f"Error loading file: {str(e)}")
        st.stop()

    # Display info
    st.markdown("---")
    st.subheader("Image Information")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Shape", f"{' × '.join(map(str, metadata['shape']))}")
    col2.metric("Dimensions", f"{metadata['ndim']}D")
    col3.metric("Data Type", metadata["dtype"])
    col4.metric("Size (MB)", f"{image_data.nbytes / 1024 / 1024:.2f}")

    # Statistics
    st.markdown("---")
    st.subheader("Statistics")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Min", f"{image_data.min():.2f}")
    col2.metric("Max", f"{image_data.max():.2f}")
    col3.metric("Mean", f"{image_data.mean():.2f}")
    col4.metric("Std", f"{image_data.std():.2f}")

    # Preview
    st.markdown("---")
    st.subheader("Image Preview")

    # For 3D, use middle slice
    if image_data.ndim == 3:
        slice_idx = st.slider(
            "Slice", 0, image_data.shape[2] - 1, image_data.shape[2] // 2
        )
        slice_data = image_data[:, :, slice_idx]
    else:
        slice_data = image_data

    # Display
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(slice_data, cmap="gray")
    ax.axis("off")
    st.pyplot(fig)
    plt.close()

    # Histogram
    st.markdown("---")
    st.subheader("Intensity Distribution")

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.hist(image_data.flatten(), bins=50, color="steelblue", alpha=0.7)
    ax.set_xlabel("Intensity")
    ax.set_ylabel("Frequency")
    ax.grid(alpha=0.3)
    st.pyplot(fig)
    plt.close()

    # 3D view for 3D images
    if image_data.ndim == 3:
        st.markdown("---")
        st.subheader("3D Visualization")

        with st.spinner("Generating 3D view..."):
            # Subsample for performance
            step = max(1, image_data.shape[0] // 50)
            vol = image_data[::step, ::step, ::step]

            # Threshold
            threshold = np.percentile(vol, 70)

            # Create mesh
            x, y, z = np.where(vol > threshold)

            # Plot
            fig = go.Figure(
                data=[
                    go.Scatter3d(
                        x=x,
                        y=y,
                        z=z,
                        mode="markers",
                        marker=dict(size=2, color=z, colorscale="Viridis"),
                    )
                ]
            )

            fig.update_layout(
                scene=dict(
                    xaxis_title="X", yaxis_title="Y", zaxis_title="Z", aspectmode="data"
                ),
                height=600,
            )

            st.plotly_chart(fig, use_container_width=True)

else:
    st.info("Upload a file to get started")

    with st.expander("Supported Formats"):
        st.markdown(
            """
        - **NIfTI** (.nii, .nii.gz) - Neuroimaging standard
        - **DICOM** (.dcm) - Medical imaging standard  
        - **NRRD** (.nrrd) - Research format
        - **MetaImage** (.mha, .mhd) - ITK format
        - **NumPy** (.npy) - Python arrays
        """
        )
