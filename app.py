"""
Medical Image Processing Web Application

A Streamlit web app for medical image processing including:
- File upload and preview (NIfTI, DICOM, NRRD, MetaImage, NumPy)
- Image information display
- 2D and 3D visualization with slice navigation
- Metadata extraction

Author: HaiSGU
Date: 2025-10-28
Repository: https://github.com/HaiSGU/medical-image-processing
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import tempfile
from pathlib import Path
import sys
import json
from io import BytesIO

# Add src to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Import from src/ modules
from utils.file_io import MedicalImageIO

# ==============================================================================
# PAGE CONFIGURATION
# ==============================================================================

st.set_page_config(
    page_title="Medical Image Processing",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get Help": "https://github.com/HaiSGU/medical-image-processing",
        "Report a bug": "https://github.com/HaiSGU/medical-image-processing/issues",
        "About": """
        # Medical Image Processing System
        
        **Developed by:** HaiSGU
        
        **Features:**
        - File I/O for multiple formats
        - DICOM anonymization
        - Brain segmentation
        - CT/MRI reconstruction
        - Image preprocessing
        
        **Repository:** https://github.com/HaiSGU/medical-image-processing
        """,
    },
)

# ==============================================================================
# CUSTOM CSS
# ==============================================================================

st.markdown(
    """
    <style>
    /* Main container */
    .main {
        padding: 0rem 1rem;
    }
    
    /* Headers */
    h1 {
        color: #1f77b4;
        padding-bottom: 0.5rem;
        border-bottom: 3px solid #1f77b4;
    }
    
    h2 {
        color: #2ca02c;
        padding-top: 1rem;
        margin-top: 1rem;
    }
    
    h3 {
        color: #ff7f0e;
    }
    
    /* Metrics */
    [data-testid="stMetricValue"] {
        font-size: 1.5rem;
        font-weight: bold;
    }
    
    /* Info boxes */
    .stAlert {
        margin-top: 1rem;
        margin-bottom: 1rem;
    }
    
    /* File uploader */
    [data-testid="stFileUploader"] {
        border: 2px dashed #1f77b4;
        border-radius: 10px;
        padding: 1rem;
        background-color: #f0f8ff;
    }
    
    /* Buttons */
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        font-weight: bold;
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #f8f9fa;
    }
    
    /* Expanders */
    .streamlit-expanderHeader {
        font-weight: bold;
        font-size: 1.1rem;
    }
    
    /* Download button */
    .stDownloadButton>button {
        background-color: #28a745;
        color: white;
    }
    
    /* Code blocks */
    .stCodeBlock {
        background-color: #f6f8fa;
        border-radius: 5px;
    }
    </style>
""",
    unsafe_allow_html=True,
)

# ==============================================================================
# SESSION STATE INITIALIZATION
# ==============================================================================


def init_session_state():
    """Initialize session state variables."""
    if "uploaded_file" not in st.session_state:
        st.session_state.uploaded_file = None
    if "image_data" not in st.session_state:
        st.session_state.image_data = None
    if "metadata" not in st.session_state:
        st.session_state.metadata = {}
    if "file_path" not in st.session_state:
        st.session_state.file_path = None


init_session_state()

# ==============================================================================
# SIDEBAR
# ==============================================================================

with st.sidebar:
    st.title("🏥 Medical Imaging")
    st.markdown("---")

    # Navigation (for future pages)
    st.subheader("📚 Navigation")
    page = st.radio(
        "Select Page:",
        [
            "📁 File Upload",
            "🔒 Anonymization (Coming Soon)",
            "🧠 Segmentation (Coming Soon)",
            "🔬 CT Recon (Coming Soon)",
            "🧲 MRI Recon (Coming Soon)",
            "⚙️ Preprocessing (Coming Soon)",
        ],
        index=0,
    )

    st.markdown("---")

    # About section
    st.subheader("ℹ️ About")
    st.info(
        """
    **Medical Image Processing**
    
    Upload and analyze medical images in multiple formats.
    
    **Supported Formats:**
    - NIfTI (.nii, .nii.gz)
    - DICOM (.dcm)
    - NRRD (.nrrd)
    - MetaImage (.mha, .mhd)
    - NumPy (.npy)
    
    **Features:**
    - Interactive viewer
    - Metadata extraction
    - Slice navigation
    - Statistics
    
    ---
    
    **Developed by:** HaiSGU
    
    **GitHub:** [medical-image-processing](https://github.com/HaiSGU/medical-image-processing)
    """
    )

    st.markdown("---")

    # File info in sidebar (if file loaded)
    if st.session_state.image_data is not None:
        st.subheader("📄 Current File")
        st.success(f"**{st.session_state.uploaded_file}**")

        meta = st.session_state.metadata
        st.markdown(
            f"""
        - **Shape:** {' × '.join(map(str, meta['shape']))}
        - **Type:** {meta['dtype']}
        - **Dimension:** {meta['ndim']}D
        """
        )

# ==============================================================================
# MAIN PAGE - FILE UPLOAD
# ==============================================================================

if page == "📁 File Upload":
    st.title("📁 File Upload & Preview")
    st.markdown("Upload medical images and visualize their properties.")

    # Supported formats expandable info
    with st.expander("ℹ️ Supported File Formats", expanded=False):
        st.markdown(
            """
        | Format | Extensions | Description | Common Use |
        |--------|------------|-------------|------------|
        | **NIfTI** | `.nii`, `.nii.gz` | Neuroimaging Informatics Technology Initiative | MRI, fMRI, DTI |
        | **DICOM** | `.dcm` | Digital Imaging and Communications in Medicine | CT, MRI, X-Ray |
        | **NRRD** | `.nrrd` | Nearly Raw Raster Data | Research, 3D Slicer |
        | **MetaImage** | `.mha`, `.mhd` | ITK MetaImage Format | ITK, VTK applications |
        | **NumPy** | `.npy` | NumPy Array Format | Python data exchange |
        
        **💡 Tips:**
        - NIfTI is most common for brain imaging
        - DICOM contains patient metadata
        - NRRD good for research data
        - Compressed formats (.nii.gz) save space
        """
        )

    st.markdown("---")

    # ==============================================================================
    # FILE UPLOADER
    # ==============================================================================

    uploaded_file = st.file_uploader(
        "📤 Choose a medical image file",
        type=["nii", "gz", "dcm", "nrrd", "mha", "mhd", "npy"],
        help="Upload NIfTI, DICOM, NRRD, MetaImage, or NumPy files",
        accept_multiple_files=False,
    )

    if uploaded_file is not None:
        # Save to temp file
        with tempfile.NamedTemporaryFile(
            delete=False, suffix=Path(uploaded_file.name).suffix
        ) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_path = tmp_file.name

        # Success message
        file_size_kb = uploaded_file.size / 1024
        file_size_mb = file_size_kb / 1024

        if file_size_mb >= 1:
            size_str = f"{file_size_mb:.2f} MB"
        else:
            size_str = f"{file_size_kb:.2f} KB"

        st.success(f"✅ **Uploaded:** {uploaded_file.name} ({size_str})")

        # Load image
        try:
            with st.spinner("🔄 Loading image..."):
                io = MedicalImageIO()
                image_data, metadata = io.read_image(tmp_path)

            # Store in session state
            st.session_state.uploaded_file = uploaded_file.name
            st.session_state.image_data = image_data
            st.session_state.metadata = metadata
            st.session_state.file_path = tmp_path

            st.success("✅ Image loaded successfully!")

        except Exception as e:
            st.error(f"❌ **Error loading image:** {str(e)}")
            st.exception(e)
            st.stop()

    # ==============================================================================
    # DISPLAY IMAGE INFORMATION
    # ==============================================================================

    if st.session_state.image_data is not None:
        image_data = st.session_state.image_data
        metadata = st.session_state.metadata

        st.markdown("---")
        st.header("📊 Image Information")

        # Metrics in columns
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric(
                label="Dimensions",
                value=f"{metadata['ndim']}D",
                help="Number of dimensions (2D or 3D)",
            )

        with col2:
            shape_str = " × ".join(map(str, metadata["shape"]))
            st.metric(
                label="Shape", value=shape_str, help="Image dimensions in pixels/voxels"
            )

        with col3:
            st.metric(
                label="Data Type", value=metadata["dtype"], help="Pixel data type"
            )

        with col4:
            file_format = metadata.get("format", "Unknown")
            st.metric(label="Format", value=file_format, help="File format detected")

        # Value range
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                label="Min Value",
                value=f"{metadata['min_value']:.2f}",
                help="Minimum pixel/voxel value",
            )

        with col2:
            st.metric(
                label="Max Value",
                value=f"{metadata['max_value']:.2f}",
                help="Maximum pixel/voxel value",
            )

        with col3:
            value_range = metadata["max_value"] - metadata["min_value"]
            st.metric(
                label="Value Range",
                value=f"{value_range:.2f}",
                help="Difference between max and min",
            )

        # Additional metadata
        st.markdown("---")
        st.subheader("🔍 Additional Metadata")

        col1, col2 = st.columns(2)

        with col1:
            if metadata.get("spacing"):
                spacing_str = " × ".join([f"{s:.3f}" for s in metadata["spacing"]])
                st.info(f"**Spacing (mm):** {spacing_str}")
            else:
                st.warning("**Spacing:** Not available")

            if metadata.get("origin"):
                origin_str = " × ".join([f"{o:.2f}" for o in metadata["origin"]])
                st.info(f"**Origin (mm):** {origin_str}")
            else:
                st.warning("**Origin:** Not available")

        with col2:
            # DICOM-specific
            if metadata.get("modality"):
                st.info(f"**Modality:** {metadata['modality']}")

            if metadata.get("patient_name"):
                st.warning(f"⚠️ **Patient Name:** {metadata['patient_name']}")
                st.caption("⚠️ Consider using DICOM Anonymization!")

            if metadata.get("study_date"):
                st.info(f"**Study Date:** {metadata['study_date']}")

        # ==============================================================================
        # IMAGE STATISTICS
        # ==============================================================================

        st.markdown("---")
        st.header("📈 Image Statistics")

        col1, col2, col3, col4, col5 = st.columns(5)

        mean_val = np.mean(image_data)
        std_val = np.std(image_data)
        median_val = np.median(image_data)
        nonzero_pct = (np.count_nonzero(image_data) / image_data.size) * 100
        total_voxels = image_data.size

        with col1:
            st.metric("Mean", f"{mean_val:.2f}")

        with col2:
            st.metric("Std Dev", f"{std_val:.2f}")

        with col3:
            st.metric("Median", f"{median_val:.2f}")

        with col4:
            st.metric("Non-zero", f"{nonzero_pct:.1f}%")

        with col5:
            if total_voxels >= 1e6:
                voxels_str = f"{total_voxels/1e6:.2f}M"
            else:
                voxels_str = f"{total_voxels/1e3:.1f}K"
            st.metric("Total Voxels", voxels_str)

        # ==============================================================================
        # IMAGE PREVIEW
        # ==============================================================================

        st.markdown("---")
        st.header("🖼️ Image Preview")

        if metadata["ndim"] == 2:
            # ========== 2D IMAGE ==========
            st.markdown("**2D Image Display**")

            # Display options
            col1, col2 = st.columns([3, 1])

            with col2:
                colormap = st.selectbox(
                    "Colormap",
                    options=["gray", "viridis", "plasma", "hot", "jet", "bone"],
                    index=0,
                    help="Color scheme for visualization",
                )

                show_colorbar = st.checkbox("Show Colorbar", value=True)

            # Plot 2D image
            fig, ax = plt.subplots(figsize=(10, 10))
            im = ax.imshow(image_data, cmap=colormap)
            ax.axis("off")
            ax.set_title(
                f"2D Image: {st.session_state.uploaded_file}",
                fontsize=14,
                fontweight="bold",
                pad=20,
            )

            if show_colorbar:
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

            st.pyplot(fig)
            plt.close()

        elif metadata["ndim"] == 3:
            # ========== 3D IMAGE ==========
            st.markdown("**3D Image Slice Viewer**")

            # Controls
            col1, col2, col3 = st.columns([2, 2, 1])

            with col1:
                axis = st.radio(
                    "View Axis:",
                    options=["Axial (Z)", "Coronal (Y)", "Sagittal (X)"],
                    horizontal=True,
                    help="Select which anatomical plane to view",
                )

            with col2:
                colormap = st.selectbox(
                    "Colormap:",
                    options=["gray", "viridis", "plasma", "hot", "jet", "bone"],
                    index=0,
                )

            with col3:
                show_colorbar = st.checkbox("Colorbar", value=True)
                show_grid = st.checkbox("Grid", value=False)

            # Slice selection based on axis
            if axis == "Axial (Z)":
                max_slice = image_data.shape[2] - 1
                default_slice = max_slice // 2
                slice_idx = st.slider(
                    "Slice Index",
                    min_value=0,
                    max_value=max_slice,
                    value=default_slice,
                    help=f"Navigate through {max_slice + 1} axial slices",
                )
                slice_data = image_data[:, :, slice_idx]
                slice_title = f"Axial Slice {slice_idx}/{max_slice}"

            elif axis == "Coronal (Y)":
                max_slice = image_data.shape[1] - 1
                default_slice = max_slice // 2
                slice_idx = st.slider(
                    "Slice Index",
                    min_value=0,
                    max_value=max_slice,
                    value=default_slice,
                    help=f"Navigate through {max_slice + 1} coronal slices",
                )
                slice_data = image_data[:, slice_idx, :]
                slice_title = f"Coronal Slice {slice_idx}/{max_slice}"

            else:  # Sagittal (X)
                max_slice = image_data.shape[0] - 1
                default_slice = max_slice // 2
                slice_idx = st.slider(
                    "Slice Index",
                    min_value=0,
                    max_value=max_slice,
                    value=default_slice,
                    help=f"Navigate through {max_slice + 1} sagittal slices",
                )
                slice_data = image_data[slice_idx, :, :]
                slice_title = f"Sagittal Slice {slice_idx}/{max_slice}"

            # Display slice
            fig, ax = plt.subplots(figsize=(12, 10))
            im = ax.imshow(slice_data.T, cmap=colormap, origin="lower")
            ax.axis("off")
            ax.set_title(
                f"{slice_title}\n{st.session_state.uploaded_file}",
                fontsize=14,
                fontweight="bold",
                pad=20,
            )

            if show_colorbar:
                cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                cbar.set_label("Intensity", rotation=270, labelpad=20)

            if show_grid:
                ax.grid(True, alpha=0.3, color="white", linestyle="--")

            st.pyplot(fig)
            plt.close()

            # Slice info
            st.caption(
                f"**Slice shape:** {slice_data.shape[0]} × {slice_data.shape[1]} | "
                f"**Min:** {slice_data.min():.2f} | "
                f"**Max:** {slice_data.max():.2f} | "
                f"**Mean:** {slice_data.mean():.2f}"
            )

        else:
            st.error(f"❌ Unsupported dimension: {metadata['ndim']}D")

        # ==============================================================================
        # HISTOGRAM
        # ==============================================================================

        with st.expander("📊 Intensity Histogram", expanded=False):
            st.markdown("Distribution of pixel/voxel intensities")

            col1, col2 = st.columns([3, 1])

            with col2:
                bins = st.slider("Number of Bins", 20, 200, 50, step=10)
                log_scale = st.checkbox("Log Scale Y-axis", value=False)

            with col1:
                fig, ax = plt.subplots(figsize=(12, 4))

                # Flatten image data
                flat_data = image_data.flatten()

                # Plot histogram
                ax.hist(
                    flat_data,
                    bins=bins,
                    color="steelblue",
                    alpha=0.7,
                    edgecolor="black",
                )

                ax.set_xlabel("Intensity Value", fontsize=12)
                ax.set_ylabel("Frequency", fontsize=12)
                ax.set_title("Intensity Distribution", fontsize=14, fontweight="bold")
                ax.grid(True, alpha=0.3)

                if log_scale:
                    ax.set_yscale("log")

                # Add statistics lines
                ax.axvline(
                    mean_val,
                    color="red",
                    linestyle="--",
                    linewidth=2,
                    label=f"Mean: {mean_val:.2f}",
                )
                ax.axvline(
                    median_val,
                    color="green",
                    linestyle="--",
                    linewidth=2,
                    label=f"Median: {median_val:.2f}",
                )
                ax.legend()

                st.pyplot(fig)
                plt.close()

        # ==============================================================================
        # 3D VOLUME RENDERING (OPTIONAL)
        # ==============================================================================

        if metadata["ndim"] == 3:
            with st.expander("🎲 3D Volume Preview (Beta)", expanded=False):
                st.markdown("Interactive 3D visualization using Plotly")

                # Subsample for performance
                subsample = st.slider(
                    "Subsample Factor",
                    1,
                    10,
                    4,
                    help="Higher = faster but lower quality",
                )

                if st.button("Generate 3D View"):
                    with st.spinner("Generating 3D visualization..."):
                        # Subsample data
                        vol_data = image_data[::subsample, ::subsample, ::subsample]

                        # Create meshgrid
                        X, Y, Z = np.mgrid[
                            0 : vol_data.shape[0],
                            0 : vol_data.shape[1],
                            0 : vol_data.shape[2],
                        ]

                        # Flatten
                        values = vol_data.flatten()

                        # Filter to only show significant voxels
                        threshold = np.percentile(values, 70)
                        mask = values > threshold

                        # Create 3D scatter
                        fig = go.Figure(
                            data=go.Scatter3d(
                                x=X.flatten()[mask],
                                y=Y.flatten()[mask],
                                z=Z.flatten()[mask],
                                mode="markers",
                                marker=dict(
                                    size=2,
                                    color=values[mask],
                                    colorscale="Viridis",
                                    showscale=True,
                                    opacity=0.6,
                                ),
                            )
                        )

                        fig.update_layout(
                            title="3D Volume Rendering",
                            scene=dict(
                                xaxis_title="X", yaxis_title="Y", zaxis_title="Z"
                            ),
                            width=800,
                            height=600,
                        )

                        st.plotly_chart(fig, use_container_width=True)

        # ==============================================================================
        # DOWNLOAD & EXPORT
        # ==============================================================================

        st.markdown("---")
        st.header("💾 Download & Export")

        col1, col2, col3 = st.columns(3)

        with col1:
            # Export as NumPy
            st.markdown("**NumPy Array (.npy)**")
            npy_buffer = BytesIO()
            np.save(npy_buffer, image_data)
            npy_bytes = npy_buffer.getvalue()

            st.download_button(
                label="📥 Download as .npy",
                data=npy_bytes,
                file_name=f"{Path(st.session_state.uploaded_file).stem}.npy",
                mime="application/octet-stream",
                help="NumPy array format for Python",
            )

        with col2:
            # Export metadata as JSON
            st.markdown("**Metadata (.json)**")

            # Convert metadata to JSON-serializable
            json_metadata = {}
            for k, v in metadata.items():
                if isinstance(v, (list, tuple)):
                    json_metadata[k] = list(v)
                elif isinstance(v, np.ndarray):
                    json_metadata[k] = v.tolist()
                elif isinstance(v, (np.integer, np.floating)):
                    json_metadata[k] = float(v)
                else:
                    json_metadata[k] = str(v)

            json_str = json.dumps(json_metadata, indent=2)

            st.download_button(
                label="📥 Download Metadata",
                data=json_str,
                file_name=f"{Path(st.session_state.uploaded_file).stem}_metadata.json",
                mime="application/json",
                help="Image metadata in JSON format",
            )

        with col3:
            # Export statistics as CSV
            st.markdown("**Statistics (.csv)**")

            stats_str = f"""Statistic,Value
Filename,{st.session_state.uploaded_file}
Dimensions,{metadata['ndim']}D
Shape,"{' × '.join(map(str, metadata['shape']))}"
Data Type,{metadata['dtype']}
Min Value,{metadata['min_value']}
Max Value,{metadata['max_value']}
Mean,{mean_val}
Std Dev,{std_val}
Median,{median_val}
Non-zero %,{nonzero_pct}
Total Voxels,{total_voxels}
"""

            st.download_button(
                label="📥 Download Statistics",
                data=stats_str,
                file_name=f"{Path(st.session_state.uploaded_file).stem}_stats.csv",
                mime="text/csv",
                help="Image statistics in CSV format",
            )

    else:
        # No file uploaded yet
        st.info("👆 **Please upload a medical image file to get started**")

        # Show example
        st.markdown("---")
        st.subheader("📝 Example Usage in Python")

        st.code(
            """
# Using the Medical Image Processing library:

from utils.file_io import MedicalImageIO

# Create I/O handler
io = MedicalImageIO()

# Read image
image_data, metadata = io.read_image('brain_mri.nii.gz')

# Print information
print(f"Shape: {image_data.shape}")
print(f"Spacing: {metadata['spacing']} mm")
print(f"Data type: {metadata['dtype']}")
print(f"Value range: [{metadata['min_value']}, {metadata['max_value']}]")

# Access specific slice (for 3D)
slice_data = image_data[:, :, 100]  # Axial slice 100

# Save as different format
io.save_image('brain_mri.nrrd', image_data, metadata)
        """,
            language="python",
        )

        st.markdown("---")
        st.subheader("🎓 Quick Guide")

        st.markdown(
            """
        **How to use this app:**
        
        1. **Upload** your medical image file using the file uploader above
        2. **View** automatic analysis of image properties and metadata
        3. **Navigate** through 3D images using the slice viewer
        4. **Analyze** intensity distribution with interactive histogram
        5. **Download** processed data and statistics
        
        **Next steps:**
        - Try the **Anonymization** page to remove patient information from DICOM files
        - Use **Segmentation** to extract brain regions
        - Explore **Reconstruction** for CT/MRI image reconstruction
        - Apply **Preprocessing** to enhance images
        
        **Need help?**
        - Check the [GitHub repository](https://github.com/HaiSGU/medical-image-processing)
        - Read the documentation
        - Report issues on GitHub
        """
        )

# ==============================================================================
# OTHER PAGES (PLACEHOLDERS)
# ==============================================================================

else:
    st.title(page)
    st.info("🚧 This page is under construction. Coming soon!")

    st.markdown(
        """
    **Planned features for this page:**
    - Interactive tools
    - Real-time processing
    - Results visualization
    - Export functionality
    
    Stay tuned! 🚀
    """
    )

# ==============================================================================
# FOOTER
# ==============================================================================

st.markdown("---")
st.markdown(
    """
<div style='text-align: center; color: #666; padding: 2rem 0;'>
    <p><strong>🏥 Medical Image Processing System</strong></p>
    <p>Developed by <strong>HaiSGU</strong> | 2025</p>
    <p>
        <a href='https://github.com/HaiSGU/medical-image-processing' target='_blank' 
           style='margin: 0 10px; text-decoration: none;'>📦 GitHub</a> |
        <a href='https://github.com/HaiSGU/medical-image-processing/issues' target='_blank'
           style='margin: 0 10px; text-decoration: none;'>🐛 Report Issue</a> |
        <a href='https://github.com/HaiSGU/medical-image-processing#readme' target='_blank'
           style='margin: 0 10px; text-decoration: none;'>📖 Documentation</a>
    </p>
</div>
""",
    unsafe_allow_html=True,
)
