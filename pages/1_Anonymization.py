"""
DICOM Anonymization Page

Remove Protected Health Information (PHI) from DICOM files.

Author: HaiSGU
Date: 2025-10-28
"""

import streamlit as st
import tempfile
from pathlib import Path
import sys
import zipfile
import io
import pandas as pd

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import from src/ modules
from src.anonymization.dicom_anonymizer import DICOMAnonymizer
import pydicom

# Page config
st.set_page_config(page_title="DICOM Anonymization", layout="wide")

# Header
st.title("DICOM Anonymization")
st.markdown("Remove patient information from DICOM files")

# Info box
with st.expander("What gets removed?"):
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(
            """
        **Patient Info:**
        - Patient Name, ID
        - Birth Date, Age, Sex
        - Address, Phone
        """
        )
    with col2:
        st.markdown(
            """
        **Study Info:**
        - Study Date/Time
        - Institution Name
        - Physician Name
        """
        )

st.markdown("---")

# Sidebar settings
with st.sidebar:
    st.header("Settings")

    patient_prefix = st.text_input(
        "Anonymous ID Prefix", value="ANON", help="Prefix for new patient IDs"
    )

    st.info("Uploaded files will be anonymized and available for download as ZIP")

# File upload
st.subheader("Upload DICOM Files")

uploaded_files = st.file_uploader(
    "Choose DICOM files (.dcm)",
    type=["dcm"],
    accept_multiple_files=True,
    help="Select one or more DICOM files",
)

if uploaded_files:
    st.success(f"✅ Uploaded {len(uploaded_files)} file(s)")

    # Show preview of first file
    st.subheader("Preview: Original Metadata")

    try:
        # Read first file
        first_file = uploaded_files[0]
        dicom_data = pydicom.dcmread(io.BytesIO(first_file.getvalue()))

        # Show important tags
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("**Patient:**")
            st.text(f"Name: {dicom_data.get('PatientName', 'N/A')}")
            st.text(f"ID: {dicom_data.get('PatientID', 'N/A')}")
            st.text(f"Birth: {dicom_data.get('PatientBirthDate', 'N/A')}")

        with col2:
            st.markdown("**Study:**")
            st.text(f"Date: {dicom_data.get('StudyDate', 'N/A')}")
            st.text(f"Time: {dicom_data.get('StudyTime', 'N/A')}")
            st.text(f"Modality: {dicom_data.get('Modality', 'N/A')}")

        with col3:
            st.markdown("**Institution:**")
            st.text(f"Name: {dicom_data.get('InstitutionName', 'N/A')}")
            st.text(f"Station: {dicom_data.get('StationName', 'N/A')}")

    except Exception as e:
        st.warning(f"Could not read metadata: {str(e)}")

    st.markdown("---")

    # Anonymize button
    if st.button("🔒 Anonymize Files", type="primary", use_container_width=True):

        with st.spinner("Processing..."):
            try:
                # Create temp directories
                with tempfile.TemporaryDirectory() as tmp_dir:
                    tmp_path = Path(tmp_dir)
                    input_dir = tmp_path / "input"
                    output_dir = tmp_path / "output"
                    input_dir.mkdir()
                    output_dir.mkdir()

                    # Progress bar
                    progress_bar = st.progress(0)

                    # Save uploaded files
                    for idx, uploaded_file in enumerate(uploaded_files):
                        file_path = input_dir / uploaded_file.name
                        file_path.write_bytes(uploaded_file.getvalue())
                        progress_bar.progress((idx + 1) / (len(uploaded_files) * 2))

                    # Anonymize
                    anonymizer = DICOMAnonymizer(patient_id_prefix=patient_prefix)
                    results = anonymizer.anonymize_directory(
                        str(input_dir), str(output_dir)
                    )

                    progress_bar.progress(1.0)

                    # Show results
                    st.success(
                        f"""
                    ✅ **Anonymization Complete!**
                    
                    - Success: {results['successful']} files
                    - Failed: {results['failed']} files
                    - Patients: {len(results['id_mapping'])}
                    """
                    )

                    # ID Mapping Table
                    if results["id_mapping"]:
                        st.subheader("🔑 ID Mapping")

                        df_mapping = pd.DataFrame(
                            {
                                "Original ID": list(results["id_mapping"].keys()),
                                "Anonymous ID": list(results["id_mapping"].values()),
                            }
                        )

                        st.dataframe(df_mapping, use_container_width=True)

                        # Download mapping CSV
                        csv = df_mapping.to_csv(index=False)
                        st.download_button(
                            label="📥 Download ID Mapping (CSV)",
                            data=csv,
                            file_name="id_mapping.csv",
                            mime="text/csv",
                        )

                    st.markdown("---")

                    # Create ZIP file for download
                    st.subheader("Download Anonymized Files")

                    zip_buffer = io.BytesIO()
                    with zipfile.ZipFile(
                        zip_buffer, "w", zipfile.ZIP_DEFLATED
                    ) as zip_file:
                        for file_path in output_dir.glob("*.dcm"):
                            zip_file.write(file_path, file_path.name)

                    zip_buffer.seek(0)

                    st.download_button(
                        label="📥 Download Anonymized Files (ZIP)",
                        data=zip_buffer,
                        file_name="anonymized_dicoms.zip",
                        mime="application/zip",
                        type="primary",
                        use_container_width=True,
                    )

                    # Show preview of anonymized file
                    st.markdown("---")
                    st.subheader("✅ Preview: Anonymized Metadata")

                    anon_files = list(output_dir.glob("*.dcm"))
                    if anon_files:
                        anon_dicom = pydicom.dcmread(str(anon_files[0]))

                        col1, col2, col3 = st.columns(3)

                        with col1:
                            st.markdown("**Patient:**")
                            st.text(f"Name: {anon_dicom.get('PatientName', 'N/A')}")
                            st.text(f"ID: {anon_dicom.get('PatientID', 'N/A')}")
                            st.text(
                                f"Birth: {anon_dicom.get('PatientBirthDate', 'N/A')}"
                            )

                        with col2:
                            st.markdown("**Study:**")
                            st.text(f"Date: {anon_dicom.get('StudyDate', 'N/A')}")
                            st.text(f"Time: {anon_dicom.get('StudyTime', 'N/A')}")
                            st.text(f"Modality: {anon_dicom.get('Modality', 'N/A')}")

                        with col3:
                            st.markdown("**Institution:**")
                            st.text(f"Name: {anon_dicom.get('InstitutionName', 'N/A')}")
                            st.text(f"Station: {anon_dicom.get('StationName', 'N/A')}")

                        st.success("All PHI removed successfully!")

            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
                st.exception(e)

else:
    st.info("👆 Upload DICOM files to start")

    # Example
    st.markdown("---")
    st.subheader("How to use:")
    st.markdown(
        """
    1. Click "Browse files" above
    2. Select one or more DICOM files (.dcm)
    3. Review original metadata
    4. Click "Anonymize Files"
    5. Download anonymized files as ZIP
    6. Save ID mapping for your records
    """
    )

# Footer
st.markdown("---")
st.caption("⚠️ Always keep ID mapping secure and separate from anonymized files")
