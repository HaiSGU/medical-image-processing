import pydicom
from pydicom import dcmread
import os


def anonymize_dicom(input_file_path, output_file_path, new_patient_name="Anonymous"):
    """
    Anonymizes a DICOM file by modifying patient-identifying metadata.

    Parameters:
    - input_file_path: str, path to the input DICOM file.
    - output_file_path: str, path to save the anonymized DICOM file.
    - new_patient_name: str, new name to replace the original patient name.
    """
    # Read the DICOM file
    ds = dcmread(input_file_path)

    # Modify patient-identifying data
    ds.PatientName = new_patient_name
    ds.PatientID = "000000"
    ds.PatientBirthDate = ""
    ds.PatientAddress = ""
    ds.PatientTelephoneNumbers = ""

    # Save the anonymized DICOM file
    ds.save_as(output_file_path)
    print(f"Anonymized DICOM saved to: {output_file_path}")


if __name__ == "__main__":
    # Example usage
    input_dicom_path = os.path.join("data", "anonym", "our_sample_dicom.dcm")
    output_dicom_path = os.path.join("data", "anonym", "anonymized_dicom.dcm")
    anonymize_dicom(input_dicom_path, output_dicom_path)
