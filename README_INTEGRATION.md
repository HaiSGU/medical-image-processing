# Integration Update

The following tools have been fully integrated into `pages/1_Processing_Pipeline.py`:

1.  **Preprocessing**: Full functionality with detailed explanations.
2.  **Anonymization**: DICOM anonymization.
3.  **Segmentation**: Brain segmentation with multiple methods.
4.  **CT Reconstruction**: Sinogram and Phantom reconstruction with FBP/SART.
5.  **MRI Reconstruction**: K-space and Image reconstruction with Partial Fourier.
6.  **Registration**: Rigid, Affine, and Deformable registration.

You can now use `pages/1_Processing_Pipeline.py` as the main entry point for all processing tasks.
The standalone files (`pages/MRI_Reconstruction.py`, `pages/Registration.py`, etc.) can be kept as backups or reference.
