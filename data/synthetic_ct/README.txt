CT RECONSTRUCTION SYNTHETIC DATA
==================================================

Generated using Shepp-Logan phantom

Files:
------
1. phantom_ground_truth.npy
   - Ground truth image (400x400)
   - Use for comparison with reconstructed images

2. sinogram_full_180angles.npy
   - Full sampling: 180 projection angles
   - Best reconstruction quality

3. sinogram_sparse_90angles.npy
   - Sparse view: 90 projection angles
   - Test reconstruction with 50% data

4. sinogram_verysparse_45angles.npy
   - Very sparse: 45 projection angles
   - Test reconstruction with 25% data

5. sinogram_limited_120deg.npy
   - Limited angle: 0-120 degrees only
   - Test reconstruction with incomplete coverage

6. angles_*.npy
   - Corresponding projection angles for each sinogram

Usage:
------
Load sinogram and angles, then apply reconstruction:
- Filtered Back Projection (FBP)
- Algebraic Reconstruction Technique (ART)
- Simultaneous ART (SART)
