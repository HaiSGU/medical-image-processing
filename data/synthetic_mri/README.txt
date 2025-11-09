MRI K-SPACE RECONSTRUCTION SYNTHETIC DATA
==================================================

Generated 5 synthetic MRI samples with k-space data

Files per sample (sample_XX_):
------
1. original_image.npy
   - Synthetic MRI image (256x256)
   - Ground truth for comparison

2. kspace_full.npy
   - Full k-space (100% sampling)
   - Complex-valued frequency domain data

3. kspace_50percent.npy
   - Undersampled k-space (50% data)
   - Center + random lines

4. kspace_25percent.npy
   - Heavily undersampled (25% data)
   - Center + sparse random lines

5. mask_50percent.npy & mask_25percent.npy
   - Binary sampling masks

Usage:
------
1. Load k-space data
2. Apply inverse FFT: ifft2(ifftshift(kspace))
3. Compare with original image
4. Test different reconstruction methods:
   - Zero-filling
   - Compressed sensing
   - Deep learning methods
