import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib

def load_kspace_data(file_path):
    """Load k-space data from a .npy file."""
    return np.load(file_path)

def inverse_fourier_transform(kspace):
    """Perform inverse Fourier transform on k-space data."""
    return np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(kspace, axes=(-2, -1)), norm='ortho'), axes=(-2, -1))

def reconstruct_mri_image(kspace_file):
    """Reconstruct MRI image from k-space data."""
    kspace_data = load_kspace_data(kspace_file)
    reconstructed_image = inverse_fourier_transform(kspace_data)
    return np.abs(reconstructed_image)

def display_reconstruction(reconstructed_image):
    """Display the reconstructed MRI image."""
    plt.figure(figsize=(8, 8))
    plt.imshow(reconstructed_image, cmap='gray')
    plt.title('Reconstructed MRI Image')
    plt.axis('off')
    plt.show()

# Example usage
if __name__ == "__main__":
    kspace_file_path = 'data/medical/slice_kspace.npy'
    reconstructed_image = reconstruct_mri_image(kspace_file_path)
    display_reconstruction(reconstructed_image)