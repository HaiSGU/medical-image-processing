import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import iradon
from skimage.transform import iradon_sart

def load_sinogram(file_path):
    """Load a sinogram from a .npy file."""
    return np.load(file_path)

def reconstruct_fbp(sinogram):
    """Reconstruct an image from a sinogram using Filtered Back Projection."""
    theta = np.linspace(0.0, 180.0, max(sinogram.shape), endpoint=False)
    return iradon(sinogram, theta=theta, filter_name='ramp')

def reconstruct_sart(sinogram):
    """Reconstruct an image from a sinogram using Simultaneous Algebraic Reconstruction Technique."""
    theta = np.linspace(0.0, 180.0, max(sinogram.shape), endpoint=False)
    return iradon_sart(sinogram, theta=theta)

def plot_reconstruction(sinogram, reconstruction_fbp, reconstruction_sart):
    """Plot the sinogram and the two reconstruction results."""
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(12, 6), sharex=True, sharey=True)
    ax1.set_title("Sinogram")
    ax1.imshow(sinogram, cmap=plt.cm.Greys_r)
    ax2.set_title("Reconstruction\nFiltered Back Projection")
    ax2.imshow(reconstruction_fbp, cmap=plt.cm.Greys_r)
    ax3.set_title("Reconstruction\nSART")
    ax3.imshow(reconstruction_sart, cmap=plt.cm.Greys_r)
    ax4.set_title("Difference\nbetween reconstructions")
    ax4.imshow(reconstruction_sart - reconstruction_fbp, cmap=plt.cm.Greys_r)
    plt.show()