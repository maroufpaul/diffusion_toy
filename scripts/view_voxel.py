import numpy as np
import matplotlib.pyplot as plt

# Load a sample
vox = np.load("data/synthetic/vox_0000.npy")

# Show a middle slice (2D view of the 3D cube)
plt.imshow(vox[32], cmap='gray')  # slice through the middle
plt.title("Slice through 3D voxel sphere")
plt.show()

