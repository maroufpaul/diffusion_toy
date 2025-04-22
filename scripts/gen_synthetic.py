"""
Generate a synthetic toy dataset of 3D voxel spheres at 64×64×64 resolution.
Each sample is a binary 3D grid saved as `data/synthetic/vox_XXXX.npy`.
"""
import numpy as np
import os


def main():
    # Configuration
    N = 1000  # number of samples
    vox_res = (64, 64, 64)
    center = np.array(vox_res) // 2
    radius_min, radius_max = 8, 16

    # Output directory (relative to project root)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.normpath(os.path.join(script_dir, "../data/synthetic"))
    os.makedirs(output_dir, exist_ok=True)

    print(f"Generating {N} synthetic voxel grids at resolution {vox_res} into {output_dir}")
    for i in range(N):
        grid = np.zeros(vox_res, dtype=np.float32)
        # random sphere radius
        radius = np.random.uniform(radius_min, radius_max)
        # fill in sphere
        for x in range(vox_res[0]):
            for y in range(vox_res[1]):
                for z in range(vox_res[2]):
                    if np.linalg.norm(np.array([x, y, z]) - center) < radius: # basicallly distance to center is less than radius that part of the circle
                        grid[x, y, z] = 1.0
        # save
        filename = f"vox_{i:04d}.npy"
        np.save(os.path.join(output_dir, filename), grid)

    print("Done.")


if __name__ == "__main__":
    main()

