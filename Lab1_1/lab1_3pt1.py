Sure, I'll provide the full code including the block-by-block warping based on the calculated motion vectors, rotation, and scaling. Here's how it looks:

python

import numpy as np
import matplotlib.pyplot as plt
import cv2
from glob import glob

# read in and convert all images to greyscale
images = []
image_files = sorted(glob("Flower60im3/*"))
for file in image_files:
    images.append(cv2.cvtColor(cv2.imread(file), cv2.COLOR_BGR2GRAY))

# use first images to find image dimensions (assuming all images consistently sized)
h, w = images[0].shape

# define block size and threshold
blockSize = 16
threshold = 5

# find the centers of each block where vectors will be displayed and create meshgrid
X = np.arange(0, w, blockSize, dtype=int) + int(blockSize / 2)
Y = np.arange(0, h, blockSize, dtype=int) + int(blockSize / 2)
X, Y = np.meshgrid(X, Y)

for i in range(len(images)-1):
    img_current = images[i]
    img_next = images[i+1]

    U = np.zeros(X.shape, dtype=np.float32)
    V = np.zeros(Y.shape, dtype=np.float32)
    R = np.zeros(X.shape, dtype=np.float32)  # Rotation
    S = np.zeros(Y.shape, dtype=np.float32)  # Scale

    # Initialize a new image for the warped result
    warped_image = np.zeros_like(img_current)

    for ix in range(0, w, blockSize):
        for iy in range(0, h, blockSize):
            if ix == 0 or iy == 0 or ix + blockSize > w or iy + blockSize > h:
                continue

            current_block = img_current[iy:iy + blockSize, ix:ix + blockSize]
            next_block = img_next[iy:iy + blockSize, ix:ix + blockSize]

            It = np.float32(current_block) - np.float32(next_block)
            It[It <= threshold] = 0

            Ix = current_block - img_current[iy:iy + blockSize, ix - 1:ix + blockSize - 1]
            Iy = current_block - img_current[iy - 1:iy + blockSize - 1, ix:ix + blockSize]

            ix_c = ix + blockSize // 2
            iy_c = iy + blockSize // 2

            Ir = -iy_c * Ix + ix_c * Iy
            Is = (ix_c * Ix + iy_c * Iy) / (np.sqrt(ix_c**2 + iy_c**2) + 1e-5)

            A = np.stack([Ix.flatten(), Iy.flatten(), Ir.flatten(), Is.flatten()], axis=1)
            b = -It.flatten()

            x = np.linalg.lstsq(A, b, rcond=None)[0]

            block_ix = ix // blockSize
            block_iy = iy // blockSize

            U[block_iy, block_ix] = x[0]
            V[block_iy, block_ix] = x[1]
            R[block_iy, block_ix] = x[2]
            S[block_iy, block_ix] = x[3]

            # Calculate the transformation matrix
            translation = np.float32([[1, 0, U[block_iy, block_ix]], [0, 1, V[block_iy, block_ix]]])
            center = (ix_c, iy_c)
            rotation = np.degrees(np.arctan(R[block_iy, block_ix]))  # Convert radians to degrees
            scale = S[block_iy, block_ix]

            rotate_scale = cv2.getRotationMatrix2D(center, rotation, scale)
            transformation = rotate_scale + translation

            # Apply the affine transformation
            transformed_block = cv2.warpAffine(img_current, transformation, (w, h))

            # Place the transformed block in the new image
            warped_image[iy:iy + blockSize, ix:ix + blockSize] = transformed_block[iy:iy + blockSize, ix:ix + blockSize]

    plt.imshow(warped_image)
    plt.quiver(X, Y, U, V, color='r', angles='xy', scale_units='xy', scale=None)
    plt.pause(0.01)
    plt.savefig(f'lab1_2_imgs/warped_frame_{i}.png', format="png")
    plt.clf()