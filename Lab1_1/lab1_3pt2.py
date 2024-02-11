import numpy as np
import matplotlib.pyplot as plt
import cv2
from glob import glob

# read in and convert all images to greyscale
images = []
image_files = sorted(glob("armD32im1/*"))
for file in image_files:
    images.append(cv2.cvtColor(cv2.imread(file), cv2.COLOR_BGR2GRAY))

h, w = images[0].shape      # image dimensions
blockSize = 16              # define block size
threshold = 5               # define threshold

# find the centers of each block where vectors will be displayed and create meshgrid
X = np.arange(0, w, blockSize, dtype=int) + int(blockSize / 2)
Y = np.arange(0, h, blockSize, dtype=int) + int(blockSize / 2)
X, Y = np.meshgrid(X, Y)    

for i in range(len(images)-1):
    img_current = images[i]
    img_next = images[i+1]

    U = np.zeros(X.shape, dtype=np.float32)
    V = np.zeros(Y.shape, dtype=np.float32)

    # loop through blocks of the image
    for ix in range(0, w, blockSize):
        for iy in range(0, h, blockSize):
            if ix == 0 or iy == 0 or ix + blockSize > w or iy + blockSize > h:
                continue 

            # segment a specific block in the image
            current_block = img_current[iy:iy + blockSize, ix:ix + blockSize]
            next_block = img_next[iy:iy + blockSize, ix:ix + blockSize]

            # compute temporal derivative and apply threshold
            It = np.float32(next_block) - np.float32(current_block)
            It[It <= threshold] = 0

            # compute spatial gradients within the block
            Ix = current_block - img_current[iy:iy + blockSize, ix - 1:ix + blockSize - 1]
            Iy = current_block - img_current[iy - 1:iy + blockSize - 1, ix:ix + blockSize]

            block_X, block_Y = np.meshgrid(np.arange(blockSize), np.arange(blockSize))

            # flatten matrices for calculations
            It = It.flatten()
            Ix = Ix.flatten()
            Iy = Iy.flatten()
            block_X = block_X.flatten()
            block_Y = block_Y.flatten()

            # compute A and b for each block
            A_block = np.stack([Ix * block_X, Ix * block_Y, Iy * block_X, Iy * block_Y, Ix, Iy], axis=1)
            b_block = -It

            # solve for affine parameters for each block
            affine_params = np.linalg.lstsq(A_block, b_block, rcond=None)[0]
            a1, a2, a3, a4, a5, a6 = affine_params

            # Use the affine parameters to compute the displacement of the block
            displacement = np.dot(np.array([[a1, a2], [a3, a4]]), np.array([ix, iy])) + np.array([a5, a6])

            # update displacement arrays
            block_ix = ix // blockSize
            block_iy = iy // blockSize
            U[block_iy, block_ix] = displacement[0] - ix
            V[block_iy, block_ix] = displacement[1] - iy

    # plot original image and with the displacement vectors overlaid
    plt.imshow(img_current)
    plt.quiver(X, Y, U, V, color='r', angles='xy', scale_units='xy', scale=1)
    plt.pause(0.01)
    plt.savefig(f'lab1_3pt2_imgs/affine_{i}.png', format='png')
    plt.clf()