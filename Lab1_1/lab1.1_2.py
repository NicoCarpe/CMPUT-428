import numpy as np
import matplotlib.pyplot as plt
import cv2
from glob import glob


############################################################

#   Can this optic flow account for any type of motion? 
#   If not, give two distinct cases when it will not work
#   well.

#   No, this optic flow does not account for any type of 
#   motion. Two distinct cases where this optic flow would
#   not work well for would be cases with large motion, and
#   cases were there are significant changes in changes in 
#   illumination. Since we are assuming motion between 
#   frames is relatively small and constant illumination 
#   we would expect this computation to break down if these
#   assumptions are not satisfied.

############################################################


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
    
    # compute motion vectors
    # need to make sure make the data float values as the optic flow calculations may involve fractional values
    U = np.zeros(X.shape, dtype=np.float32)
    V = np.zeros(Y.shape, dtype=np.float32)

    # we can use the dimensions of the image and the block size to loop with skips
    for ix in range(0, w, blockSize):
        for iy in range(0, h, blockSize):
            
            # we want to skip the first row and column (since they have no left and upper neighbour respectively)
            # and we want to skip if the last blocks would be out of range
            if ix == 0 or iy == 0 or ix + blockSize > w or iy + blockSize > h:
                continue  

            # segment a specific block in our image
            current_block = img_current[iy:iy + blockSize, ix:ix + blockSize]
            next_block = img_next[iy:iy + blockSize, ix:ix + blockSize]
          
            # compute temporal difference
            It = np.float32(current_block) - np.float32(next_block)
            
            # apply threshold to temporal difference
            It[It <= threshold] = 0

            # compute spatial gradients within the block

            # the difference between the block and its left neighbour 
            Ix = current_block - img_current[iy:iy + blockSize, ix - 1:ix + blockSize - 1]

            # the difference between the block and its upper neighbour 
            Iy = current_block - img_current[iy - 1:iy + blockSize - 1, ix:ix + blockSize]

            # compute optic flow, flatten matricies for 
            A = np.stack([Ix.flatten(), Iy.flatten()], axis=1)
            b = -It.flatten()

            # compute motion vectors
            x = np.linalg.lstsq(A, b, rcond=None)[0]

            # need to find the block indicies to find the correct position for our motion vectors
            block_ix = ix // blockSize
            block_iy = iy // blockSize

            # assign the motion vectors to the correct block position
            U[block_iy, block_ix] = x[0]
            V[block_iy, block_ix] = x[1]

    plt.imshow(img_current)
    plt.quiver(X, Y, U, V, color='r', angles='xy', scale_units='xy', scale=None)
    plt.pause(0.01)
    plt.savefig(f'lab1_2_imgs/frame_{i}.png', format="png")
    plt.clf()
