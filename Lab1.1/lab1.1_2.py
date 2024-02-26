import numpy as np
import matplotlib.pyplot as plt
import cv2
from glob import glob
import re
import re 
import math
from pathlib import Path 

# this is a functionality to ensure that the files are being read in correct order
file_pattern = re.compile(r'.*?(\d+).*?')
def get_order(file):
    match = file_pattern.match(Path(file).name)
    if not match:
        return math.inf
    return int(match.groups()[0])

# get a list of filenames
image_files = glob("armD32im1/*")

# sort filenames numerically
image_files.sort(key=get_order)

# read in and convert all images to greyscale
images = []

for file in image_files:
    print(file)
    images.append(cv2.cvtColor(cv2.imread(file), cv2.COLOR_BGR2GRAY))

# use first images to find image dimensions (assuming all images consistently sized)
h, w = images[0].shape

# define block size and threshold
blockSize = 8
threshold = 20
max_arrow_length = 20

# find the centers of each block where vectors will be displayed and create meshgrid
X = np.arange(0, w, blockSize, dtype=int) + int(blockSize / 2)
Y = np.arange(0, h, blockSize, dtype=int) + int(blockSize / 2)
X, Y = np.meshgrid(X, Y)

for i in range(len(images)-1):
    img_current = images[i]
    img_next = images[i+1]
    
    # compute motion vectors
    U = np.zeros(X.shape, dtype=np.float32)
    V = np.zeros(Y.shape, dtype=np.float32)

    for ix in range(0, w, blockSize):
        for iy in range(0, h, blockSize):
            
            # Check if the block is within the image bounds
            if ix + blockSize > w or iy + blockSize > h:
                continue  

            # segment a specific block in our image
            current_block = img_current[iy:iy + blockSize, ix:ix + blockSize]
            next_block = img_next[iy:iy + blockSize, ix:ix + blockSize]
          
            # compute temporal difference and apply threshold
            It = np.float32(next_block) - np.float32(current_block)
            It[abs(It) <= threshold] = 0

            # initialize spatial derivatives
            Ix = np.zeros(current_block.shape)
            Iy = np.zeros(current_block.shape)

            # use forward differences to compute spatial derivatives
            # difference between each pixel and its imediate right or upper neighbour
            Ix[:, :-1] = current_block[:, 1:] - current_block[:, :-1]  
            Iy[:-1, :] = current_block[1:, :] - current_block[:-1, :]  


            Ix = Ix.flatten()
            Iy = Iy.flatten()

            A = np.vstack((Ix, Iy)).T 
            b = It.flatten()
            
            # Compute motion vectors
            x = np.linalg.lstsq(A, b, rcond=None)[0]

            # Find the block indices for the motion vectors
            block_ix = ix // blockSize
            block_iy = iy // blockSize

            # Assign the motion vectors to the correct block position
            U[block_iy, block_ix] = x[0]
            V[block_iy, block_ix] = x[1]

    # find the current maximum vector length for normalization
    current_max_length = np.sqrt(np.max(U**2 + V**2))

    # scale the vectors avoiding division by zero
    if current_max_length > 0: 
        scale_factor = max_arrow_length / current_max_length
        U *= scale_factor
        V *= scale_factor

    plt.imshow(img_current, cmap='gray')
    plt.quiver(X, Y, U, V, color='r', angles='xy', scale_units='xy', scale=1)
    plt.pause(0.01)
    plt.tight_layout()
    plt.savefig(f'lab1.1_2_imgs/frame_{i}.jpg', format="jpg")
    plt.clf()

# get the images to turn into a video
image_files = glob("lab1.1_2_imgs/*")

# sort filenames numerically
image_files.sort(key=get_order)

# read the first image to determine video dimensions
frame = cv2.imread(image_files[0])
height, width, layers = frame.shape

# initialize video
video = cv2.VideoWriter("lab1.1_2.mp4", -1, 2, (width, height))

# add each frame to the video
for image_file in image_files:
    img = cv2.imread(image_file)
    video.write(img)

video.release()
