import numpy as np
import cv2
from glob import glob
import re
import math
from pathlib import Path

# function to sort files numerically
file_pattern = re.compile(r'.*?(\d+).*?')
def get_order(file):
    match = file_pattern.match(Path(file).name)
    if not match:
        return math.inf
    return int(match.groups()[0])

# grab files
image_files = glob("Flower60im3/*")
image_files.sort(key=get_order)

# read and convert all images to greyscale
images = []
for file in image_files:
    images.append(cv2.cvtColor(cv2.imread(file), cv2.COLOR_BGR2GRAY))

# define block size and threshold, get image shape from first frame
h, w = images[0].shape
blockSize = 4
threshold = 10

# apply the rotation, scale, and translation to a block
def warp_block(block, u, v, r, s):
    h, w = block.shape[:2]
    center = (w // 2, h // 2)

    # create the translation and rotation matrices
    M_translate = np.array([[1, 0, u], [0, 1, v]], dtype=np.float32)
    M_rotate_scale = cv2.getRotationMatrix2D(center, np.degrees(r), 1 + s)

    # apply the translation then the rotation
    block_translated = cv2.warpAffine(block, M_translate, (w, h))
    block_warped = cv2.warpAffine(block_translated, M_rotate_scale, (w, h))
    return block_warped

# initialize the video, make sure that the width is double so that we can compare frames
video_height, video_width = h, 2 * w
video = cv2.VideoWriter("lab1.1_3.mp4", -1, 2, (video_width, video_height))

for i in range(len(images)-1):
    img_current = images[i]
    img_next = images[i+1]
    
    U = np.zeros((h, w))
    V = np.zeros((h, w))
    R = np.zeros((h, w))
    S = np.zeros((h, w))

    warped_image = np.zeros_like(img_current)

    for ix in range(0, w, blockSize):
        for iy in range(0, h, blockSize):
            if ix + blockSize > w or iy + blockSize > h:
                continue

            current_block = img_current[iy:iy + blockSize, ix:ix + blockSize]
            next_block = img_next[iy:iy + blockSize, ix:ix + blockSize]

            It = np.float32(next_block) - np.float32(current_block)
            It[abs(It) <= threshold] = 0

            Ix = np.zeros(current_block.shape)
            Iy = np.zeros(current_block.shape)

            # Use forward differences for spatial derivatives
            Ix[:, :-1] = current_block[:, 1:] - current_block[:, :-1]
            Iy[:-1, :] = current_block[1:, :] - current_block[:-1, :]

            ix_c = ix + blockSize // 2
            iy_c = iy + blockSize // 2
            Ir = -iy_c * Ix + ix_c * Iy
            Is = (ix_c * Ix + iy_c * Iy) / (np.sqrt(ix_c**2 + iy_c**2) + 1e-5)

            A = np.stack([Ix.flatten(), Iy.flatten(), Ir.flatten(), Is.flatten()], axis=1)
            b = -It.flatten()

            x = np.linalg.lstsq(A, b, rcond=None)[0]

            U[iy:iy + blockSize, ix:ix + blockSize] = x[0]
            V[iy:iy + blockSize, ix:ix + blockSize] = x[1]
            R[iy:iy + blockSize, ix:ix + blockSize] = x[2]
            S[iy:iy + blockSize, ix:ix + blockSize] = x[3]
            
            # for each block apply the warp
            u = U[iy, ix]
            v = V[iy, ix]
            r = R[iy, ix]
            s = S[iy, ix]
            warped_block = warp_block(current_block, u, v, r, s)
            warped_image[iy:iy + blockSize, ix:ix + blockSize] = warped_block
    
    # compare the warped image and the next image side by side
    comparison_frame = np.concatenate((warped_image, img_next), axis=1)

    # convert to BGR for video saving
    comparison_frame = cv2.cvtColor(comparison_frame, cv2.COLOR_GRAY2BGR)
    video.write(comparison_frame)

video.release()
