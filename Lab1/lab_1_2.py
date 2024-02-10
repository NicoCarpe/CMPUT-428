import numpy as np
import matplotlib as plt
import cv2 as cv

from glob import glob

# grab images from directory
images = sorted(glob("~/Documents/CMPUT_428/Lab_1/armD32im1"))

# use first images to find image dimensions (assuming all images consistently sized)
w = images[0][0]
h = images[0][1]

blockSize = 4
threshold = None


# find the centers of each block where vectors will be displayed
X = np.arange(0, w, blockSize, dtype=int) + int(blockSize / 2)
Y = np.arange(0, h, blockSize, dtype=int) + int(blockSize / 2)
X, Y = np.meshgrid(X, Y)    


# find number of x and y blocks in image
xblocks = w / blockSize
yblocks = h / blockSize


for img in images:
  img = plt.imread(img)
  
  for ix in range(xblocks):
    for iy in range(yblocks):
      x = ix*blockSize
      y = iy*blockSize
      
      block = img[int(y):int(y+blockSize), int(x):int(x+blockSize)]
      
      #  compute motion vectors ...
      #  U[ix,iy] = xMotion
      #  V[ix,iy] = yMotion
      

  
  plt.imshow(img)
  plt.quiver(X, Y, U, V, color='r', units='dots',
           angles='xy', scale_units='xy', scale=None)
  plt.pause(0.01)
  plt.savefig(fname, format=”png”)
  plt.clf()

