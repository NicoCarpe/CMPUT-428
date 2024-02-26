import cv2
import numpy as np
import os


############################################################

#   Noise in the camera image could be seen as movement if no 
#   threshold is set

############################################################

def capture_images(sequence_length, img_folder):
    cam = cv2.VideoCapture(0) #, cv2.CAP_FIREWIRE)  
    for i in range(sequence_length):
        ret, img = cam.read()
        if ret:
            cv2.imwrite(f"{img_folder}/{i}.jpg", img)
    cam.release()


def temporal_derivatives(images, thresh_folder):
    threshold = 150
    for i in range(1, len(images)):
        # we need to convert our images from colour to grayscale (takes in an image and color space conversion code)
        prev_frame = cv2.cvtColor(images[i-1], cv2.COLOR_BGR2GRAY)
        curr_frame = cv2.cvtColor(images[i], cv2.COLOR_BGR2GRAY)

        # we compute our temporal derivatives aproximately through successive image differences 
        frame_diff = curr_frame - prev_frame

        # we now threshold the difference image, where all pixels under the floor threshold
        # are set to 0
        frame_diff[frame_diff <= threshold] = 0

        cv2.imwrite(f"{thresh_folder}/{i}.jpg", frame_diff)
        
    


# capture and save the images, load them, then find their temporal derivatives and save those 
img_folder = "lab1.1_imgs"
sequence_length = 60

capture_images(sequence_length, img_folder)

images = []
for filename in os.listdir(img_folder):
    img = cv2.imread(os.path.join(img_folder, filename))
    images.append(img)

thresh_folder = "lab1.1_threshs"

temporal_derivatives(images, thresh_folder)
