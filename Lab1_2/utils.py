"""
Template Tracking Python
Utils code.

# To download template tracking datasets in MTF formats, check:
http://webdocs.cs.ualberta.ca/~vis/mtf/index.html

"""

import os
import sys

import cv2
import numpy as np


# ------------------------------------------------------------
# Functions for reading dataset
# ------------------------------------------------------------

def read_tracking_data(filename):
    """Read ground truth data in MTF format.
    """
    if not os.path.isfile(filename):
        print("Tracking data file not found:\n ", filename)
        sys.exit()

    data_file = open(filename, 'r')
    data_array = []
    line_id = 0
    for line in data_file:
        if line_id > 0:
            line = line.strip().split()
            coordinate = np.array([[float(line[1]), float(line[3]), float(line[5]), float(line[7])],
                                [float(line[2]), float(line[4]), float(line[6]), float(line[8])]])
            coordinate = np.round(coordinate)
            data_array.append(coordinate.astype(int))
        line_id += 1
    data_file.close()
    return np.array(data_array)

def read_imgs_folders(path,
                      video_name):
    """Read tracking dataset stored as sequences of images and return an OpenCV pipeline.
    """
    src_fname = os.path.join(path, video_name, 'frame%05d.jpg')
    cap = cv2.VideoCapture()
    if not cap.open(src_fname):
        raise Exception('The video file ', src_fname, ' could not be opened')

    ground_truths = read_tracking_data(os.path.join(path, video_name+'.txt'))
    return cap, ground_truths


# ------------------------------------------------------------
# Functions for tracking metric
# ------------------------------------------------------------

def alignment_error(corners_pred, 
                    corners_true):
    """Calculate Alignment Error (l2) error between corners.
    """
    return np.mean(np.sqrt(np.sum((corners_pred - corners_true)**2, axis=0)))


# ------------------------------------------------------------
# Functions for visualization
# ------------------------------------------------------------

def draw_region(img, corners, color=(0, 255, 0), thickness=2):
    """
    Draw the bounding box specified by the given corners
    of shape (2, 4).
    corners: [(x1, y1), (x2, y2), (x3, y3), (x4, y4)] four rectangle 
             corner coordinates.
    """
    if len(img.shape) < 3:
        vis = cv2.cvtColor(img.copy(), cv2.COLOR_GRAY2BGR)
    else:
        vis = img.copy()
    corners = np.round(corners).astype(int) # Force int
    for i in range(4):
        p1 = (corners[0, i], corners[1, i])
        p2 = (corners[0, (i + 1) % 4], corners[1, (i + 1) % 4])
        cv2.line(vis, p1, p2, color, thickness)
    return vis