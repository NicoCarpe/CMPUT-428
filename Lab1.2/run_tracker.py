"""
Template Tracking Python
Main script to run a tracker.

"""

import os

import numpy as np
import cv2

import utils

def run_tracker(cap, 
                ground_truths):
    """Helper function to run a tracker.
    """
    # Prepare 1st frame
    ret, frame0_rgb = cap.read()
    frame0 = cv2.cvtColor(frame0_rgb, cv2.COLOR_BGR2GRAY)   # Operate on grayscale image
    corners0 = ground_truths[0,:]

    # Initialize a tracker
    # #################################
    #tracker = None
    #tracker.initialize(frame0, corners0)
    # #################################

    # Visualization
    window_name = 'Tracking Result'
    cv2.namedWindow(window_name)
    errors = []

    # Track
    for i in range(1, ground_truths.shape[0]):
        ret, frame = cap.read()
        if not ret:
            print('Frame ", i, " could not be read')
            break
        
        # Update your tracker here
        # #################################
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        #corners = tracker.update(frame_gray)
        corners = ground_truths[i,:]    # NOTE: REPLACE THIS WITH YOUR PREDICTION
        # #################################

        # Calculate mean squared error between predicted and target corners
        err = utils.alignment_error(corners, ground_truths[i,:])
        errors.append(err)
        print('Frame id: {}, Mean Corners Error: {}'.format(i, err))

        # Drawings and visualization here
        frame = utils.draw_region(frame, ground_truths[i,:])
        frame = utils.draw_region(frame, corners, color=(0, 0, 255))
        cv2.imshow(window_name, frame)

        if cv2.waitKey(20) & 0xFF == ord('q'):
            break

    print('\nSummary')
    print('-'*20)
    print('Average Mean Corners Error: {}'.format(sum(errors)/len(errors)))

    return

if __name__ == '__main__':
    # Parameters
    path = 'data/0_Easy'
    video_name = 'box'

    # Experiment
    cap, ground_truths = utils.read_imgs_folders(path, video_name)
    run_tracker(cap, ground_truths)