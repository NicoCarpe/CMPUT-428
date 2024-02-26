import cv2
import numpy as np
import matplotlib.pyplot as plt
import lk


def select_corners(frame):
    # display image 
    plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    # user clicks on four corners
    number_of_points = 4
    corners = plt.ginput(number_of_points)
    plt.close()

    # convert list to NumPy array
    corners = np.array(corners, dtype=np.float32)
    return corners


def line_to_line_constraint(points):
    # convert corner coordinates to numpy array and add homogeneous coordinate
    points = np.hstack([points, np.ones((4, 1))])

     # normalize before constraint calculation
    p1 = points[0] / np.linalg.norm(points[0])
    p2 = points[1] / np.linalg.norm(points[1])
    p3 = points[2] / np.linalg.norm(points[2])
    p4 = points[3] / np.linalg.norm(points[3])

    # calculate line-to-line contraint (assuming you pick points from one edge to another, top to bottom)
    ell = np.dot(p1, np.cross(p3, p4)) + np.dot(p2, np.cross(p3,p4))

    return ell


def draw_lines(img, 
               corners, 
               color = (0, 255, 0), 
               thickness = 2):
    
    corners = corners.astype(int)
    
    # draw the lines between points on each of the objects
    cv2.line(img, corners[0], corners[1], color, thickness)
    cv2.line(img, corners[2], corners[3], color, thickness)


def display_error(frame, 
                  ell, 
                  position = (10, 30), 
                  font = cv2.FONT_HERSHEY_SIMPLEX, 
                  font_scale = 0.7, 
                  color = (0, 0, 0), 
                  thickness = 1):
    
    # display ell in vectorized format
    str = "({:.5f})".format(ell)

    # annotate image with text
    cv2.putText(frame, str, position, font, font_scale, color, thickness)


def main():
    # start video
    cam = cv2.VideoCapture(0)

    # wait for a frame to initialize trackers
    ret, frame = cam.read()
    if not ret:
        print("Failed to capture video")
        return
    
    # initialize video
    out = cv2.VideoWriter('2c2_ell.mp4', -1, 30.0, (frame.shape[1], frame.shape[0]))

    # let the user select corners (use non-greyscale for visual selection)
    corners = select_corners(frame)
    
    # initialize trackers on 2 points of the MOVING book
    # STATIC book does not need to be tracked
    tracked_corners = corners[:2,]
    lk.initTracker(frame, tracked_corners)

    while True:
        ret, frame = cam.read()
        if not ret:
            break
        

        # update the tracker with the current frame
        tracked_corners = lk.updateTracker(frame)

        # update the corners set
        corners[:2,] = tracked_corners
        
        # find the midpoint and midline from these points
        ell = line_to_line_constraint(corners)
        
        draw_lines(frame, corners)
        
        display_error(frame, ell)

        # write the frame to the video file
        out.write(frame)

        # display frame
        cv2.imshow("Frame", frame)
        
        # exit loop if escape key is pressed
        if cv2.waitKey(1) == 27:  # 27 is the ASCII code for the escape key
            break

    cam.release()
    out.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
