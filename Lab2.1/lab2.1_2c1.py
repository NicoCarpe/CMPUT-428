import cv2
import numpy as np
import matplotlib.pyplot as plt
import lk


def select_corners(frame):
    # display image 
    plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    # user clicks on four corners
    number_of_points = 8
    corners = plt.ginput(number_of_points)
    plt.close()

    # convert list to NumPy array
    corners = np.array(corners, dtype=np.float32)
    return corners


def parallel_line_constraint(points):
    # convert corner coordinates to numpy array and add homogeneous coordinate
    points = np.hstack([points, np.ones((8, 1))])

    # find lines on books (assuming you pick points in clockwise pattern, and from books left to right in image)
    l1 = np.cross(points[0], points[3])
    l2 = np.cross(points[1], points[2])
    l3 = np.cross(points[4], points[7])
    l4 = np.cross(points[5], points[6])

    # normalize before constraint calculation
    l1 /= np.linalg.norm(l1)
    l2 /= np.linalg.norm(l2)
    l3 /= np.linalg.norm(l3)
    l4 /= np.linalg.norm(l4)

    # compute parallel line contraint
    epar = np.cross(np.cross(l1, l2), np.cross(l3, l4))
    
    # convert back from homogeneous coordinates by dividing by the homogenous component
    # then select the x and y values
    epar = epar / epar[2] 
    epar = epar[:2]

    return epar


def draw_lines(img, 
               corners,
               color = (0, 255, 0), 
               thickness = 2):
    
    corners = corners.astype(int)

    # draw the lines on the sides of the books (we have 2 books)
    cv2.line(img, corners[0], corners[3], color, thickness)
    cv2.line(img, corners[1], corners[2], color, thickness)
    cv2.line(img, corners[4], corners[7], color, thickness)
    cv2.line(img, corners[5], corners[6], color, thickness)


def display_error(frame, 
                  epar, 
                  position = (10, 30), 
                  font = cv2.FONT_HERSHEY_SIMPLEX, 
                  font_scale = 0.7, 
                  color = (0, 0, 0), 
                  thickness = 1):
    
    # display ell in vectorized format
    str = "({:.6f}, {:.6f})".format(epar[0], epar[1])

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
    out = cv2.VideoWriter('2c1_epar.mp4', -1, 30.0, (frame.shape[1], frame.shape[0]))

    # let the user select corners (use non-greyscale for visual selection)
    corners = select_corners(frame)

    # initialize trackers on the first frame and initial corners of the MOVING book
    # STATIC book does not need to be tracked
    tracked_corners = corners[:4,]
    lk.initTracker(frame, tracked_corners)

    while True:
        ret, frame = cam.read()
        if not ret:
            break
        
        # update the tracker with the current frame
        tracked_corners = lk.updateTracker(frame)

        # update the corners set
        corners[:4,] = tracked_corners

        epar = parallel_line_constraint(corners)

        draw_lines(frame, corners)
        
        display_error(frame, epar)

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
