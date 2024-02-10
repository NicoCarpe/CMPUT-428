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

    # convert list to NumPy array and ensure it's shaped as (2, N) for transposition in initTracker
    corners = np.array(corners, dtype=np.float32).reshape(2, -1)
    return corners.T


def compute_mid_line(points):
    # convert corner coordinates to numpy array and add homogeneous coordinate
    points_homogeneous = np.hstack([points, np.ones((4, 1))])

    # compute two diagonal lines l1 and l2
    l1 = np.cross(points_homogeneous[0], points_homogeneous[2])
    l2 = np.cross(points_homogeneous[1], points_homogeneous[3])

    # compute mid-point as intersection of l1 and l2
    pm = np.cross(l1, l2)

    # convert back from homogeneous coordinates by dividing by the homogenous component
    # then select the x and y values
    pm = pm / pm[2] 
    pm = pm[:2]

    # compute vanishing point as the intersection of two opposite edges
    l3 = np.cross(points_homogeneous[0], points_homogeneous[1])
    l4 = np.cross(points_homogeneous[2], points_homogeneous[3])
    p_infinity = np.cross(l3, l4)

    # convert back from homogeneous coordinates by dividing by the homogenous component
    p_infinity = p_infinity / p_infinity[2]

    # compute mid-line (lm) using intersection of mid-point and vanishing point
    lm = np.cross(pm, p_infinity)
    
    return pm, lm
    

def calculate_line_length(point1, point2):
    return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)


def draw_midline(frame, pm, lm, corners):
    # calculate lengths of l3 and l4
    len_l3 = calculate_line_length(corners[0], corners[1])
    len_l4 = calculate_line_length(corners[2], corners[3])

    # compute the average length
    average_length = (len_l3 + len_l4) / 2

    # find the midline direction (perpendicular to the line equation coefficients)
    # normalize so we can set it to the average length
    direction = np.array([lm[1], -lm[0]])
    direction_norm = direction / np.linalg.norm(direction)

    # calculate start and end points of the midline segment using our middle point
    start_point = (pm - direction_norm * (average_length / 2)).astype(int)
    end_point = (pm + direction_norm * (average_length / 2)).astype(int)

    # Draw the midline
    cv2.line(frame, tuple(start_point), tuple(end_point), (0, 255, 0), 5)

    #test
    cv2.line(frame, tuple(0,0), tuple(150,150), (0, 255, 0), 2)

def main():
    # start video
    cam = cv2.VideoCapture(0)

    # wait for a frame to initialize trackers
    ret, frame = cam.read()
    if not ret:
        print("Failed to capture video")
        return
    
    # let the user select corners (use non-greyscale for visual selection)
    corners = select_corners(frame)

    # initialize trackers on the first frame and initial corners
    lk.initTracker(frame, corners)

    while True:
        ret, frame = cam.read()
        if not ret:
            break
        
        #frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Update the tracker with the current frame
        tracker_corners = lk.updateTracker(frame)

        # convert corners for use in drawing region
        tracker_corners = tracker_corners.reshape(-1, 2)

        # Draw the tracked region on the frame
        lk.drawRegion(frame, tracker_corners, (0, 255, 0), 2)
        
    
        # find the midpoint and midline from these points
        pm, lm = compute_mid_line(tracker_corners)
        
        # draw the midpoint of the roi
        cv2.circle(frame, (int(pm[0]), int(pm[1])), 5, (0, 0, 255), -1)

        #draw the midline of the roi
        draw_midline(frame, pm, lm, corners)

        # Display frame
        cv2.imshow("Frame", frame)
        
        # Exit loop if Escape key is pressed
        if cv2.waitKey(1) == 27:  # 27 is the ASCII code for the Escape key
            break

    cam.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
