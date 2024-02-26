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



def compute_mid_line(points):
    # convert corner coordinates to numpy array and add homogeneous coordinate
    points = np.hstack([points, np.ones((4, 1))])

    # compute two diagonal lines l1 and l2 (assuming you pick points in clockwise pattern)
    l1 = np.cross(points[0], points[2])
    l2 = np.cross(points[1], points[3])

    # compute mid-point as intersection of l1 and l2
    pm = np.cross(l1, l2)

    # compute vanishing point as the intersection of two opposite edges
    l3 = np.cross(points[0], points[3])
    l4 = np.cross(points[1], points[2])
    p_infinity = np.cross(l3, l4)

    # compute mid-line (lm) using intersection of mid-point and vanishing point
    lm = np.cross(pm, p_infinity)

    # convert back from homogeneous coordinates by dividing by the homogenous component
    # then select the x and y values
    pm = pm / pm[2] 
    pm = pm[:2]

    lm = lm / lm[2]
    lm = lm[:2]
    
    return pm, lm


def calculate_line_length(point1, point2):
    return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)


def draw_midline(frame, pm, lm, corners):
    # calculate lengths of l3 and l4
    len_l3 = calculate_line_length(corners[0], corners[3])
    len_l4 = calculate_line_length(corners[1], corners[2])

    # compute the average length
    average_length = (len_l3 + len_l4) / 2

    # find the midline direction (perpendicular to the line equation coefficients)
    # normalize so we can set it to the average length
    direction = np.array([lm[1], -lm[0]])
    direction_norm = direction / np.linalg.norm(direction)

    # calculate start and end points of the midline segment using our middle point
    start_point = (pm - direction_norm * (average_length / 2)).astype(int)
    end_point = (pm + direction_norm * (average_length / 2)).astype(int)

    # draw the midline
    cv2.line(frame, tuple(start_point), tuple(end_point), (0, 255, 0), 2)


def main():
    # start video
    cam = cv2.VideoCapture(0)

    # wait for a frame to initialize trackers
    ret, frame = cam.read()
    if not ret:
        print("Failed to capture video")
        return
    
    # initialize video
    out = cv2.VideoWriter('2b_midline_video.mp4', -1, 30.0, (frame.shape[1], frame.shape[0]))


    # let the user select corners (use non-greyscale for visual selection)
    corners = select_corners(frame)
    
    # initialize trackers on the first frame and initial corners
    lk.initTracker(frame, corners)

    while True:
        ret, frame = cam.read()
        if not ret:
            break
        

        # update the tracker with the current frame
        corners = lk.updateTracker(frame)

        # draw the tracked region on the frame
        lk.drawRegion(frame, corners, (0, 255, 0), 2)
        
        # find the midpoint and midline from these points
        pm, lm = compute_mid_line(corners)
        
        # draw the midpoint of the roi
        cv2.circle(frame, (int(pm[0]), int(pm[1])), 5, (0, 0, 255), -1)

        #draw the midline of the roi
        draw_midline(frame, pm, lm, corners)

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
