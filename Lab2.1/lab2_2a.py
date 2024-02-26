import cv2
import matplotlib.pyplot as plt
import numpy as np

def compute_mid_line(points):
    # convert corner coordinates to numpy array and add homogeneous coordinate
    points_homogeneous = np.hstack([points, np.ones((4, 1))])

    # compute two diagonal lines l1 and l2 (assuming you pick points in clockwise pattern)
    l1 = np.cross(points_homogeneous[0], points_homogeneous[2])
    l2 = np.cross(points_homogeneous[1], points_homogeneous[3])

    # compute mid-point as intersection of l1 and l2
    pm = np.cross(l1, l2)

    # compute vanishing point as the intersection of two opposite edges
    l3 = np.cross(points_homogeneous[0], points_homogeneous[1])
    l4 = np.cross(points_homogeneous[2], points_homogeneous[3])
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

    # draw the midline
    cv2.line(frame, tuple(start_point), tuple(end_point), (0, 255, 0), 2)


def main():
    # display image 
    image = cv2.imread("football_field.jpg")

    # convert from BGR to RGB 
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.imshow(image)

    # user clicks on four corners
    number_of_points = 4
    corners = plt.ginput(number_of_points)
    plt.close()

    points = np.array(corners)
    pm, lm = compute_mid_line(points)
    
    draw_midline(image, pm, lm, corners)

    # display image
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    cv2.imshow("Midline Displayed", image)
    cv2.waitKey(0)  

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()