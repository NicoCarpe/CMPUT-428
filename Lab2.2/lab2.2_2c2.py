import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import lk
import os

def load_frames(saved_frames_directory):
    # Sort files to ensure correct order
    filenames = sorted(os.listdir(saved_frames_directory))
    frames = []
    for filename in filenames:
        frame_path = os.path.join(saved_frames_directory, filename)
        if os.path.isfile(frame_path):
            # Assuming you want to track in grayscale
            frame = cv2.imread(frame_path)
            frames.append(frame)
    return frames

def select_points(frame, number_of_points = 7):
    # display image 
    plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    # user clicks on points
    points = plt.ginput(number_of_points)
    plt.close()

    # convert list to NumPy array
    points = np.array(points, dtype=np.float32)
    return points

def track_points(frames):
    tracked_points_list = []

    # Let the user select points on the first frame
    tracked_points = select_points(frames[0])
    tracked_points_list.append(tracked_points)

    frame_shape = frames[0].shape

    lk.initTracker(frames[0], tracked_points)

    for frame in frames[1:]:
        
        # update the tracker with the current frame
        tracked_points = lk.updateTracker(frame)
        tracked_points_list.append(tracked_points)

         # draw the tracked corners on the frame
        for x, y in tracked_points:
            cv2.circle(frame, (int(x), int(y)), radius=5, color=(0, 255, 0), thickness=-1)
        
        # display frame
        cv2.imshow("Frame", frame)
        cv2.waitKey(0)
        # exit loop if escape key is pressed
        if cv2.waitKey(1) == 27:  # 27 is the ASCII code for the escape key
            break

    cv2.destroyAllWindows()

    return tracked_points_list, frame_shape


def depth_estimation(points, f, baseline):
    num_images = len(points)
    num_points = points[0].shape[0]

    # initialize matrices for the least squares solution
    A = np.zeros((num_images * num_points, num_points))
    b = np.zeros(num_images * num_points) 
    
    # fill A and b to compute disparities
    for i in range(1, num_images):
        for j in range(num_points):
            # each point's change in position should reflect its disparity due to the camera's movement

            # implies a direct relation between disparity and observed movement
            A[(i-1) * num_points + j, j] = 1  

            # b is the difference in x-coordinates between consecutive images
            b[(i-1) * num_points + j] = points[i][j, 0] - points[i-1][j, 0]
    
    # solve the least squares problem for disparities
    disparities = np.linalg.lstsq(A, b, rcond=None)[0]

    # compute depths from disparities Depth = (f * baseline) / disparity
    depths = (f * baseline) / disparities

    return depths


def reconstruct_3d_points(img1_points, depths, f, img_width, img_height):    
    # adjust points based on the principal point (assumed at image center)
    cx = (img1_points[:, 0] - img_width / 2)
    cy = (img1_points[:, 1] - img_height / 2)
    
    # calculate 3D coordinates
    x = cx * depths / f
    y = cy * depths / f
    z = depths
    
    # stack coordinates to create 3D points
    points_3d = np.vstack((x, y, z)).T
    
    return points_3d


def plot_3d_points(points_3d):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    # extracting the x, y, and z coordinates
    xs = points_3d[:, 0]
    ys = points_3d[:, 1]
    zs = points_3d[:, 2]
    
    ax.scatter(xs, ys, zs, c='r', marker='o')
    
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    
    plt.show()


def main():
    f = 515             # focal length of our camera in pixels
    baseline = 0.05     # lateral movement in camera in meters
    
    saved_frames_directory = ".\\pictures\\2c\\"
    frames = load_frames(saved_frames_directory)
    tracked_points_list, frame_shape = track_points(frames)

    # solve for the point depths using linear least squates
    depths = depth_estimation(tracked_points_list, f, baseline)

    # reconstruct 3D points from the first image's projections and estimated depths
    points_3d = reconstruct_3d_points(tracked_points_list[0], depths, f, frame_shape[0], frame_shape[1])

    # plot 3D points
    plot_3d_points(points_3d)

if __name__ == "__main__":
    main()