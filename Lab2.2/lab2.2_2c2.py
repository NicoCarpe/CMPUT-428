import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import lk

def select_points(img_path, num_points):
    # read in images and convert to proper colors
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  
    img_shape = img.shape[:2]

    # display the two images side by side
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].imshow(img)

    # allow user to select corresponding points on each image
    points = plt.ginput(num_points) 
    plt.close()

    # return the corresponding points on each image
    img_points = np.array(points[:num_points]).astype(np.float32)
    
    return img_points, img_shape

def track_points(img_points, img_set):
    img_point_set = []

    for i, img in enumerate(img_set):
        img = cv2.imread(img)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if i == 0:
            lk.initTracker(img, img_points)

        else:
            img_points = lk.updateTracker(img)

        img_point_set.append(img_points)
    return img_point_set

def depth_estimation(points, f, baseline, img_width, img_height):
    num_images = len(points)
    num_points = points[0].shape[0]

    A = np.zeros((num_images * num_points, num_points))
    b = np.zeros(num_images * num_points)
    
    # fill A matrix and b vector based on disparity
    for i, point in enumerate(points):
        for j in range(num_points):
            A[i * num_points + j, j] = point[j, 0] - img_width / 2
            b[i * num_points + j] = f * i * baseline
    
    # solve least squares problem
    depths = np.linalg.lstsq(A, b, rcond=None)[0]
    return depths


def reconstruct_3d_points(img1_points, depths, f, img_width, img_height):
    # img_width and img_height are the dimensions of the image used for calculating the coordinates
    
    # adjust points based on the principal point (assumed at image center)
    adjusted_x = (img1_points[:, 0] - img_width / 2)
    adjusted_y = (img1_points[:, 1] - img_height / 2)
    
    # calculate 3D coordinates
    x = adjusted_x * depths / f
    y = adjusted_y * depths / f
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
    baseline = 0.05            # lateral movement in camera in meters
    num_points = 7      # number of point correspondences 

    img_set = ["pictures/2c/box_x0.0.jpg",
               "pictures/2c/box_x0.5.jpg",
               "pictures/2c/box_x1.0.jpg",
               "pictures/2c/box_x1.5.jpg",
               "pictures/2c/box_x2.0.jpg",
               "pictures/2c/box_x2.5.jpg",
               "pictures/2c/box_x3.0.jpg",
               "pictures/2c/box_x3.5.jpg",
               "pictures/2c/box_x4.0.jpg",
               "pictures/2c/box_x4.5.jpg"
    ]
               

               

    # select points
    img_points, img_shape = select_points(img_set[0], num_points)
    
    img_points_set = track_points(img_points, img_set)
    # solve for the point depths using linear least squates
    depths = depth_estimation(img_points_set, f, baseline, img_shape[0], img_shape[1])

    # reconstruct 3D points from the first image's projections and estimated depths
    points_3d = reconstruct_3d_points(img_points_set[0], depths, f, img_shape[0], img_shape[1])

    # plot 3D points
    plot_3d_points(points_3d)

if __name__ == "__main__":
    main()