import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def select_points(img1_path, img2_path, num_points):
    # read in images and convert to proper colors
    img1 = cv2.imread(img1_path)
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)  
    img1_shape = img1.shape[:2]

    img2 = cv2.imread(img2_path)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)  
    img2_shape = img2.shape[:2]

    # display the two images side by side
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].imshow(img1)
    ax[0].set_title('First Image')

    ax[1].imshow(img2)
    ax[1].set_title('Second Image')
    
    # allow user to select corresponding points on each image
    points = plt.ginput(num_points * 2) 
    plt.close()

    # return the corresponding points on each image
    img1_points = np.array(points[:num_points]).astype(np.float32)
    img2_points = np.array(points[num_points:]).astype(np.float32)
    
    return img1_points, img2_points, img1_shape, img2_shape


def point_diffs(img1_points, img2_points):
    diffs = np.abs(img1_points[:, 0] - img2_points[:, 0])
    return diffs


def calculate_depth(diffs, f, b):
    # depth calculation using the formula Z = (f * b) / d
    depths = diffs.copy()  

    for i, diff in enumerate(diffs):
        if diff != 0:
            depths[i] = (f * b) / diff
        else:  
            # avoid division by zero 
            depths[i] = 0 

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
    b = 0.03            # lateral movement in camera in meters
    num_points = 7      # number of point correspondences 

    img1_path = "pictures/2b/box_x0.jpg"
    img2_path = "pictures/2b/box_x3.jpg"

    # select points
    img1_points, img2_points, img1_shape, img2_shape = select_points(img1_path, img2_path, num_points)
    
    # calculate point differences
    diffs = point_diffs(img1_points, img2_points)
    
    # calculate depth
    depths = calculate_depth(diffs, f, b)
    
    # reconstruct 3D coordinates
    points_3d = reconstruct_3d_points(img1_points, depths, f, img1_shape[0], img1_shape[1])
    
    # plot 3D points
    plot_3d_points(points_3d)

if __name__ == "__main__":
    main()