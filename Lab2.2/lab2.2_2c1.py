import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def generate_box_3d():
    # define the original points of box with side length of 2 centered around the origin
    points_3d = np.array([
        [1, 1, 1], [1, 1, -1], [1, -1, 1], [1, -1, -1],
        [-1, 1, 1], [-1, 1, -1], [-1, -1, 1], [-1, -1, -1]
    ])

    return points_3d

def project_points(points, camera_position, f):
   # Extract camera position coordinates
    cx, cy, cz = camera_position
    
    # Construct the projection matrix
    projection_matrix = np.array([
        [f, 0, cx, 0],
        [0, f, cy, 0],
        [0, 0, 1, -cz/f],
        [0, 0, 0, 1]
    ])

    # convert to homogeneous coordinates
    points_homogeneous = np.hstack([points, np.ones((points.shape[0], 1))])

    # apply transformation
    projected_points = points_homogeneous @ projection_matrix.T

    # return in cartesian coordinates
    return projected_points[:, :-1] / (projected_points[:, -1:])

"""
def depth_estimation(projected_points, f, baseline, img_width, img_height):
    num_images = len(projected_points)
    num_points = projected_points[0].shape[0]

    A = np.zeros((num_images * num_points, num_points))
    b = np.zeros(num_images * num_points)
    
    # fill A matrix and b vector based on disparity
    for i, projected_point in enumerate(projected_points):
        for j in range(num_points):
            A[i * num_points + j, j] = projected_point[j, 0] - img_width / 2
            b[i * num_points + j] = f * i * baseline
    
    # solve least squares problem
    depths = np.linalg.lstsq(A, b, rcond=None)[0]
    return depths
"""
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
    
    # Solve the least squares problem for disparities
    disparities = np.linalg.lstsq(A, b, rcond=None)[0]

    # Compute depths from disparities Depth = (f * baseline) / disparity
    depths = (f * baseline) / disparities

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
    # set focal length 
    f = 1500
    baseline = 100
    # create a 3D box  
    box_3d = generate_box_3d()  
    img_width, img_height = 640, 480 

    # plot box
    plot_3d_points(box_3d)

    # project the box onto images from different camera positions (moving camera along x axis distance of baseline)
    camera_positions = [np.array([i, 10, 10]) for i in range(-5, 5)] 
    projected_points_set = [project_points(box_3d, pos, f) for pos in camera_positions]

    # solve for the point depths using linear least squates
    depths = depth_estimation(projected_points_set, f, baseline)#, img_width, img_height)

    # reconstruct 3D points from the first image's projections and estimated depths
    points_3d = reconstruct_3d_points(projected_points_set[0], depths, f, img_width, img_height)

    # plot 3D points
    plot_3d_points(points_3d)

if __name__ == "__main__":
    main()



