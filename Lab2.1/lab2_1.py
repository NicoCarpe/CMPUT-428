import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def translation_matrix(translation, type):
    x = translation[0]
    y = translation[1]
    z = translation[2]

    if type == "affine":
        return np.array([x, y, z])
    
    if type == "homogeneous":
        return np.array([[1, 0, 0, x],
                         [0, 1, 0, y],
                         [0, 0, 1, z],
                         [0, 0, 0, 1]])
    

def rotation_matrix_z(theta, type):
    c, s = np.cos(np.radians(theta)), np.sin(np.radians(theta))
    if type == "affine":
        return np.array([[c, -s, 0],
                         [s, c, 0],
                         [0, 0, 1]])
    
    if type == "homogeneous":
        return np.array([[c, -s, 0, 0],
                         [s, c, 0, 0],
                         [0, 0, 1, 0],
                         [0, 0, 0, 1]])


def rotation_matrix_y(theta, type):
    c, s = np.cos(np.radians(theta)), np.sin(np.radians(theta))

    if type == "affine":
        return np.array([[c, 0, s],
                         [0, 1, 0],
                         [-s, 0, c]])
    
    if type == "homogeneous":
            return np.array([[c, 0, s, 0],
                             [0, 1, 0, 0],
                             [-s, 0, c, 0],
                             [0, 0, 0, 1]])


def affine_transform(point, translation = [0, 20, 0], rotation = [0, -10, 30]):
    point = np.array(point)

    # find the translation and rotation matricies 
    T = translation_matrix(translation, "affine")  
    Rz = rotation_matrix_z(rotation[2], "affine")
    Ry = rotation_matrix_y(rotation[1], "affine")

    # create rotation matrix 
    R = Ry @ Rz

    # apply transformation first then Rz and Ry 
    transformed_point = R @ (point + T)

    return transformed_point, R


def homogeneous_transform(point, translation = [0, 20, 0], rotation = [0, -10, 30]):
    point = np.array(point)
    point_homogeneous = np.append(point, 1)

    # find the translation and rotation matricies 
    T = translation_matrix(translation, "homogeneous")  
    Rz = rotation_matrix_z(rotation[2], "homogeneous")
    Ry = rotation_matrix_y(rotation[1], "homogeneous")

    # create the composite transformation matrix
    C = Ry @ Rz @ T 

    # apply composite transformation matrix to homogeneous point to find the tranformed point
    transformed_point = C @ point_homogeneous

    # convert back from homogeneous coordinates
    transformed_point = transformed_point / transformed_point[3]
    transformed_point = transformed_point[:3]

    return transformed_point, C


def plot_points(original_point, aff_transformed_point, homog_transformed_point):
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # plot initial point
    ax.scatter(original_point[0], original_point[1], original_point[2], color='blue', s=100, label='Initial Point')
    
    # plot transformed points
    ax.scatter(aff_transformed_point[0], aff_transformed_point[1], aff_transformed_point[2], color='red', s=100, label='Affine Transformed Point')
    ax.scatter(homog_transformed_point[0], homog_transformed_point[1], homog_transformed_point[2], color='green', s=100, label='Homogeneous Transformed Point')
    
    # Labels and legend
    ax.set_xlabel('X Axis')
    ax.set_ylabel('Y Axis')
    ax.set_zlabel('Z Axis')
    ax.legend()
    
    plt.show()


def plot_frames(original_points, aff_transformed_points, homog_transformed_points):
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # helper function to plot frames
    def plot_frame(points, label, color):
        # plot points
        for point in points:
            ax.scatter(point[0], point[1], point[2], color=color, s=50, label=label)
        
        # plot lines connnecting points
        ax.plot([points[0][0], points[2][0]],
                [points[0][1], points[2][1]],
                [points[0][2], points[2][2]], color=color)
        ax.plot([points[0][0], points[3][0]],
                [points[0][1], points[3][1]],
                [points[0][2], points[3][2]], color=color)
        ax.plot([points[1][0], points[3][0]],
                [points[1][1], points[3][1]],
                [points[1][2], points[3][2]], color=color)
        ax.plot([points[1][0], points[2][0]],
                [points[1][1], points[2][1]],
                [points[1][2], points[2][2]], color=color)
    
    # plot original frame
    plot_frame(original_points, 'Original Frame', 'blue')
    
    # plot affine transformed frame
    plot_frame(aff_transformed_points, 'Affine Transformed Frame', 'red')
    
    # plot homogeneous transformed frame
    plot_frame(homog_transformed_points, 'Homogeneous Transformed Frame', 'green')
    
    # labels and legend
    ax.set_xlabel('X Axis')
    ax.set_ylabel('Y Axis')
    ax.set_zlabel('Z Axis')
    ax.legend()
    
    plt.show()


def main():
    # Part 1:
    point = [10, 10, 10]
    aff_transformed_point, rotation_matrix = affine_transform(point)
    homog_transformed_point, transformation_matrix = homogeneous_transform(point)
    print("Affine Rotation Matrix:\n", rotation_matrix)
    print("Homogeneous Transformation Matrix:\n", transformation_matrix)

    # visualize the points
    plot_points(point, aff_transformed_point, homog_transformed_point)

    # Part 2: 
    points = [[10, 10, 10], [0, 0, 10], [10, 0, 10], [0, 10, 10]]
    aff_transformed_points = []
    homog_transformed_points =[]

    # find the transformation for each point in the frame
    for point in points:
        aff_transformed_point, rotation_matrix = affine_transform(point)
        homog_transformed_point, transformation_matrix = homogeneous_transform(point)
        print("Affine Rotation Matrix:\n", rotation_matrix)
        print("Homogeneous Transformation Matrix:\n", transformation_matrix)

        aff_transformed_points.append(aff_transformed_point)
        homog_transformed_points.append(homog_transformed_point)

    # plot the original frame and its transformations 
    plot_frames(points, aff_transformed_points, homog_transformed_points)
    
    
if __name__ == "__main__":
    main()