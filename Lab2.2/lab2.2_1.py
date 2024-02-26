import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Function to create a random 3D line point cloud
def create_line(length=5, num_points=100):
    points = np.zeros((num_points, 3))
    points[:, 0] = np.random.uniform(0, length, num_points)  # x coordinate
    return points

def create_rectangle(length=5, width=5, num_points=100):
    num_points_per_edge = num_points // 4
    lines = []

    # define the 4 corners of the rectangle
    corners = np.array([[0, 0, 0], [length, 0, 0], [length, width, 0], [0, width, 0]])

    # define the lines connecting the corners
    edges = [(0, 1), (1, 2), (2, 3), (3, 0)]

    # Create points along each line
    for start, end in edges:
        x = np.random.uniform(corners[start][0], corners[end][0], num_points_per_edge)
        y = np.random.uniform(corners[start][1], corners[end][1], num_points_per_edge)
        z = np.random.uniform(corners[start][2], corners[end][2], num_points_per_edge)
        lines.append(np.column_stack((x, y, z)))

    return np.vstack(lines)

def create_circle(radius=5, num_points=100):
    angles = np.random.uniform(0, 2 * np.pi, num_points)
    x = radius * np.cos(angles)
    y = radius * np.sin(angles)
    return np.column_stack((x, y, np.zeros(num_points)))


def create_cube(size=5, num_points=100):
    num_points_per_line = num_points // 12
    lines = []

    # define the eight corners of the cube
    corners = [(0, 0, 0), (size, 0, 0), (size, size, 0), (0, size, 0), 
               (0, 0, size), (size, 0, size), (size, size, size), (0, size, size)]

    # define the lines connecting the corners
    cube_lines = [(0, 1), (1, 2), (2, 3), (3, 0),
                  (4, 5), (5, 6), (6, 7), (7, 4),
                  (0, 4), (1, 5), (2, 6), (3, 7)]

    # create points along each line
    for start, end in cube_lines:
        x = np.random.uniform(corners[start][0], corners[end][0], num_points_per_line)
        y = np.random.uniform(corners[start][1], corners[end][1], num_points_per_line)
        z = np.random.uniform(corners[start][2], corners[end][2], num_points_per_line)
        lines.append(np.column_stack((x, y, z)))

    return np.vstack(lines)


def rotate_points(points, angle_x, angle_y, angle_z):
    # define rotation matrices
    Rx = np.array([
        [1, 0, 0, 0],
        [0, np.cos(angle_x), -np.sin(angle_x), 0],
        [0, np.sin(angle_x), np.cos(angle_x), 0],
        [0, 0, 0, 1]
    ])
    
    Ry = np.array([
        [np.cos(angle_y), 0, np.sin(angle_y), 0],
        [0, 1, 0, 0],
        [-np.sin(angle_y), 0, np.cos(angle_y), 0],
        [0, 0, 0, 1]
    ])

    Rz = np.array([
        [np.cos(angle_z), -np.sin(angle_z), 0, 0],
        [np.sin(angle_z), np.cos(angle_z), 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])

    # apply rotation
    points_homogeneous = np.hstack([points, np.ones((points.shape[0], 1))])
    rotated_points = points_homogeneous @ Rx.T @ Ry.T @ Rz.T

    # return rotated points in cartesian coordinates
    return rotated_points[:, :-1] / rotated_points[:, -1:]


def orthographic_projection(points):
    # convert to homogeneous coordinates
    points_homogeneous = np.hstack([points, np.ones((points.shape[0], 1))])

    # build orthogaphic projection matrix
    projection_matrix = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 1]
    ])

    # apply projection
    projected_points = points_homogeneous @ projection_matrix.T

    # return in cartesian coordinates
    return projected_points[:, :-1] / projected_points[:, -1:]


def perspective_projection(points, f=1, z_scene = 20):
    # convert to homogeneous coordinates
    points_homogeneous = np.hstack([points, np.ones((points.shape[0], 1))])

    # build weak perspective projection matrix
    projection_matrix = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 0, f/z_scene]
    ])

    # apply transformation
    projected_points = points_homogeneous @ projection_matrix.T

    # return in cartesian coordinates
    return projected_points[:, :-1] / projected_points[:, -1:]


def perspective_projection_with_rotation(points, angle_x=30, angle_y=30, angle_z=0):
    # apply rotation 
    rotated_points = rotate_points(points, angle_x, angle_y, angle_z)

    # get weak perspective projection of the rotated poits
    return perspective_projection(rotated_points)



def plot_shape_and_projections(index, shape, shape_name):
    # 3D plot
    ax = fig.add_subplot(4, 4, index, projection='3d')
    ax.scatter(shape[:, 0], shape[:, 1], shape[:, 2], s=15)
    ax.set_title(f'3D {shape_name}', fontsize=10)

    # projections
    ortho = orthographic_projection(shape)
    persp = perspective_projection(shape)
    persp_rot = perspective_projection_with_rotation(shape)

    plt.subplot(4, 4, index + 1)
    plt.plot(ortho[:, 0], ortho[:, 1], '.', markersize=3) 
    plt.title(f'Ortho Projection of {shape_name}', fontsize=8)

    plt.subplot(4, 4, index + 2)
    plt.plot(persp[:, 0], persp[:, 1], '.', markersize=3)  
    plt.title(f'Perspective Projection of {shape_name}', fontsize=8)

    plt.subplot(4, 4, index + 3)
    plt.plot(persp_rot[:, 0], persp_rot[:, 1], '.', markersize=3) 
    plt.title(f'Perspective with Rotation of {shape_name}', fontsize=8)

# create the shapes
line = create_line()
rectangle = create_rectangle()
circle = create_circle()
cube = create_cube()

fig = plt.figure(figsize=(20, 15))  # Adjusted figure size

# plot each shape and its projections
plot_shape_and_projections(1, line, "Line")
plot_shape_and_projections(5, rectangle, "Rectangle")
plot_shape_and_projections(9, circle, "Circle")
plot_shape_and_projections(13, cube, "Cube")

plt.tight_layout()
plt.show()
