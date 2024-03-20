import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def generate_box_3d():
    # Define the original 3D coordinates of the box corners centered around the origin
    # Box with side length of 2 for simplicity
    points_3d = np.array([
        [1, 1, 1], [1, 1, -1], [1, -1, 1], [1, -1, -1],
        [-1, 1, 1], [-1, 1, -1], [-1, -1, 1], [-1, -1, -1]
    ])
    return points_3d.T  # Return as 3xN for consistency with rotation matrices

def project_points(points, camera_position, f):
    # Adjust points based on camera position
    adjusted_points = points - camera_position[:, np.newaxis]
    
    # Perspective projection formula to project 3D points to 2D
    projected_points = f * adjusted_points[:2, :] / adjusted_points[2, :]
    
    return projected_points

def main():
    # Camera and projection parameters
    f = 800  # Focal length in arbitrary units
    b = 0.2  # Baseline (distance between camera positions)
    n_images = 10
    img_width, img_height = 640, 480
    
    # Generate a 3D box
    box_3d = generate_box_3d()
    
    # Generate camera positions along the X-axis
    camera_positions = [np.array([i * b, 0, 5]) for i in range(n_images)]  # Place camera in front of the box

    # Project the 3D box onto 2D planes from different camera positions
    projected_points_set = [project_points(box_3d, pos, f) for pos in camera_positions]

    # Simulate depth estimation (this part is simplistic and requires a proper stereo vision algorithm for real scenarios)
    # Assume we know the real Z coordinates (depths) of points for demonstration
    real_depths = box_3d[2, :]

    # Reconstruct 3D points based on the first projection and estimated depths
    # For simplicity, this example uses the real depths
    first_projection = projected_points_set[0]
    reconstructed_3d_points = np.vstack((first_projection, real_depths))

    # Plot the original 3D box and reconstructed points for comparison
    fig = plt.figure(figsize=(14, 7))
    
    # Original 3D Box
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.scatter(*box_3d, color='blue', label='Original Box')
    ax1.set_title('Original 3D Box')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.legend()

    # Reconstructed 3D Points
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.scatter(*reconstructed_3d_points, color='red', label='Reconstructed Points')
    ax2.set_title('Reconstructed 3D Points')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')
    ax2.legend()

    plt.show()

if __name__ == "__main__":
    main()
