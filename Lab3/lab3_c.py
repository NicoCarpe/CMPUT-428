import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from mpl_toolkits.mplot3d import Axes3D

def isotropic_scaling(W):
    mean = np.mean(W, axis=1, keepdims=True)
    std_dev = np.std(W, axis=1, keepdims=True)
    W_normalized = (W - mean) / std_dev

    return W_normalized, mean, std_dev

def reproject_and_update_depths(M, S_hat, W_observed, n_frames):
    n_points = S_hat.shape[1]
   
    # initialize depths to ones
    depths = np.ones((n_frames, n_points))
    
    for i in range(n_frames):
        # extract the camera matrix for the current view (including its 2D projection part)
        M_i = M[2*i:2*i+2, :]
    
        projected_points = M_i @ S_hat

        projected_x = projected_points[0, :] 
        projected_y = projected_points[1, :] 
        
        # extract observed points for comparison
        observed_x = W_observed[2*i, :]
        observed_y = W_observed[2*i+1, :]
        
        # calculate the reprojection error (Euclidean distance in 2D space)
        error_x = observed_x - projected_x
        error_y = observed_y - projected_y
        reprojection_error = np.sqrt(error_x**2 + error_y**2)
        
        # update depths based on reprojection error
        depths[i, :] += reprojection_error
    
    return depths

def normalize_depths(depths):
    # normalize rows of depths
    for i in range(depths.shape[0]):
        norm = np.linalg.norm(depths[i, :], ord=2)

        # divide row by l2 norm if norm > 0 to set norm to 1
        if norm > 0:
            depths[i, :] /= norm
    
    # normalize columns of depths
    for j in range(depths.shape[1]):
        norm = np.linalg.norm(depths[:, j], ord=2)

        # divide column by l2 norm if norm > 0 to set norm to 1
        if norm > 0:
            depths[:, j] /= norm
            
    return depths


def projective_reconstruction(W, n_frames, max_iterations=100, convergence_threshold=1e-4):
    W_normalized, mean, std_dev = isotropic_scaling(W)
    
    # initialize projective depths to 1 for all points, therefore implicitly W_normalized is our measurement matrix
    
    # perform initial SVD on the adjusted measurement matrix, keeping rank-4 approximation
    U, S, V_t = np.linalg.svd(W_normalized, full_matrices=False)
    U = U[:, :4]
    S_matrix = np.diag(S[:4])
    V_t = V_t[:4, :]
    
    # decomposition to find camera matrices (M) and 3D structure (S_hat)
    M = U @ np.sqrt(S_matrix)
    S_hat = np.sqrt(S_matrix) @ V_t
    
    for iteration in range(max_iterations):
        # reproject points and update depths
        depths = reproject_and_update_depths(M, S_hat, W_normalized, n_frames)

        # normalize depths
        depths_normalized = normalize_depths(depths)

        # ensure depths_normalized is ready for broadcasting (need to apply depths to both x and y coordinates)
        depths_expanded = np.repeat(depths_normalized, 2, axis=0)
    
        # apply the depths to W_normalized using broadcasting
        W_adjusted = W_normalized * depths_expanded
            
        # re-perform SVD with the adjusted measurement matrix
        U, S, V_t = np.linalg.svd(W_adjusted, full_matrices=False)

        M = U[:, :4] @ np.sqrt(np.diag(S[:4]))
        S_hat = np.sqrt(np.diag(S[:4])) @ V_t[:4, :]
        
        # check convergence based on the change in S or another suitable metric
        if np.linalg.norm(S_matrix - np.diag(S[:4]), ord='fro') < convergence_threshold:
            print(f"Converged after {iteration + 1} iterations.")
            break
        S_matrix = np.diag(S[:4])

    # extract 3D points before denormalizing
    struct_3D = S_hat[:3, :]
    
    # denormalize the 3D structure
    for i in range(3):
        struct_3D[i, :] = (struct_3D[i, :] * std_dev[i]) + mean[i]

    # Display the results
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(struct_3D[0, :], struct_3D[1, :], struct_3D[2, :], c='r', marker='o')
    plt.title('Projective Reconstruction')
    plt.show()

    return M, struct_3D

def main():
    dataset = "HouseTallBasler64.mat"
    #dataset = "affrec1.mat"
    #dataset = "affrec3.mat"

    data = loadmat(dataset)
    
    W = data["W"].astype(np.float64)
    print(W.shape)

    try:
        n_frames = data["NrFrames"].item()
        imgs = data["mexVims"]
        #plot_origin_data(W, n_frames, imgs)

    except:
        n_frames = W.shape[0] // 2
        imgs = None

    M, struct_3D = projective_reconstruction(W, n_frames)

if __name__ == "__main__":
    main()

    