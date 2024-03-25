import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.io import loadmat
from mpl_toolkits.mplot3d import Axes3D


def affine_SfM(W, n_frames):
    """
    1.  Compute the translations for each frame to center the data.
    2.  Center the data based on the computed centroids.
    3.  Construct the measurement matrix W from the centered data.
    4.  Perform Singular Value Decomposition (SVD) on W.
    5.  Extract the camera matrices and 3D structure from the SVD result.
    """

    # step 1 & 2: center the data
    # step 3: center the already constructed measurement matrix W
    # compute the centroid of points in each frame and center the points
    W_centered = np.copy(W)
    for i in range(2*n_frames):
        centroid = np.mean(W[i, :])
        W_centered[i, :] -= centroid

    # step 4: perform SVD on the centered measurement matrix
    U, S, V_t = np.linalg.svd(W_centered, full_matrices=False)
    S = np.diag(S)  # convert the vector S to a diagonal matrix

    # step 5: extract the camera matrices and 3D structure
    # we want to only keep  the first 3 singular values for an affine camera model
    U = U[:, :3]
    S = S[:3, :3]
    V_t = V_t[:3, :]

    # use square root to balance the scaling made by the singular value across both the motion and shape
    # camera matrices (motion)
    M = U @ np.sqrt(S)

    # 3D structure (shape)
    stuct_3D = np.sqrt(S) @ V_t

    # display the results
    print("Camera matrices (M):", M.shape)
    print("3D structure:", stuct_3D.shape)

    # plot the 3D structure
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(stuct_3D[0, :], stuct_3D[1, :], stuct_3D[2, :], c='r', marker='o')
    plt.show()


def plot_origin_data(W, n_frames, imgs):
    for frame_i in range(n_frames):
        img = cv2.cvtColor(imgs[:, :, frame_i], cv2.COLOR_BAYER_BG2RGB)
        img = cv2.resize(img, (240, 320))
        
        y = W[frame_i, :]
        x = W[n_frames + frame_i, :]

        plt.imshow(img)
        plt.scatter(x, y)
        plt.pause(0.1)
        plt.clf()
        

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
        plot_origin_data(W, n_frames, imgs)

    except:
        n_frames = W.shape[0] // 2
        imgs = None

    affine_SfM(W, n_frames)

if __name__ == "__main__":
    main()

    