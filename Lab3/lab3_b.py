import cv2
import numpy as np
import matplotlib.pyplot as plt
import lk
from mpl_toolkits.mplot3d import Axes3D


def select_corners(frame, number_of_points = 10):
    # display image 
    plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    # user clicks on corners
    corners = plt.ginput(number_of_points)
    plt.close()

    # convert list to NumPy array
    corners = np.array(corners, dtype=np.float32)
    return corners


def track_corners():
    # start video
    cam = cv2.VideoCapture(0)

    # wait for a frame to initialize trackers
    ret, frame = cam.read()
    if not ret:
        print("Failed to capture video")
        return
    
    tracked_corners_list = []

    # let the user select corners (use non-greyscale for visual selection)
    tracked_corners = select_corners(frame)
    tracked_corners_list.append(tracked_corners)
    
    lk.initTracker(frame, tracked_corners)

    while True:
        ret, frame = cam.read()
        if not ret:
            break
        
        # update the tracker with the current frame
        tracked_corners = lk.updateTracker(frame)
        tracked_corners_list.append(tracked_corners)

         # draw the tracked corners on the frame
        for x, y in tracked_corners:
            cv2.circle(frame, (int(x), int(y)), radius=5, color=(0, 255, 0), thickness=-1)
        
        # display frame
        cv2.imshow("Frame", frame)
        # exit loop if escape key is pressed
        if cv2.waitKey(1) == 27:  # 27 is the ASCII code for the escape key
            break

    cam.release()
    cv2.destroyAllWindows()

    return tracked_corners_list


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

    # camera matrices (motion)
    M = U @ np.sqrt(S)

    # 3D structure (shape)
    struct_3D = np.sqrt(S) @ V_t

    # display the results
    print("Camera matrices (M):", M.shape)
    print("3D structure:", struct_3D.shape)

    # plot the 3D structure
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(struct_3D[0, :], struct_3D[1, :], struct_3D[2, :], c='r', marker='o')
    plt.show()

    return  M, struct_3D


def main():
    tracked_corners_list = track_corners()

    m = len(tracked_corners_list)  # Number of frames
    n = tracked_corners_list[0].shape[0]  # Number of points

    # initialize W with zeros
    W = np.zeros((2*m, n))

    for i, points in enumerate(tracked_corners_list):
        W[i, :] = points[:, 0]      # X coordinates
        W[m + i, :] = points[:, 1]  # Y coordinates

    n_frames = W.shape[0] // 2

    M, struct_3D = affine_SfM(W, n_frames)


if __name__ == "__main__":
    main()
