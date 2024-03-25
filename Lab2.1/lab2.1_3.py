import cv2
import numpy as np
import matplotlib.pyplot as plt

# flags for task selection
L1_NORM  = False            # task (b) flag
NORMALIZATION = False       # task (c) flag

def select_points(img1_path, img2_path, num_points):
    # read in images and convert to proper colors
    img1 = cv2.imread(img1_path)
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)  

    img2 = cv2.imread(img2_path)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)  

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
    
    return img1_points, img2_points


def create_matrix_A(img1_points, img2_points, num_points):
    # we will create A by stacking matrices A_i
    A = []

    for i in range(0, num_points):
        # values from first image, assume w is 1
        x = img1_points[i, 0]
        y = img1_points[i, 1]
        w = 1

        # "prime" values from second image
        x_p = img2_points[i, 0]
        y_p = img2_points[i, 1]
        w_p = 1

        # create A_i 
        A.append([0, 0, 0, -w_p*x, -w_p*y, -w_p*w, y_p*x, y_p*y, y_p*w])
        A.append([w_p*x, w_p*y, w_p*w, 0, 0, 0, -x_p*x, -x_p*y, -x_p*w])

    A = np.array(A)

    return A


def solve_for_h(A, n):
    # handle division by zero in weights by adding small offset
    offset = 1e-8
    w = 1 / (np.sqrt(n) + offset)
    
    # apply weights to A
    diag_w = np.diag(w)
    diag_wA = np.dot(diag_w, A)
    
    # solve for h using SVD on our Diag(1/sqrt(n)) . A instead of A
    U, S, V = np.linalg.svd(diag_wA)

    # transpose V.T to get V and then take last column to find h
    V = V.T
    h = V[:, -1] 

    # normalize h to satisfy ||h||_2 = 1
    h = h / np.linalg.norm(h)  

    return h 


def solve_for_n(A, h):
    n = np.abs(np.dot(A, h))
    return n


def minimize_l1_norm(A, max_iter=100, tol=1e-4):
    m = A.shape[0]

    # use a ones vector as our intital guess
    n_i = np.ones(m)  
    
    # use coordinate descent to find h that minimizes ||Ah||_1
    for iteration in range(max_iter):
        h = solve_for_h(A, n_i)
        n_new = solve_for_n(A, h)
        
        # check that change in n is significant
        if np.linalg.norm(n_new - n_i) < tol:
            break
        
        # update our n
        n_i = n_new
    
    return h


def compute_homography(img1_points, img2_points, num_points):
    # find matrix A
    A = create_matrix_A(img1_points, img2_points, num_points)

    if L1_NORM == False:
        # compute the SVD of A
        # we solve for Ah = 0 by minimizing the L2 norm of Ah through SVD which finds the homography vector h
        U, S, V = np.linalg.svd(A)

        # since svd gives us V.T transpose back to get V
        V = V.T

        # find h, which is the last column of V
        # this corresponds to the eigenvector associated with the smallest singular value of A
        h = V[:,-1]

        # normalize h to satisfy ||h||_2 = 1
        h = h / np.linalg.norm(h)  
    
    if L1_NORM == True:
        h = minimize_l1_norm(A)

    # obtain the homography H by reshaping into a 3x3 matrix
    H = h.reshape((3,3))

    # constrain H such that its Euclidean norm is 1 (we now have 8 DOF not 9)
    H = H / H[2,2]

    return H


def normalize_points(points):
    # calculate the centroid of the points
    centroid = np.mean(points, axis=0)
    
    # translate points to move the centroid to the origin
    shift_points = points - centroid
    
    # calculate the distance of each shifted point from the origin
    shift_dists = np.sqrt(shift_points[:,0]**2 + shift_points[:,1]**2)
    
    # find the average distance of these points
    avg_dist = np.mean(shift_dists)
    
    # find the scaling factor that when applied makes the average distance sqrt(2)
    scale = np.sqrt(2) / avg_dist
    
    # using this scale factor and our centroid we can
    # construct the normalization transformation matrix
    T = np.array([[scale,   0,      -scale * centroid[0]],
                  [0,       scale,  -scale * centroid[1]],
                  [0,       0,      1                   ]])
    
    # we can then find our normalized points by appling this transformation to them
    # (points need to first be in homogeneous coordinates and then transposed work with the transform)
    homogeneous_points  = np.hstack([points, np.ones((points.shape[0], 1))])  
    normalized_points = np.dot(T, homogeneous_points.T)

    # return to cartesian coordinates
    normalized_points = normalized_points / normalized_points[2, :]
    normalized_points = normalized_points[:2, :]

    # transpose the points back to be in the coorect orientation
    return normalized_points.T, T



def main():
    # paths to your images
    img1_path = 'key1.jpg'
    img2_path = 'key3.jpg'

    # select points from both images
    num_points = 6
    img1_points, img2_points = select_points(img1_path, img2_path, num_points)

    if NORMALIZATION == True:
        # normalize data
        img1_normalized, T = normalize_points(img1_points)
        img2_normalized, T_p = normalize_points(img2_points)

        # find H~
        H_t = compute_homography(img1_normalized, img2_normalized, num_points)

        # denormalize using the transformations applied to the points to find homography
        H = np.linalg.inv(T_p) @ H_t @ T

    if NORMALIZATION == False:
        H = compute_homography(img1_points, img2_points, num_points)     

    # read in the first image and get dimensions for warping from the second image
    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)

    height = img2.shape[0]
    width  = img2.shape[1]

    # apply homography H to warp the first image to align with second one
    warped_image = cv2.warpPerspective(img1, H, (width, height))

    # plot the input images and warped image together
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    
    # plot the first image and its selected points
    axs[0].imshow(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
    axs[0].plot(img1_points[:, 0], img1_points[:, 1], 'ro')  # 'ro' for red circles
    axs[0].set_title('First Image with Points')
    
    # plot the second image and its selected points
    axs[1].imshow(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))
    axs[1].plot(img2_points[:, 0], img2_points[:, 1], 'bo')  # 'bo' for blue circles
    axs[1].set_title('Second Image with Points')
    
    # plot the warped image
    axs[2].imshow(cv2.cvtColor(warped_image, cv2.COLOR_BGR2RGB))
    axs[2].set_title('Warped Image')
    
   
    #plt.tight_layout()
    
    # save the figure
    plt.savefig('homography.png')

    # display the figure
    plt.show()


if __name__ == "__main__":
    main()
