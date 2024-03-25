import cv2
import numpy as np
import matplotlib.pyplot as plt
import cv2


def select_points(img1_path, img2_path, num_points=2):
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

def find_point_difference(img_points):
    # since our ruler is along the x-axis we only want to find the difference in x-value
    diff = abs(img_points[0][0] - img_points[1][0])
    return diff

def calculate_focal_length(size_real, depth, size_image):
    focal_length = size_image * (depth / size_real)
    return focal_length

def main():
    img1_path = "pictures/2a/ruler_1_d13.jpg"
    img2_path = "pictures/2a/ruler_2_d10.5.jpg"

    # distances in meters
    size_real = 0.1    # choose points 10 cm apart on each image

    depth1 = 0.13      # known depth of image 1 
    depth2 = 0.105     # known depth of image 2

    # select points on image to (note the distance between using ruler)
    img1_points, img2_points = select_points(img1_path, img2_path)

    # take the points and find their difference in x-value in pixels
    size1_image = find_point_difference(img1_points)
    size2_image = find_point_difference(img2_points)

    focal_length_1 = calculate_focal_length(size_real, depth1, size1_image)
    focal_length_2 = calculate_focal_length(size_real, depth2, size2_image)

    print("Focal Length of Image 1:", focal_length_1, "pixels") # usually ~515 pixels
    print("Focal Length of Image 2:", focal_length_2, "pixels") # usually ~520 pixels
    # some measurement error in the depth of image was found
    # decided on the focal length of 515 pixels at 0.13 meter depth

if __name__ == "__main__":
    main()

