import cv2
import numpy as np

def warp_image(image, p):
    """
    warp the given image using the provided affine parameters
    """
    # construct the affine warp matrix
    warp_matrix = np.float32(np.array([
        [p[0], p[1], p[2]],
        [p[3], p[4], p[5]]
        ]))

    return cv2.warpAffine(image, warp_matrix, (image.shape[1], image.shape[0]))


def compute_gradients(image):
    """
    compute gradients using forward differences
    """

    rows, cols = image.shape
    grad_x = np.zeros(image.shape)
    grad_y = np.zeros(image.shape)

    """"
    # forward differences using slices
    grad_x[:, :-1] = np.diff(np.float32(image), axis=1)  # diff in the x direction
    grad_y[:-1, :] = np.diff(np.float32(image), axis=0)  # diff in the y direction

    return grad_x, grad_y
    """
    # Forward difference for x gradient
    for y in range(rows):
        for x in range(cols - 1):  # avoid going out of bounds
            grad_x[y, x] = np.float32(image[y, x + 1]) - np.float32(image[y, x])

    # Forward difference for y gradient
    for y in range(rows - 1):  # avoid going out of bounds
        for x in range(cols):
            grad_y[y, x] = np.float32(image[y + 1, x]) - np.float32(image[y, x])
    

    return grad_x, grad_y


def calculate_motion(template, frame, p, h, w, threshold=0.01, max_iterations=10):
    """
    apply the Lucas-Kanade method with an affine warp to find the warp parameters
    """

    for _ in range(max_iterations):
        # make the affine transformation matrix
        warp_matrix = np.float32([[p[0], p[1], p[2]],
                                  [p[3], p[4], p[5]]])
        
        # warp the frame with the current parameters
        warped_frame = cv2.warpAffine(frame, warp_matrix, (frame.shape[1], frame.shape[0]))

        

        # find the roi of our warped image using our translation arguments x and y (p[2] and p[5] respectively)
        x_trans = int(p[2])
        y_trans = int(p[5])
        warped_roi = warped_frame[y_trans:y_trans+h, x_trans:x_trans+w]

        # compute the error image
        error_image = np.float32(template) - np.float32(warped_roi)
        cv2.imshow("Error Image", error_image)
        cv2.waitKey(1) 
        # compute gradients of the warped frame
        grad_x, grad_y = compute_gradients(warped_roi)

        # initialize the Hessian matrix and the updates for the affine parameters
        H = np.zeros((6, 6))
        p_updates = np.zeros(6)

        # compute steepest descent images and update the hessian and parameter updates
        for y in range(h):
            for x in range(w):
                # compute the gradient vector for the current pixel
                grad = [grad_x[y, x], grad_y[y, x]]

                # jacobian for our current pixel
                Jacobian = np.array([
                    [x, y, 1, 0, 0, 0],
                    [0, 0, 0, x, y, 1]
                ])

                # create the steepest descent image for the current pixel
                steepest_descent_image = np.dot(grad, Jacobian)

                # update the Hessian matrix
                H += np.outer(steepest_descent_image, steepest_descent_image)
                p_updates += np.transpose(steepest_descent_image) * error_image[y, x]

        # compute the parameter update step for the whole image
        p_step = np.linalg.lstsq(H, p_updates, rcond=None)[0]

        # check to see if p_step will remain in image bounds
       
        if ((p[2] + p_step[2]) < frame.shape[1]) and ((p[2] + p_step[2]) > 0) and ((p[5] + p_step[5]) < frame.shape[0]) and ((p[5] + p_step[5]) > 0):
            # update the warp parameters
            p += p_step

        # check for convergence
        if np.linalg.norm(p_step) < threshold:
            break

    return p


def main():
    cam = cv2.VideoCapture(0)

    ret, frame0 = cam.read()
    if not ret:
        print("Failed to grab frame")
        cam.release()
        cv2.destroyAllWindows()
        return

    frame0 = cv2.cvtColor(frame0, cv2.COLOR_BGR2GRAY)
    x, y, w, h = cv2.selectROI("ROI", frame0, fromCenter=False)
    cv2.destroyAllWindows()

    # define the corners of the original ROI used as the template
    template = frame0[y:y+h, x:x+w]

    # initialize affine parameters 
    p = np.float32(np.array([1, 0, x, 0, 1, y]))

    while True:
        ret, frame = cam.read()
        if not ret:
            break

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        p = calculate_motion(template, frame, p, h, w)

        # extract our new translations from our parameters (make sure they are ints)
        x = int(p[2])
        y = int(p[5])

        # apply them 
        roi = frame[y:y+h, x:x+w]

        # display the current position of the tracker
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.imshow('Tracker', frame)

        k = cv2.waitKey(1)
        if k%256 == 27:
            # ASCII:ESC pressed
            print("Escape hit, closing...")
            break

    cam.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
