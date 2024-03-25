import cv2
import numpy as np

def warp_image(image, p):   
    # construct the affine warp matrix
    warp_matrix = np.array(np.float32(
        [[p[0], p[1], p[2]],
         [p[3], p[4], p[5]]],
        ))
  

    # warp the given image using the provided affine parameters
    return cv2.warpAffine(image, warp_matrix, (image.shape[1], image.shape[0]))


def compute_gradients(image):
    # compute gradients using forward differences
   
    grad_x = np.zeros(image.shape)
    grad_y = np.zeros(image.shape)

    # forward differences
    grad_x[:, :-1] = image[:, 1:] - image[:, :-1]
    grad_y[:-1, :] = image[1:, :] - image[:-1, :]

    return grad_x, grad_y


def calculate_motion(template, frame, p, h, w, threshold=0.01, max_iterations=10):
    # apply the Lucas-Kanade method with an affine warp to find the warp parameters
    for _ in range(max_iterations):
        # warp the entire frame with the current estimate of p
        warped_frame = warp_image(frame, p)

        # compute gradients of the warped frame (now doing it after warping)
        grad_x, grad_y = compute_gradients(warped_frame)

        # extract the ROI from the warped frame and its gradients
        x = int(p[2])
        y = int(p[5])
        warped_roi = warped_frame[y:y+h, x:x+w]
        grad_x_roi = grad_x[y:y+h, x:x+w]
        grad_y_roi = grad_y[y:y+h, x:x+w]

        # compute the error image
        error_image = np.float32(template) - np.float32(warped_roi)

        # initialize the Hessian matrix and the updates for the affine parameters
        H = np.zeros((6, 6))
        sd_update = np.zeros(6)

        # compute steepest descent images and update the Hessian and parameter updates
        for i in range(h):
            for j in range(w):
                # calculate the Jacobian for an affine warp at the current pixel
                J = np.array([[j, 0, i, 0, 1, 0],
                              [0, j, 0, i, 0, 1]])

                # steepest descent image for the current pixel
                sd_img = np.array([[grad_x_roi[i, j], grad_y_roi[i, j]]]) @ J

                # update Hessian and steepest descent parameter update
                H += sd_img.T @ sd_img
                sd_update += sd_img.flatten() * error_image[i, j]

        # compute the parameter update step for the whole image
        p_step = np.linalg.lstsq(H, sd_update, rcond=None)[0]

        # check and update the warp parameters p using p_step
        p += p_step

        # check for convergence
        if np.linalg.norm(p_step) < threshold:
            break

    return p



def main():
    cam = cv2.VideoCapture(0)

    ret, frame = cam.read()
    if not ret:
        print("Failed to grab frame")
        cam.release()
        cv2.destroyAllWindows()
        return

    # initialize video
    out = cv2.VideoWriter('lab1.2_3.mp4', -1, 30.0, (frame.shape[1], frame.shape[0]))

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    x, y, w, h = cv2.selectROI("ROI", frame, fromCenter=False)
    cv2.destroyAllWindows()

    # define the corners of the original ROI used as the template
    template = frame[y:y+h, x:x+w]

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

        # display the current position of the tracker
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.imshow('Tracker', frame)

        # write the frame to the video file
        out.write(frame)


        k = cv2.waitKey(1)
        if k%256 == 27:
            # ASCII:ESC pressed
            print("Escape hit, closing...")
            break

    cam.release()
    out.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
