import cv2
import numpy as np

def ssd(image1, image2):
    """Compute the sum of squared differences between two images."""
    return np.sum((image1.astype("float") - image2.astype("float")) ** 2)

def construct_gaussian_pyramid(image, levels, resize_factor=0.5):
    pyramid = [image]
    for _ in range(1, levels):
        image = cv2.pyrDown(image)
        pyramid.append(image)
    return pyramid

def calculate_motion(roi, frame_next, x, y, w, h, max_iterations=10, tolerance=0.01):
    best_move = (0, 0)
    for iteration in range(max_iterations):
        local_min_ssd = float('inf')
        local_best_move = (0, 0)

        for dx in range(-5, 6):  # Adjusted range
            for dy in range(-5, 6):  # Adjusted range
                new_x, new_y = int(x + dx), int(y + dy)

                if 0 <= new_x <= frame_next.shape[1] - w and 0 <= new_y <= frame_next.shape[0] - h:
                    candidate_roi = frame_next[new_y:new_y+h, new_x:new_x+w]

                    if candidate_roi.shape == roi.shape:
                        current_ssd = ssd(roi, candidate_roi)

                        if current_ssd < local_min_ssd:
                            local_min_ssd = current_ssd
                            local_best_move = (dx, dy)

        if np.linalg.norm(local_best_move) < tolerance:
            break

        best_move = local_best_move
        x, y = x + best_move[0], y + best_move[1]

    return best_move

def calculate_motion_pyramid(roi_pyramid, frame_next_pyramid, x, y, w, h, levels, resize_factor):
    for level in range(levels-1, -1, -1):
        scale = resize_factor ** level
        scaled_x, scaled_y = int(x * scale), int(y * scale)
        scaled_w, scaled_h = int(w * scale), int(h * scale)
        best_move = calculate_motion(roi_pyramid[level], frame_next_pyramid[level], scaled_x, scaled_y, scaled_w, scaled_h)
        x, y = (x + best_move[0]) / resize_factor, (y + best_move[1]) / resize_factor  # Adjust position for next level

    return int(x), int(y)

def main():
    cam = cv2.VideoCapture(0)
    ret, frame0 = cam.read()
    if not ret:
        print("Failed to grab frame")
        cam.release()
        cv2.destroyAllWindows()
        return

    levels = 3                  # number of levels in the pyramid
    resize_factor = 0.5         # resize factor between pyramid levels
    
    frame0 = cv2.cvtColor(frame0, cv2.COLOR_BGR2GRAY)
    pyramid0 = construct_gaussian_pyramid(frame0, levels, resize_factor)
    x, y, w, h = cv2.selectROI("ROI", frame0, fromCenter=False)
    cv2.destroyWindow("ROI")

    roi_pyramid = []

    # loop through each level of the pyramid
    for pyramid_level_image in pyramid0:
        # Extract the ROI from the current pyramid level
        # The ROI coordinates and size are the same as selected in the original image
        # Note: This assumes the coordinates (x, y) and size (w, h) are valid for the first level
        roi_at_current_level = pyramid_level_image[y:y+h, x:x+w]
        
        # Append the extracted ROI to the list
        roi_pyramid.append(roi_at_current_level)

    while True:
        ret, frame_next = cam.read()
        if not ret:
            break
        frame_next_gray = cv2.cvtColor(frame_next, cv2.COLOR_BGR2GRAY)
        pyramid_next = construct_gaussian_pyramid(frame_next_gray, levels, resize_factor)

        x, y = calculate_motion_pyramid(roi_pyramid, pyramid_next, x, y, w, h, levels, resize_factor)

        # ensure integers are passed to rectangle function
        cv2.rectangle(frame_next, (int(x), int(y)), (int(x) + int(w), int(y) + int(h)), (255, 0, 0), 2)
        cv2.imshow('Pyramidal Tracker', frame_next)

        k = cv2.waitKey(1)
        if k%256 == 27:
            # ASCII:ESC pressed
            print("Escape hit, closing...")
            break

    cam.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()