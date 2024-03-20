import cv2
import numpy as np


def ssd(image1, image2):
    return np.sum((image1.astype("float") - image2.astype("float")) ** 2)


def build_gaussian_pyramid(image, levels, resize_factor):
    # start with original image in array
    pyramid = [image]
    for _ in range(1, levels):
        # find image dimensions
        rows, cols, _ = image.shape

        # downsample according to the resize factor
        image = cv2.pyrDown(image, dstsize=(cols // resize_factor, rows // resize_factor))

        # append downsampled image to array
        pyramid.append(image)
        
    return pyramid


def calculate_motion(roi, frame_next, x, y, w, h, max_iterations=50, tolerance=0.01):
    best_move = (0, 0)
    for _ in range(max_iterations):
        local_min_ssd = float('inf')
        local_best_move = (0, 0)
        
        for dx in range(-5, 5):  
            for dy in range(-5, 5):  
                new_x, new_y = x + dx, y + dy

                # check if the new coordinates are within the frame
                if (0 <= new_x <= frame_next.shape[1] - w) and (0 <= new_y <= frame_next.shape[0] - h):
                    next_roi = frame_next[new_y:new_y+h, new_x:new_x+w]

                    # calculate SSD (make sure only if candidate ROI is valid
                    if next_roi.shape == roi.shape:
                        current_ssd = ssd(roi, next_roi)
                        
                        if current_ssd < local_min_ssd:
                            local_min_ssd = current_ssd
                            local_best_move = (dx, dy)

        # Update the position only if a better move is found
        if np.linalg.norm(local_best_move) >= tolerance:
            best_move = local_best_move
            x, y = x + best_move[0], y + best_move[1]
        else:
            break

    return best_move


def calculate_motion_pyramid(roi_pyramid, frame_pyramid, x, y, w, h, resize_factor):
    levels = len(roi_pyramid)
    move = (0, 0)  # Initialize motion
    
    for level in range(levels-1, -1, -1):
        scale = resize_factor ** level
        scaled_x, scaled_y = int(x / scale + move[0] * 2 ** (level - levels + 1)), int(y / scale + move[1] * 2 ** (level - levels + 1))
        scaled_w, scaled_h = int(w / scale), int(h / scale)
        
        # Adjust starting position based on move from previous level
        move = calculate_motion(
            roi_pyramid[level], frame_pyramid[level], 
            scaled_x, scaled_y, scaled_w, scaled_h
        )
        
        # Scale move up for the next (finer) level
        if level > 0:
            move = (move[0] * 2, move[1] * 2)
    
    # Apply final move to original coordinates
    return move

def main():
    cam = cv2.VideoCapture(0)

    ret, frame = cam.read()
    if not ret:
        print("Failed to capture video")
        return
    
    out = cv2.VideoWriter('output.mp4', -1, 20.0, (frame.shape[1], frame.shape[0]))
    
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    x, y, w, h = cv2.selectROI("ROI Selector", frame, fromCenter=False)
    cv2.destroyWindow("ROI Selector")
    
    roi = frame[y:y+h, x:x+w]
    
    levels = 3  # Number of levels in the Gaussian pyramid
    resize_factor = 2  # Resize factor between levels

    while True:
        ret, frame = cam.read()
        if not ret:
            break

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        roi_pyramid = build_gaussian_pyramid(roi, levels, resize_factor)
        frame_pyramid = build_gaussian_pyramid(frame, levels, resize_factor)
        
        best_move = calculate_motion_pyramid(roi_pyramid, frame_pyramid, x, y, w, h, resize_factor=resize_factor)
        
        x += best_move[0]
        y += best_move[1]
        roi = frame[y:y+h, x:x+w]

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
