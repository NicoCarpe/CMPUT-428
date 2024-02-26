import cv2
import numpy as np

def ssd(image1, image2):
    return np.sum((image1.astype("float") - image2.astype("float")) ** 2)

def calculate_motion(roi, frame_next, x, y, w, h, max_iterations=50, tolerance=0.01):
    best_move = (0, 0)
    for iteration in range(max_iterations):
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


def main():
    # start video
    cam = cv2.VideoCapture(1)

    # wait for a frame to initialize trackers
    ret, frame = cam.read()
    if not ret:
        print("Failed to capture video")
        return
    
    # initialize video
    out = cv2.VideoWriter('lab1.2_1.mp4', -1, 30.0, (frame.shape[1], frame.shape[0]))

    # convert to greyscale and choose roi
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    x, y, w, h = cv2.selectROI("ROI", frame, fromCenter=False)
    cv2.destroyWindow("ROI")
    
    # grab roi from image0
    roi = frame[y:y+h, x:x+w]

    while True:
        # grab next frame and convert to greyscale
        ret, frame = cam.read()
        if not ret:
            break

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # find the best movement to apply to our image 
        best_move = calculate_motion(roi, frame, x, y, w, h)
        
        # update position based on the best move found
        x, y = x + best_move[0], y + best_move[1]
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
