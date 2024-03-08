import cv2

def take_picture():
    # Open a video capture object
    cap = cv2.VideoCapture(1)

    # check if frame is captured
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture video")
        return

    # Save the frame to an image file
    cv2.imwrite('pictures/2c/box_d13_x5.0.jpg', frame)
  
    cap.release()

def main():
    take_picture()
    print("Image Captured")

if __name__ == "__main__":
    main()