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
    cv2.imwrite('pictures/ruler_0cm.jpg', frame)
  
    cap.release()




def calculate_focal_length(object_size_real, distance_to_object, object_size_image):
    """
    Calculate the focal length of a camera.

    :param object_size_real: Actual size of the object (e.g., length of the ruler) in meters.
    :param distance_to_object: Distance from the camera to the object in meters.
    :param object_size_image: Size of the object in the image in pixels.
    :return: Focal length in pixels.
    """
    focal_length = object_size_image * (distance_to_object / object_size_real)
    return focal_length


take_picture()

# distances in meters
object_size_real = 0.3   
distance_to_object = 2  

# size in pixels
object_size_image = 500

focal_length = calculate_focal_length(object_size_real, distance_to_object, object_size_image)
print("Focal Length:", focal_length, "pixels")


