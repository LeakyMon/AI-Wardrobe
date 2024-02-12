import pathlib
import cv2
import time
import face_recognition_models
import face_recognition
import os


# Resolve the path to the Haar cascade file
cascade_path = pathlib.Path(cv2.__file__).parent.absolute() / "data/haarcascade_frontalface_default.xml"

# Initialize the classifier with the Haar cascade file
clf = cv2.CascadeClassifier(str(cascade_path))

screenshots_dir = pathlib.Path('screenshots')
screenshots_dir.mkdir(exist_ok=True)


# Initialize some variables
face_locations = []
face_encodings = []
face_names = []
process_this_frame = True


face_detected_in_previous_frame = False

# Initialize the camera
camera = cv2.VideoCapture(0)

try:
    while True:
        # Read a frame from the camera
        ret, frame = camera.read()
        if not ret:
            break  # If the frame is not properly captured, exit the loop

        #faster face recognition process
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        # Convert the frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rgb_small_frame = small_frame[:, :, ::-1]

        # Detect faces in the grayscale frame
        faces = clf.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE
        )

        if len(faces) > 0 and not face_detected_in_previous_frame:
            # Construct a filename with the current timestamp
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            filename = f"screenshots/face_detected_{timestamp}.png"
            
            # Save the current frame as an image file
            cv2.imwrite(filename, frame)
            print(f"Screenshot taken: {filename}")

            face_detected_in_previous_frame = True

        elif len(faces) == 0:
            # Reset the flag if no face is detected
            face_detected_in_previous_frame = False

        # Draw rectangles around the detected faces
        for (x, y, width, height) in faces:
            cv2.rectangle(frame, (x, y), (x+width, y+height), (255, 255, 0), 2)

        # Display the frame with detected faces
        cv2.imshow("Faces", frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    # Release the camera and destroy all OpenCV windows
    camera.release()
    cv2.destroyAllWindows()
