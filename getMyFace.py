import face_recognition
import cv2
import numpy as np

# this project uses our webcam and tries to detect and if possible recognize the faces in view.
# to speed things up we'll use 1/4 resolution of our camera and skip every other frame

video_capture = cv2.VideoCapture(0)  # references webcam #0 (the default one)

# Load in our first sample face:
gisele1_image = face_recognition.load_image_file("gisele1.jpeg")
gisele1_face_encoding = face_recognition.face_encodings(gisele1_image)[0]

# Load in our second sample face:
ryan_image = face_recognition.load_image_file("ryan.jpeg")
ryan_face_encoding = face_recognition.face_encodings(ryan_image)[0]

# Load in our third sample face:
diane_image = face_recognition.load_image_file("diane.jpeg")
diane_face_encoding = face_recognition.face_encodings(diane_image)[0]

# create an array of known faces encodings and their names
known_face_encodings = [gisele1_face_encoding,
                        ryan_face_encoding, diane_face_encoding]
known_face_names = ["Gisele", "Ryan", "Diane"]

# Setup some variables...
face_locations = []
face_encodings = []
face_names = []
process_this_frame = True

while True:  # loop forever until we quit
    # grab a single frame from our webcam
    ret, frame = video_capture.read()

    if process_this_frame:
        # resize frame to 1/4 size for speed:
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        # convert the image from BGR (which OpenCV uses) to RGB (which face_recognition uses)
        rgb_small_frame = np.ascontiguousarray(small_frame[:, :, ::-1])

        # find all the faces and face encodings in the current frame of video
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(
            rgb_small_frame, face_locations)

        face_names = []
        for face in face_encodings:
            # see if this face matches any known faces:
            matches = face_recognition.compare_faces(
                known_face_encodings, face)
            name = "Unknown"
            # we will use the known_face with the smallest distance to the current face
            face_distances = face_recognition.face_distance(
                known_face_encodings, face)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_face_names[best_match_index]

            face_names.append(name)

    process_this_frame = not process_this_frame

    # Display the results:
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        # since we scaled our image down to 1/4 we need to multiply by 4 to get an actual values
        top *= 4   # top = top * 4
        right *= 4
        bottom *= 4
        left *= 4

        # Draw a red box around each face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
        # Draw a label with a name below the face
        cv2.rectangle(frame, (left, bottom - 35),
                      (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6),
                    font, 1.0, (255, 255, 255), 1)

    # Display the resulting image
    cv2.imshow('Face Detection (type Q to quit)', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam
video_capture.release()
cv2.destroyAllWindows()
