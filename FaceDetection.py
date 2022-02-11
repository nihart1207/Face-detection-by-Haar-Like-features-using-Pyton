import cv2
import os
import numpy as np

#  path to the “haarcascade_frontalface_alt2.xml” and “haarcascade_eye_tree_eyeglasses.xml” files
cascPathface = os.path.dirname(
    cv2.__file__) + "/data/haarcascade_frontalface_alt2.xml"
cascPatheyes = os.path.dirname(
    cv2.__file__) + "/data/haarcascade_eye_tree_eyeglasses.xml"

# we load our classifier. We are using two classifiers, one for detecting the face and others for detection eyes
faceCascade = cv2.CascadeClassifier(cascPathface)
eyeCascade = cv2.CascadeClassifier(cascPatheyes)

video_capture = cv2.VideoCapture(0)
while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # method detectMultiScale(), which receives a frame(image) as an
    # argument and runs the classifier cascade over the image
    faces = faceCascade.detectMultiScale(gray,
                                         scaleFactor=1.1,
                                         minNeighbors=5,
                                         minSize=(60, 60),
                                         flags=cv2.CASCADE_SCALE_IMAGE)
    # print(type(faces)) to identify type and if type is tuple then face not
    #  detected and if it is ndarray then face is detected
    if type(faces).__module__ == np.__name__:
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            faceROI = frame[y:y + h, x:x + w]
            eyes = eyeCascade.detectMultiScale(faceROI)
            for (x2, y2, w2, h2) in eyes:
                eye_center = (x + x2 + w2 // 2, y + y2 + h2 // 2)
                radius = int(round((w2 + h2) * 0.25))
                frame = cv2.circle(frame, eye_center, radius, (255, 0, 0), 4)

            print("Face detected")
            # print("Face Not detected")
            # label = 'Face not detected'
            # cv2.putText(frame, label, (0, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
            #             (0, 255, 0))
            # Display the resulting frame
    else:
        print("Face not detected")
    cv2.imshow('Video', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
video_capture.release()
cv2.destroyAllWindows()
