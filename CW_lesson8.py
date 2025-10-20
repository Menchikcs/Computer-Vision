import cv2 as cv
import numpy as np

# face_cascade = cv.CascadeClassifier('data/haarcascades/haarcascade_frontalface_default.xml')
face_cascade = cv.CascadeClassifier('data/haarcascades/haarcascade_frontalface_alt2.xml')
eye_cascade = cv.CascadeClassifier('data/haarcascades/haarcascade_eye.xml')
smile_cascade = cv.CascadeClassifier('data/haarcascades/haarcascade_smile.xml')

cap = cv.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 10, minSize=(30,30))
    # print(faces)
    for (x,y,w,h) in faces:
        cv.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 2)

        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y + h, x:x + w]

        eyes = eye_cascade.detectMultiScale(roi_gray, 1.1, 15, minSize=(15,15))
        for (ex, ey, ew, eh) in eyes:
            cv.rectangle(roi_color, (ex,ey), (ex+ew, ey+eh), (255,0,0), 2)

        smile = smile_cascade.detectMultiScale(roi_gray, 1.5, 10, minSize=(17,17))
        for (sx, sy, sw, sh) in smile:
            cv.rectangle(roi_color, (sx,sy), (sx+sw, sy+sh), (0,0,255), 2)

        cv.putText(frame, f'Faces detected: {len(faces)}', (10,30), cv.FONT_HERSHEY_PLAIN, 1, 0, 2)



    cv.imshow('video', frame)
    if cv.waitKey(1) & 0xff == ord('q'):
        break
cap.release()
cv.destroyAllWindows()