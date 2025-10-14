import cv2 as cv
import numpy as np

cap = cv.VideoCapture(0)
ret, frame1 = cap.read()
grey1 = cv.cvtColor(frame1, cv.COLOR_BGR2GRAY)
grey1 = cv.convertScaleAbs(grey1, alpha=1.5, beta=10)

while True:
    ret, frame2 = cap.read()
    if not ret:
        print("Can't receive video")
        break
    grey2 = cv.cvtColor(frame2, cv.COLOR_BGR2GRAY)
    grey2 = cv.convertScaleAbs(grey2, alpha=1.5, beta=10)

    diff = cv.absdiff(grey1, grey2)
    _, thresh = cv.threshold(diff, 30, 255, cv.THRESH_BINARY)

    contours, _ = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        if cv.contourArea(cnt) > 2000:
            x, y, w, h = cv.boundingRect(cnt)
            cv.rectangle(frame2, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv.imshow("Video", frame2)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release() #дає змогу звільнити камеру
cv.destroyAllWindows()
