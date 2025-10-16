import cv2 as cv
import numpy as np

cap = cv.VideoCapture(0)

lower_red1 = np.array([0, 100, 100])
upper_red1 = np.array([10, 255, 255])
lower_red2 = np.array([160, 100, 100])
upper_red2 = np.array([180, 255, 255])

points = []


while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv.flip(frame, 1)
    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)

    mask1 = cv.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv.inRange(hsv, lower_red2, upper_red2)
    mask = cv.bitwise_or(mask1, mask2)

    contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        area = cv.contourArea(cnt)
        cv.rectangle(frame, (0, 0), (frame.shape[1], frame.shape[0]), (0, 0, 255), 2)
        cv.putText(frame, "Object not found", (frame.shape[1] // 2, frame.shape[0] - 30), cv.FONT_HERSHEY_SIMPLEX, 1, 0,
                   2)
        object_detected = True
        if area > 1000:
            cv.drawContours(frame, [cnt], -1, (0, 255, 0), 2)

            M = cv.moments(cnt)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                cv.circle(frame, (cx,cy), 7, (255, 0, 0), -1)
                points.append((cx, cy))
    # for p in range(1, len(points)):
    #     if points[p-1] is None or points[p] is None:
    #         cv.line(frame, points[p-1], points[1], (0, 0, 255), 2)
            if object_detected:
                cv.rectangle(frame, (0,0), (frame.shape[1], frame.shape[0]), (0, 255, 0), 2)
                cv.putText(frame, "Object found", (frame.shape[1] // 2, frame.shape[0] - 30), cv.FONT_HERSHEY_SIMPLEX, 1, 0, 2)


    cv.imshow('Video', frame)
    cv.imshow('HSV', mask)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()