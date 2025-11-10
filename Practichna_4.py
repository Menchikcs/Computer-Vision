import cv2 as cv
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

X = []
y = []

colors = {
    "red": (0, 0, 255),
    "green": (0, 255, 0),
    "blue": (255, 0, 0),
    "yellow": (0, 255, 255),
    "orange": (0, 128, 255),
    "purple": (255, 0, 255),
    "pink": (180, 105, 255),
    "white": (255, 255, 255)
}

for color_name, bgr in colors.items():
    for _ in range(20):
        noise = np.random.randint(-20, 20, 3)
        sample = np.clip(np.array(bgr) + noise, 0, 255)
        X.append(sample)
        y.append(color_name)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y)

model = KNeighborsClassifier(n_neighbors=3)
model.fit(X,y)

cap = cv.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv.flip(frame, 1)
    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)

    mask = cv.inRange(hsv, (20,50,50), (255,255,255))
    mask = cv.morphologyEx(mask, cv.MORPH_OPEN, np.ones((11,11), np.uint8))
    mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, np.ones((11,11), np.uint8))
    contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        area = cv.contourArea(cnt)
        if area > 3000:
            x, y, w, h = cv.boundingRect(cnt)
            approx = cv.approxPolyDP(cnt, 0.04 * cv.arcLength(cnt, True), True)
            corners = len(approx)
            shape = ''
            aspect = round(w/h, 2)
            if corners == 3:
                shape = 'Triangle'
            elif corners == 4:
                if round(aspect) == 1:
                    shape = 'Square'
                else:
                    shape = 'Rectangle'
            elif corners > 6:
                shape = 'Circle'
            else:
                shape = 'Unknown'

            roi = frame[y:y+h, x:x+w]

            mean_color = cv.mean(roi)[:3]
            mean_color = np.array(mean_color).reshape(1,-1)

            color_label = model.predict(mean_color)[0]
            cv.rectangle(frame, (x, y), (x+w, y+h), (255, 255, 255), 2)
            cv.putText(frame, f"{color_label} {shape}", (x, y-10), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

    cv.imshow('color',frame)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()
