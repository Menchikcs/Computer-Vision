import cv2 as cv
import numpy as np

img = cv.imread('images/raccoon.jpg')
img_copy = img.copy()

img = cv.GaussianBlur(img_copy, (5, 5), 5)
img = cv.cvtColor(img, cv.COLOR_BGR2HSV)
lower = np.array([0, 0, 17]) #мінімальний поріг
upper = np.array([179, 255, 252]) #максимальний поріг
mask = cv.inRange(img, lower, upper)
img = cv.bitwise_and(img, img, mask=mask)

contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
for cnt in contours:
    area = cv.contourArea(cnt)
    if area > 150:
        perimeter = cv.arcLength(cnt, True) #True - контур замкнутий
        M = cv.moments(cnt)
        #Шукаємо центр мас. Центр мас - середня позиція контуру
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])

        x, y, w, h = cv.boundingRect(cnt)
        aspect_ratio = round(w/h, 2) #допомагає відрізняти співвідношення сторін
        #Міра округлості об'єкта
        compactness = round((4 * np.pi * area) / (perimeter ** 2), 2)

        approx = cv.approxPolyDP(cnt, 0.02 * perimeter, True) #точність апроксимації
        if len(approx) == 3:
            shape = 'Triangle'
        elif len(approx) == 4:
            shape = 'Quadratic'
        elif len(approx) > 8:
            shape = 'Oval'
        else:
            shape = 'inshe'

        cv.drawContours(img_copy, [cnt], -1, (0, 0, 0), 2)
        cv.circle(img_copy, (cx, cy), 4, (0, 0, 255), -2)
        cv.putText(img_copy, f'shape: {shape}', (x, y-10), cv.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
        cv.putText(img_copy, f'Area:{int(area)}, P:{int(perimeter)}', (x,y+30), cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        cv.rectangle(img_copy, (x,y), (x+w, y+h), (0, 0, 255), 2)
        cv.putText(img_copy, f'AR:{aspect_ratio}, C:{compactness}', (x,y+10), cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)


cv.imshow('mask', img_copy)
cv.waitKey(0)
cv.destroyAllWindows()