import cv2 as cv
import numpy as np

img = cv.imread('images/IMG_4072.jpeg')
img = cv.resize(img, (img.shape[1] // 5, img.shape[0] // 5))
img_copy = img.copy()

img = cv.GaussianBlur(img_copy, (5, 5), 5)
img = cv.cvtColor(img, cv.COLOR_BGR2HSV)
lower_red = np.array([0, 31, 0])
upper_red = np.array([179, 255, 190])
lower_blue = np.array([0, 0, 0])
upper_blue = np.array([179, 255, 64])
lower_yellow = np.array([17, 26, 0])
upper_yellow = np.array([32, 244, 255])
lower_green = np.array([38, 41, 71])
upper_green = np.array([120, 255, 214])
mask_red = cv.inRange(img, lower_red, upper_red)
mask_yellow = cv.inRange(img, lower_yellow, upper_yellow)
mask_blue = cv.inRange(img, lower_blue, upper_blue)
mask_green = cv.inRange(img, lower_green, upper_green)
color = ''
contours, _ = cv.findContours(mask_red, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
for cnt in contours:
    area = cv.contourArea(cnt)
    if area > 100:
        color = 'red'
        break
    else:
        contours, _ = cv.findContours(mask_blue, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            area = cv.contourArea(cnt)
            if area > 100:
                color = 'black'
                break
            else:
                contours, _ = cv.findContours(mask_yellow, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
                for cnt in contours:
                    area = cv.contourArea(cnt)
                    if area > 100:
                        color = 'yellow'
                        break
                    else:
                        contours, _ = cv.findContours(mask_green, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
                        for cnt in contours:
                            area = cv.contourArea(cnt)
                            if area > 100:
                                color = 'green'
                                break
                            else:
                                color = 'unknown'
                                break
mask_total = cv.bitwise_or(mask_red, mask_blue)
mask_total = cv.bitwise_or(mask_total, mask_yellow)
mask_total = cv.bitwise_or(mask_total, mask_green)
img = cv.bitwise_and(img, img, mask=mask_total)

contours, _ = cv.findContours(mask_total, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
for cnt in contours:
    area = cv.contourArea(cnt)
    if area > 400:
        perimeter = cv.arcLength(cnt, True)
        M = cv.moments(cnt)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
        x, y, w, h = cv.boundingRect(cnt)
        aspect_ratio = round(w/h, 2)
        compactness = round((4 * np.pi * area) / (perimeter ** 2), 2)
        approx = cv.approxPolyDP(cnt, 0.02 * perimeter, True)
        if len(approx) == 3:
            shape = 'Triangle'
        elif len(approx) == 4:
            shape = 'Quadratic'
        elif len(approx) >= 8:
            shape = 'Oval'
        else:
            shape = 'inshe'

        cv.drawContours(img_copy, [cnt], -1, (0, 0, 0), 2)
        cv.circle(img_copy, (cx, cy), 4, (0, 0, 255), -2)
        cv.putText(img_copy, f'shape: {shape}', (x, y-50), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        cv.putText(img_copy, f'color: {color}', (x, y - 70), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0),
                   2)
        cv.putText(img_copy, f'Area:{int(area)}, P:{int(perimeter)}', (x,y-30), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        cv.rectangle(img_copy, (x,y), (x+w, y+h), (0, 0, 255), 2)
        cv.putText(img_copy, f'AR:{aspect_ratio}, C:{compactness}', (x,y-10), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

cv.imwrite("result.jpg", img_copy)
cv.imshow('mask', img_copy)
cv.waitKey(0)

cv.destroyAllWindows()
