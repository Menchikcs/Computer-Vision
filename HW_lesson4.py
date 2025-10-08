import cv2 as cv
import numpy as np

img = cv.imread('photos/photoo.jpg')
scale = 2
img = cv.resize(img, (img.shape[1] // scale, img.shape[0] // scale))
print(img.shape)

img_copy_color = img.copy()
img_copy = img.copy()
img_copy = cv.cvtColor(img_copy, cv.COLOR_BGR2GRAY)
img_copy = cv.GaussianBlur(img_copy, (3,3), 4)

img_copy = cv.equalizeHist(img_copy)
img_copy = cv.Canny(img_copy, 150, 250)

contours, hierarchy = cv.findContours(img_copy, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

for cnt in contours:
    area = cv.contourArea(cnt)
    if area>220 and area<700:
        x, y, w, h = cv.boundingRect(cnt)
        cv.rectangle(img_copy_color, (x,y), (x+w, y+h), (0, 255, 0), 2)
        text_y = y-5 if y-5 > 10 else y + 15
        text = f'x:{x}, y:{y}, S:{int(area)}'
        cv.putText(img_copy_color, text, (x, text_y), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

cv.imshow('copy', img_copy_color)
cv.waitKey(0)
cv.destroyAllWindows()