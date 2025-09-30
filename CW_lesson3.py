import cv2 as cv
import numpy as np

img = np.zeros((512,400,3), np.uint8)
#rgb = bgr
# img[:] = 145, 255, 111 #1 спосіб (увесь)
# img[100:150, 200:280] = 145, 255, 111 #2 спосіб (лиш якусь частинку)

cv.rectangle(img, (100, 100), (200,200), (145, 255, 111), -2) #в останньому -2 це заливка, а якщо 2, то товщина контурів
cv.line(img, (100, 100), (200, 200), (0, 0, 0), 5)
print(img.shape)
# cv.line(img, (0, 0), (img.shape[1], img.shape[0]), (255), 5) #діагональка
cv.line(img, (0, img.shape[0] // 2), (img.shape[1], img.shape[0] // 2), (255), 5)
cv.line(img, (img.shape[1] // 2, 0), (img.shape[1] // 2, img.shape[0]), (255, 255), 5)
cv.circle(img, (img.shape[1] // 2, img.shape[0] // 2), 25, (255, 255, 255), -1)
cv.putText(img, 'cS:gO crosshair', (100, 400), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)


cv.imshow('image',img)
cv.waitKey(0)
cv.destroyAllWindows()