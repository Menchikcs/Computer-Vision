import cv2 as cv
import numpy as np

image1 = cv.imread('photos/photo_me.jpg')
image1 = cv.resize(image1, (image1.shape[1] // 2, image1.shape[0] // 2))
image1 = cv.cvtColor(image1, cv.COLOR_BGR2GRAY)
print(image1.shape)
image1 = cv.Canny(image1, 250, 250)
kernel = np.ones((5,5), np.uint8)
image1 = cv.dilate(image1, kernel, iterations=1)
image1 = cv.erode(image1, kernel, iterations=1)

image2 = cv.imread('photos/photo_email.jpg')
image2 = cv.resize(image2, (image2.shape[1] // 3, image2.shape[0] // 3))
image2 = cv.cvtColor(image2, cv.COLOR_BGR2GRAY)
image2 = cv.Canny(image2, 250, 250)
kernel1 = np.ones((5,5), np.uint8)
image2 = cv.dilate(image2, kernel1, iterations=1)
image2 = cv.erode(image2, kernel, iterations=1)

image1 = cv.imshow('Me', image1)
image2 = cv.imshow('Email', image2)
cv.waitKey(0)
cv.destroyAllWindows()