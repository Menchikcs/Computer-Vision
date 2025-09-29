import cv2 as cv
import numpy as np

# image = cv.imread('images/mountain-9761503_1280.jpg')
# # image = cv.resize(image,(500,250))
# image = cv.resize(image,(image.shape[1] // 2, image.shape[0] // 2))
# # image = cv.rotate(image, cv.ROTATE_90_CLOCKWISE)
# # image = cv.flip(image,0)
# # image = cv.GaussianBlur(image,(7,7),5) #У ДРУГОМУ ТІЛЬКИ НЕПАРНІ ЗНАЧЕННЯ
# image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
# print(image.shape)
# image = cv.Canny(image, 200, 200)
# kernel = np.ones((5, 5), np.uint8)
# image = cv.dilate(image, kernel, iterations=1)
# image = cv.erode(image, kernel, iterations=1)
#
# cv.imshow('mountain', image)
# # cv.imshow('mountain', image[0:300, 0:350])

video = cv.VideoCapture('videos/285782_small.mp4')
# video = cv.VideoCapture(0) #вебка
while True:
    success, frame = video.read()
    cv.imshow('Video', frame)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cv.destroyAllWindows()
