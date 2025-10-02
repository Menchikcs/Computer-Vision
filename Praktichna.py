import cv2 as cv
import numpy as np

img = np.ones((400,600,3), np.uint8)
img[:] = 217, 217, 217
cv.rectangle(img, (20, 20), (580,380), (70, 70, 70), 5)
img1 = cv.imread('photos/me.jpg')
img1 = cv.resize(img1, (120,140))
img[50:50+img1.shape[0], 40:40+img1.shape[1]] = img1
cv.putText(img, 'Popenko Vyacheslav', (180,100), cv.FONT_HERSHEY_DUPLEX, 1, (0,0,0), 2)
cv.putText(img, 'Computer Vision Student', (180,150), cv.FONT_HERSHEY_DUPLEX, 0.8, (110, 110, 110), 2)
cv.putText(img,'Email: slavko.popenko@gmail.com', (180,220), cv.FONT_HERSHEY_DUPLEX, 0.7, (137, 138, 240), 1)
cv.putText(img,'Phone: +380671043535', (180,250), cv.FONT_HERSHEY_DUPLEX, 0.7, (137, 138, 240), 1)
cv.putText(img,'03/05/2010', (180,280), cv.FONT_HERSHEY_DUPLEX, 0.7, (137, 138, 240), 1)
cv.putText(img, 'OpenCV Business Card', (150,350), cv.FONT_HERSHEY_DUPLEX, 1, (0,0,0), 2)
img2 = cv.imread('photos/frame.png')
img2 = cv.resize(img2, (75,75))
img[240:240+img2.shape[0], 480:480+img2.shape[1]] = img2

cv.imshow('image',img)
cv.waitKey(0)
cv.destroyAllWindows()