import cv2 as cv
import numpy as np

img = cv.imread('images/photo_me.jpg')
print(img.shape)
cv.rectangle(img, (330, 160), (450, 300), (255, 255, 255), 5)
cv.putText(img, 'Popenko Vyacheslav', (300,130), cv.FONT_HERSHEY_TRIPLEX, 0.5, (255, 255, 255), 1)



cv.imshow('image',img)
cv.waitKey(0)

cv.destroyAllWindows()
