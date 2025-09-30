import cv2 as cv
import numpy as np

img = cv.imread('images/photo_me.jpg')
print(img.shape)
cv.rectangle(img, (330, 160), (450, 300), (255, 255, 255), 5)
cv.putText(img, 'Popenko Vyacheslav detected', (150,400), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 5)



cv.imshow('image',img)
cv.waitKey(0)
cv.destroyAllWindows()