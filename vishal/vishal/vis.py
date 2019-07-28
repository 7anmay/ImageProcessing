import cv2
import numpy as np

img = cv2.imread('ccheck.png')
hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)

lower = np.array([255,200,255])
upper = np.array([255,250,255])

vis = cv2.inRange(img,lower,upper)


cv2.imshow('img',vis)
cv2.imshow('as',img)
cv2.waitKey(0)
cv2.destroyAllWindows()