import numpy
import cv2
cap = cv2.VideoCapture(0)
for i in range(20):
    _ , arena = cap.read()
Y,X = arena.shape[:2]

print 'Y=',Y
print 'X = ',X
