import cv2
import numpy as np

def nothing(x):
    pass
cap = cv2.VideoCapture(0)

cv2.namedWindow('image',cv2.WINDOW_FREERATIO)
cv2.createTrackbar('BRIGHTNESS','image',-10,-10,nothing)
cv2.createTrackbar('CONTRAST','image',0,30,nothing)

while(1):

    ret, frame = cap.read()
    Brightness = cv2.getTrackbarPos('BRIGHTNESS', 'image')
    Contrast = cv2.getTrackbarPos('CONTRAST', 'image')

    cap.set(cv2.CAP_PROP_CONTRAST,Contrast)
    cap.set(cv2.CAP_PROP_EXPOSURE,Brightness)


    cv2.imshow('image',frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break



cv2.destroyAllWindows()
cap.release()