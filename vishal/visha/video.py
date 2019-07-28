import numpy as np
import cv2

cap = cv2.VideoCapture(0)

for i in range(20):
    cap.set(cv2.CAP_PROP_EXPOSURE,0)
    cap.set(cv2.CAP_PROP_CONTRAST,13)
    ret,frame = cap.read()


kernel = np.array((3,3),np.uint8)
frame_gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
ret,threshold = cv2.threshold(frame_gray,100,255,cv2.THRESH_BINARY)
opening = cv2.morphologyEx(threshold,cv2.MORPH_OPEN,kernel)

cv2.imshow('opening',opening)
cv2.waitKey(0)

cap.release()
cv2.destroyAllWindows()
