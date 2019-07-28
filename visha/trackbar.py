import cv2
import numpy as np

def nothing(x):
    pass
cap = cv2.VideoCapture(0)
cv2.namedWindow('image',cv2.WINDOW_FREERATIO)
cv2.createTrackbar('H_MAX','image',0,255,nothing)
cv2.createTrackbar('S_MAX','image',0,255,nothing)
cv2.createTrackbar('V_MAX','image',0,255,nothing)
cv2.createTrackbar('H_MIN','image',0,255,nothing)
cv2.createTrackbar('S_MIN','image',0,255,nothing)
cv2.createTrackbar('V_MIN','image',0,255,nothing)

switch = '0 : OFF \n 1 : ON'
cv2.createTrackbar('switch', 'image',0,1,nothing)

while(1):

    ret, frame = cap.read()
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    #cv2.imshow('image', hsv)
    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        break
    H_MAX = cv2.getTrackbarPos('H_MAX','image')
    S_MAX = cv2.getTrackbarPos('S_MAX','image')
    V_MAX = cv2.getTrackbarPos('V_MAX','image')
    H_MIN = cv2.getTrackbarPos('H_MIN', 'image')
    S_MIN = cv2.getTrackbarPos('S_MIN', 'image')
    V_MIN = cv2.getTrackbarPos('V_MIN', 'image')
    s = cv2.getTrackbarPos('switch','image')
    kernel = np.ones((9,9), np.uint8)
    kernel_1 = np.ones((1, 1), np.uint8)

    lower_color = np.array([H_MIN,S_MIN,V_MIN])
    upper_color = np.array([H_MAX,S_MAX,V_MAX])
    image = cv2.inRange(hsv,lower_color,upper_color)
   # blur = cv2.medianBlur(image,15)
    image = cv2.morphologyEx(image,cv2.MORPH_OPEN,kernel)

    cv2.imshow('image',image)
    cv2.imshow('frame',frame)



cv2.destroyAllWindows()
cap.release()