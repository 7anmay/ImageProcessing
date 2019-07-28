import cv2
import numpy as np

def nothing(x):
    pass

arena = cv2.imread('IMG_1.jpg')
imgray = cv2.cvtColor(arena,cv2.COLOR_BGR2GRAY)
imgray_invert = cv2.bitwise_not(imgray, dst=None)

cv2.namedWindow('image',cv2.WINDOW_FREERATIO)

cv2.createTrackbar('threshold_max','image',0,255,nothing)
cv2.createTrackbar('threshold_min','image',0,255,nothing)


while(1):


    t_MAX = cv2.getTrackbarPos('threshold_max','image')
    t_MIN = cv2.getTrackbarPos('threshold_min','image')

    arena_gray = cv2.cvtColor(arena, cv2.COLOR_BGR2GRAY)
    ret, thresh_arena = cv2.threshold(imgray, t_MIN, t_MAX, cv2.THRESH_BINARY)

    cv2.imshow('image',thresh_arena)
    cv2.waitKey(1)

cv2.destroyAllWindows()
cap.release()