import cv2
import numpy as np
import time

cap =cv2.VideoCapture(0)
cent_blue = np.zeros((1,2), np.uint8)
cent_pink = np.zeros((1,2), np.uint8)

time.sleep(4)

while True:
    cap.set(cv2.CAP_PROP_CONTRAST,13)
    cap.set(cv2.CAP_PROP_EXPOSURE,0)
    ret,frame = cap.read()
    #frame[:] = cv2.equalizeHist(frame[:])
    hsv = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)

    lower_pink = np.array([121, 14 , 178])
    upper_pink = np.array([166, 77, 255])

    pink = cv2.inRange(hsv, lower_pink, upper_pink)

    image_pink, contours_pink, heirarchy_pink = cv2.findContours(pink, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    area_pink = []

    for cnt_pink in contours_pink:
        approx = cv2.approxPolyDP(cnt_pink,0.05*cv2.arcLength(cnt_pink,True),True)
        rect_pink = cv2.minAreaRect(cnt_pink)
        box_pink = cv2.boxPoints(rect_pink)
        box_pink = np.int0(box_pink)
        ar_pink = cv2.contourArea(box_pink)
        area_pink.append(ar_pink)

    if(len(area_pink)>0):
        z = 0
        max = area_pink[0]
        for i in range(len(area_pink)):
            if (area_pink[i] > max):
                max = area_pink[i]
                z = i

        rect_pink_1 = cv2.minAreaRect(contours_pink[z])
        box_pink_1 = cv2.boxPoints(rect_pink_1)
        box_pink_1 = np.int0(box_pink_1)
        pink_moments = cv2.moments(box_pink_1)

        cx_pink = int(pink_moments['m10'] / pink_moments['m00'])
        cy_pink = int(pink_moments['m01'] / pink_moments['m00'])
        cent_pink[0][0] = cx_pink
        cent_pink[0][1] = cy_pink
        cv2.circle(frame,(cx_pink,cy_pink),2,(255,255,255),thickness = 2)
        frame = cv2.drawContours(frame, [box_pink_1], 0, (0, 255, 255), 2)

    cv2.imshow('pink',pink)
    cv2.imshow( 'frame',frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break



cv2.destroyAllWindows()
cap.release()