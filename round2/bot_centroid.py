import cv2
import numpy as np
import time
import math



cap =cv2.VideoCapture(0)
cent_blue = np.zeros((1,2), np.uint32)
cent_pink = np.zeros((1,2), np.uint32)
cent = np.zeros((1,2),np.uint32)

def centroid(cent_1,cent_2):
    x = (cent_1[0][0]+cent_2[0][0])/(2)
    y = (cent_1[0][1]+cent_2[0][1])/(2)
    cent[0][0]= x
    cent[0][1]= y

while True:
    ret,frame = cap.read()
    hsv = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)

    lower_blue = np.array([99, 130, 80])
    upper_blue = np.array([113, 255, 255])

    lower_pink = np.array([121, 50, 80])
    upper_pink = np.array([199, 254, 149])
    kernel = np.ones((9,9),np.uint8)

    blue = cv2.inRange(hsv, lower_blue, upper_blue)
    pink = cv2.inRange(hsv, lower_pink, upper_pink)

    blue_opening = cv2.morphologyEx(blue,cv2.MORPH_OPEN,kernel)
    pink_opening = cv2.morphologyEx(pink, cv2.MORPH_OPEN,kernel)



    image_blue,contours_blue,heirarchy_blue = cv2.findContours(blue_opening,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

    area_blue = []
    for cnt in contours_blue:
        approx = cv2.approxPolyDP(cnt,0.05*cv2.arcLength(cnt,True),True)
        rect = cv2.minAreaRect(cnt)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        ar_blue = cv2.contourArea(box)
        area_blue.append(ar_blue)
        if (len(area_blue) > 0):
            q = 0
            max = area_blue[0]
            for i in range(len(area_blue)):
                if (area_blue[i] > max):
                    max_blue = area_blue[i]
                    q = i

            rect_blue_1 = cv2.minAreaRect(contours_blue[q])
            box_blue_1 = cv2.boxPoints(rect_blue_1)
            box_blue_1 = np.int0(box_blue_1)
            blue_moments = cv2.moments(box_blue_1)
            cx = int(blue_moments['m10'] / blue_moments['m00'])
            cy = int(blue_moments['m01'] / blue_moments['m00'])
            cent_blue[0][0] = cx
            cent_blue[0][1] = cy
            #frame = cv2.drawContours(frame, [box_blue_1], 0, (0, 255, 255), 2)
            cv2.circle(frame, (cx, cy), 2, (255, 255, 255), thickness=2)
        else:
            pass

    image_pink, contours_pink, heirarchy_pink = cv2.findContours(pink_opening, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

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


    centroid(cent_blue,cent_pink)

    print cent

    cv2.line(frame,(cent_pink[0][0],cent_pink[0][1]),(cent_blue[0][0],cent_blue[0][1]),(255,255,255),thickness=3)
    cv2.line(frame,(489,97), (cent[0][0], cent[0][1]), (255, 255, 255),thickness=3)

    cv2.imshow('pink',pink)
    cv2.imshow( 'frame',frame)
    cv2.imshow('blue',blue)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cv2.destroyAllWindows()
cap.release()