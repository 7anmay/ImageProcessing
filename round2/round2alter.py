import cv2
import numpy as np
import math
import time
import serial
cap = cv2.VideoCapture(0)
# MARKER DETECTION
def marker_loc():
    cent_blue = np.zeros((1, 2), np.int32)
    cent_pink = np.zeros((1, 2), np.int32)

    while True:
        ret, frame = cap.read()
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        lower_blue = np.array([84, 43, 60])
        upper_blue = np.array([121, 211, 201])

        lower_pink = np.array([114, 76, 100])
        upper_pink = np.array([181, 175, 213])

        kernel_marker = np.ones((9, 9), np.uint8)

        blue = cv2.inRange(hsv, lower_blue, upper_blue)
        pink = cv2.inRange(hsv, lower_pink, upper_pink)

        blue_opening = cv2.morphologyEx(blue, cv2.MORPH_OPEN, kernel_marker)
        pink_opening = cv2.morphologyEx(pink, cv2.MORPH_OPEN, kernel_marker)

        image_blue, contours_blue, heirarchy_blue = cv2.findContours(blue_opening, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

        area_blue = []
        for cnt in contours_blue:
            approx = cv2.approxPolyDP(cnt, 0.05 * cv2.arcLength(cnt, True), True)
            rect = cv2.minAreaRect(cnt)
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            ar_blue = cv2.contourArea(box)
            area_blue.append(ar_blue)
            if (len(area_blue) > 0):
                q = 0;
                max_blue = area_blue[0]
                for i in range(len(area_blue)):
                    if (area_blue[i] > max_blue):
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
            else:
                pass

        image_pink, contours_pink, heirarchy_pink = cv2.findContours(pink_opening, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        area_pink = []

        for cnt_pink in contours_pink:
            approx = cv2.approxPolyDP(cnt_pink, 0.05 * cv2.arcLength(cnt_pink, True), True)
            rect_pink = cv2.minAreaRect(cnt_pink)
            box_pink = cv2.boxPoints(rect_pink)
            box_pink = np.int0(box_pink)
            ar_pink = cv2.contourArea(box_pink)
            area_pink.append(ar_pink)

        if (len(area_pink) > 0):
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

        #cv2.imshow('frame',frame)
        if cv2.waitKey(1):
            break

    return cent_blue , cent_pink

#MARKER END
#############################################################################################

#############################################################################################
#POINT TO POINT TRAVERSAL
cent =np.zeros((1,2),np.int32)
def centroid(cent_1,cent_2):
    x =int((cent_1[0][0]+cent_2[0][0])/2)
    y = int((cent_1[0][1]+cent_2[0][1])/2)
    cent[0][0]=x
    cent[0][1]=y


def distance(centr,goal):
    #finding euclidean distance
    dis = math.sqrt(math.pow((centr[0][0]-goal[0]),2) + math.pow((centr[0][1]-goal[1]),2))
    return dis


def angle(cent_1,cent_2,goal):

    if (int(cent_1[0][0] - cent_2[0][0]) == 0 and not(int((cent_1[0][0] + cent_2[0][0]) / float(2)) - goal[0]) == 0):
        m2 = (((cent_1[0][1] + cent_2[0][1]) / float(2)) - goal[1]) / float(((cent_1[0][0] + cent_2[0][0]) / float(2)) - goal[0])
        angle = math.atan(m2)*180/np.pi
        if (angle>0):
            angle = 90 - angle
            if (dis1>dis2):
                angle = angle - 180
        elif (angle<0):
            angle = math.fabs(angle) - 90
            if (dis1>dis2):
                angle = 180 - math.fabs(angle)


    elif(int(((cent_1[0][0] + cent_2[0][0]) / float(2)) - goal[0]) == 0 and not((cent_1[0][0] - cent_2[0][0]) == 0)):
        m1 = (cent_1[0][1] - cent_2[0][1]) / float           (cent_1[0][0] - cent_2[0][0])
        angle = (math.atan(m1)*180/np.pi)
        if (angle>0):
            angle = 90-angle
            if (dis1>=dis2):
                angle = (180 - angle)* (-1)
        elif (angle<=0):
            angle = math.fabs(angle) - 90
            if (dis1>dis2):
                angle = 180 - math.fabs(angle)

    elif(int(cent_1[0][0] - cent_2[0][0]) == 0 and int(((cent_1[0][0] + cent_2[0][0]) / float(2)) - goal[0]) == 0):
        if (dis1<dis2):
            angle = 1
        else:
            angle = 180

    else:

        m1 = (cent_1[0][1] - cent_2[0][1]) / float(cent_1[0][0] - cent_2[0][0])
        m2 = (((cent_1[0][1] + cent_2[0][1]) / float(2)) - goal[1]) / float(((cent_1[0][0] + cent_2[0][0]) / float(2)) - goal[0])

        ang1 = math.atan(m1) * 180 / np.pi
        ang2 = math.atan(m2) * 180 / np.pi

        # when bot ang - and goal ang +
        if (ang1 < 0 and ang2 > 0):
            print 'cond6'
            if (cent_1[0][1] < cent_2[0][1]):
                if (cent[0][0] > goal[0]):
                    angle = 180 - ang2 - math.fabs(ang1)
                elif (cent[0][0] < goal[0]):
                    angle = (ang2 + math.fabs(ang1)) * (-1)
            else:
                if (cent[0][0] < goal[0]):
                    angle = 180 - ang2 - math.fabs(ang1)

                elif (cent[0][0] > goal[0]):
                    angle = (ang2 + math.fabs(ang1)) * (-1)

        # when bot ang + and goal ang -
        elif (ang1 > 0 and ang2 < 0):
            print 'cond7'
            if (cent_1[0][1] < cent_2[0][1]):
                if (cent[0][0] < goal[0]):
                    angle = (180 - ang1 - math.fabs(ang2)) * (-1)
                elif (cent[0][0] > goal[0]):
                    angle = ang1 + math.fabs(ang2)

            else:
                if (cent[0][0] > goal[0]):
                    angle = (180 - ang1 - math.fabs(ang2)) * (-1)
                elif (cent[0][0] < goal[0]):
                    angle = ang1 + math.fabs(ang2)

        else:
            angle = (math.atan(m1) - math.atan(m2)) * 180 / np.pi
            print 'cond6'
            if (dis1 >= dis2):
                if (angle > 0):
                    print 'cond7'
                    angle = (180 - angle) * (-1)

                elif (angle <= 0):
                    angle = (math.fabs(angle) - 180) * (-1)
                    print 'cond8'
    return angle

def direction(angle,goal):

    if ((not(cent_pink[0][0] - cent_blue[0][0]) == 0) and (int((cent_pink[0][0] + cent_blue[0][0]) / float(2)) - goal[0]) == 0):
        if angle<0:
            dir = 'anticlockwise'
        else:
            dir = 'clockwise'
    else:
        if angle<0:
            dir = 'clockwise'
        else:
            dir = 'anticlockwise'
    return dir

########################################################################################
#TRAVERSE END


goal = [[223, 198],[ 13, 361],[480, 270]]

####################################last position_appens

cent_blue, cent_pink = marker_loc()
centroid(cent_pink, cent_blue)
ser = serial.Serial('COM40',9600)
time.sleep(2)
####################################
t = 0


blink = 1


'''add goal and blink'''



for i in goal:
    t = 0
    while(True):
        m = 0
        _,check = cap.read()
        cent_blue, cent_pink = marker_loc()
        centroid(cent_pink, cent_blue)
        dis1 = distance(cent_pink,i)
        dis2 = distance(cent_blue, i)
        ang = angle(cent_pink, cent_blue, i)
        dir = direction(ang,i)
        dist = distance(cent,i)
        cv2.line(check, (cent_pink[0][0], cent_pink[0][1]), (cent_blue[0][0], cent_blue[0][1]), (255, 255, 255),thickness=3)
        cv2.line(check, (i[0],i[1]), (cent[0][0], cent[0][1]), (255, 255, 255), thickness=3)
        print 'angle=',ang ,'direction=',dir
        if (dir == 'clockwise'):
            dir = 'o'
        else:
            dir = 'p'
        if (dist < 30):
            ser.write('s')
            print 'stop'
            t = 1
            time.sleep(100/1000.0)
            if (blink == 1):
                ser.write('w')

            elif (blink == 2):
                ser.write('x')

            elif (blink == 3):
                ser.write('y')

            elif (blink == 4):
                ser.write('z')


            ser.write('s')
            print 'blinked', blink



        while(math.fabs(ang)<22):
            _, check = cap.read()
            cent_blue, cent_pink = marker_loc()
            centroid(cent_pink, cent_blue)
            dis1 = distance(cent_pink, i)
            dis2 = distance(cent_blue, i)
            ang = angle(cent_pink, cent_blue, i)
            #dir = direction(ang,i,cen`)
            dist = distance(cent, i)
            if m == 0:
                ser.write('a')
                m=1
            ser.write('m')
            print 'movebot'
            print ang

            cv2.line(check, (cent_pink[0][0], cent_pink[0][1]), (cent_blue[0][0], cent_blue[0][1]), (255, 255, 255),thickness=3)
            cv2.line(check, (i[0], i[1]), (cent[0][0], cent[0][1]), (255, 255, 255), thickness=3)
            print 'distance= ',dist
            cv2.imshow('cehe', check)
            if cv2.waitKey(1) & 0xFF == 27:
                break

            if (dist<30):
                ser.write('s')
                print 'stop'
                t = 1
                time.sleep(100 / 1000.0)
                if (blink == 1):
                    ser.write('w')

                elif (blink == 2):
                    ser.write('x')

                elif (blink == 3):
                    ser.write('y')

                elif (blink == 4):
                    ser.write('z')


                ser.write('s')
                print 'blinked', blink
                break
        ser.write(dir)


        cv2.imshow('cehe',check)
        if cv2.waitKey(1) & 0xFF == 27:
            break
        if (t == 1):
            blink = blink + 1
            print blink
            break
ser.write('s')
print 'end'
cap.release()
cv2.destroyAllWindows()
#ROUND 2 END
###########################