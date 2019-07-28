import cv2
import numpy as np
import math
import time
import serial

cap = cv2.VideoCapture(0)
##############################################################################################
# MARKER DETECTION


def marker_loc():
    cent_blue = np.zeros((1, 2), np.int32)
    cent_pink = np.zeros((1, 2), np.int32)

    time.sleep(2)
    while True:
        cap.set(cv2.CAP_PROP_CONTRAST,7)
        ret, frame = cap.read()
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        lower_blue = np.array([72, 131, 101])
        upper_blue = np.array([121, 226, 255])

        lower_pink = np.array([120, 66, 127])
        upper_pink = np.array([171, 165, 255])

        kernel_marker = np.ones((9, 9), np.uint8)

        blue = cv2.inRange(hsv, lower_blue, upper_blue)
        pink = cv2.inRange(hsv, lower_pink, upper_pink)

        blue_opening = cv2.morphologyEx(blue, cv2.MORPH_OPEN, kernel_marker)
        pink_opening = cv2.morphologyEx(pink, cv2.MORPH_OPEN, kernel_marker)

        image_blue, contours_blue, heirarchy_blue = cv2.findContours(blue_opening, cv2.RETR_TREE,
                                                                     cv2.CHAIN_APPROX_SIMPLE)

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


##############################################################################################
# TEMPLATE MATCHING

color = [(0,242,255),(76,177,34),(36,28,237),(204,72,63)]


for i in  range(20):
    cap.set(cv2.CAP_PROP_CONTRAST,8)
    ret, arena = cap.read()

#GIVEN IMAGE
img = cv2.imread('image_2.png')
imgray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
imgray_invert = cv2.bitwise_not(imgray, dst=None)
ret,thresh = cv2.threshold(imgray_invert ,10 , 255 , cv2.THRESH_BINARY)

image, contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
roi = []

#EXTRACTING TEMPLATES

for cnt in contours:
    approx = cv2.approxPolyDP(cnt, 0.01 * cv2.arcLength(cnt, True), True)
    x1, y1, w, h = cv2.boundingRect(cnt)
    img = cv2.rectangle(img, (x1, y1), (x1 + w, y1 + h), (255, 255, 255), 2)
    roi.append(img[y1:(y1+h),x1:x1+w])

#SEQUENCING THEM ACCORDING TO SERIAL NUMBER

e_roi=[]
for u in range(4):
    for k in range(4):
        t = 0
        rows, cols = roi[k].shape[:2]
        for i in range(rows):
            for j in range(cols):
                if (color[u][0] == roi[k][i][j][0] and color[u][1] == roi[k][i][j][1] and color[u][2] == roi[k][i][j][2]) :
                    e_roi.append(roi[k])
                    t=1
                    break
            if t==1:
                break
            else:
                continue
f_roi = []
for i in range(4):
    img_roi = cv2.cvtColor(e_roi[i], cv2.COLOR_BGR2GRAY)
    img_roi_invert = cv2.bitwise_not(img_roi, dst=None)
    ret, thresh_roi = cv2.threshold(img_roi_invert, 10, 255, cv2.THRESH_BINARY)
    f_roi.append(thresh_roi)

#ARENA

kernel = np.ones((5,5),np.uint8)

closing_original_arena = cv2.morphologyEx(arena, cv2.MORPH_CLOSE, kernel)
arena_gray = cv2.cvtColor(closing_original_arena, cv2.COLOR_BGR2GRAY)
ret,thresh_arena = cv2.threshold(arena_gray,140,255,cv2.THRESH_BINARY)
closing_arena = cv2.morphologyEx(thresh_arena, cv2.MORPH_CLOSE, kernel)

#EXTRACTING TEMPLATE FROM ARENA

image_areana,contours_arena,hierarchy_arena = cv2.findContours(closing_arena,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

loc = []
moments = []
arena_roi = []

for cnt_a in contours_arena:
    approx_a = cv2.approxPolyDP(cnt_a, 0.05 * cv2.arcLength(cnt_a, True), True)
    x2,y2,w1,h1 = cv2.boundingRect(cnt_a)
    y = cv2.moments(cnt_a)
    moments.append(y)
    arena = cv2.rectangle(arena, (x2, y2), (x2 + w1, y2 + h1), (0, 255, 0), 2)
    loc.append(x2)
    loc.append(y2)
    arena_roi.append(closing_arena[y2:(y2+h1),x2:(x2+w1)])

#RESIZING ALL TEMPLATE TO ONE SIZE

resize_arena = []

for i in range(len(arena_roi)):
    resize_arena_temp = cv2.resize(arena_roi[i], (80, 80), interpolation=cv2.INTER_LINEAR)
    resize_arena.append(resize_arena_temp)

resize_roi = []
for j in range(len(f_roi)):
    resize_roi_temp = cv2.resize(f_roi[j], (80, 80), interpolation=cv2.INTER_LINEAR)
    resize_roi.append(resize_roi_temp)

# SCANNING EACH AND EVERY PIXEL OF IMAGE TEMPLATES

mat_roi = []
for k in range(len(resize_roi)):
    scan = resize_roi[k]
    mat = []
    sum = 0
    for i in range(20):
        for j in range(20):
            sum = 0
            for m in range(4):
                y = i * 4 + m
                for n in range(4):
                    x = j * 4 + n
                    px = scan[y, x]
                    if px == 255:
                        px = 1
                    else:
                        px = -1
                    sum = sum + px
            if (sum < 0):
                sum = -1
            elif (sum > 0):
                sum = 1
            mat.append(sum)
    mat_roi.append(mat)

# SCANNING EACH AND EVERY PIXEL OF ARENA TEMPLATES

mat_arena = []
for k in range(len(resize_arena)):
    scan = resize_arena[k]
    mat = []
    sum = 0
    for i in range(20):
        for j in range(20):
            sum = 0
            for m in range(4):
                y = i * 4 + m
                for n in range(4):
                    x = j * 4 + n
                    px = scan[y, x]
                    if px == 255:
                        px = 1
                    else:
                        px = -1
                    sum = sum + px
            if (sum < 0):
                sum = -1
            elif (sum > 0):
                sum = 1
            mat.append(sum)
    mat_arena.append(mat)


finalsequence = []
for i in range(len(mat_roi)):
    diff = []
    for l in range(len(mat_arena)):
        temp = []
        for k in range(400):
            y = mat_roi[i][k] - mat_arena[l][k]
            temp.append(y)
        diff.append(temp)

    zeros = []
    for m in range(len(diff)):
        p= 0
        for b in range(400):
            if(diff[m][b] == 0):
                p=p+1
            else:
                continue
        zeros.append(p)
    max_zero = zeros[0]
    q=0
    for t in range(len(zeros)):
        if(max_zero < zeros[t]):
            max_zero = zeros[t]
            q = t
        else:
            continue
    finalsequence.append(q)

goal = np.zeros((len(finalsequence),2),np.uint32)

for i in range(len(finalsequence)):
    cx = int(moments[finalsequence[i]]['m10'] / moments[finalsequence[i]]['m00'])
    cy = int(moments[finalsequence[i]]['m01'] / moments[finalsequence[i]]['m00'])
    goal[i][0] = cx
    goal[i][1] = cy

# TEMPLATE END
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
        m1 = (cent_1[0][1] - cent_2[0][1]) / float(cent_1[0][0] - cent_2[0][0])
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
            if (dis1 >= dis2):
                if (angle > 0):
                    angle = (180 - angle) * (-1)

                elif (angle <= 0):
                    angle = (math.fabs(angle) - 180) * (-1)
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
##############################################################################################


##############################################################################################
# main execution starts here

ser = serial.Serial('COM40',9600, timeout = 2)
time.sleep(2)

t = 0
blink = 1

for i in goal:
    t = 0
    while(True):
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
        while(math.fabs(ang)<=20):
            _, check = cap.read()
            cent_blue, cent_pink = marker_loc()
            centroid(cent_pink, cent_blue)
            dis1 = distance(cent_pink, i)
            dis2 = distance(cent_blue, i)
            ang = angle(cent_pink, cent_blue, i)
            dir = direction(ang,i)
            dist = distance(cent, i)
            ser.write('m')
            print 'movebot'
            print ang
            cv2.line(check, (cent_pink[0][0], cent_pink[0][1]), (cent_blue[0][0], cent_blue[0][1]), (255, 255, 255),thickness=3)
            cv2.line(check, (i[0], i[1]), (cent[0][0], cent[0][1]), (255, 255, 255), thickness=3)
            print 'distance= ',dist
            if (dist<50):
                ser.write('s')
                print 'stop'
                t = 1
                time.sleep(500/1000.0)
                if (blink == 1):
                    ser.write('w')
                    time.sleep(blink)
                elif (blink == 2):
                    ser.write('x')
                elif (blink == 3):
                    ser.write('y')
                elif (blink == 4):
                    ser.write('z')
                ser.write('s')
                print 'blinked', blink

                cv2.imshow('process', check)
                if cv2.waitKey(1) == 27 :
                    break
                break

        cv2.line(check, (cent_pink[0][0], cent_pink[0][1]), (cent_blue[0][0], cent_blue[0][1]), (255, 255, 255), thickness=3)
        cv2.line(check, (i[0], i[1]), (cent[0][0], cent[0][1]), (255, 255, 255), thickness=3)

        ser.write('s')
        time.sleep(100/1000.0)
        ser.write(dir)
        time.sleep(100/1000.0)
        ser.write('s')


        if (t == 1):
            blink = blink + 1
            print 'blink increment'
            print blink
            break

        cv2.imshow('process', check)
        if cv2.waitKey(1) == 27:
            break

ser.write('e')
print 'end'
cap.release()
cv2.destroyAllWindows()
#ROUND 1 END
########################################################################################