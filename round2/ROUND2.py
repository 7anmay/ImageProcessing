import cv2
import numpy as np
from scipy import ndimage
import math
import time
import serial

cv2.useOptimized()
cap = cv2.VideoCapture(0)


def marker_loc():
    cent_blue = np.zeros((1, 2), np.int32)
    cent_pink = np.zeros((1, 2), np.int32)

    while True:
        _, fortress = cap.read()
        hsv = cv2.cvtColor(fortress, cv2.COLOR_BGR2HSV)

        lower_blue = np.array([99, 130, 80])
        upper_blue = np.array([113, 255, 255])

        lower_pink = np.array([121, 50, 80])
        upper_pink = np.array([199, 254, 149])

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
                q = 0
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

for i in range(50):
    ret, arena = cap.read()


col = [(255,255,1),(0,0,254),(1,255,1),(254,0,0),(249,7,246)]

#GIVEN IMAGE

img = cv2.imread('IMG_1.jpg')
imgray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
imgray_invert = cv2.bitwise_not(imgray, dst=None)
ret,thresh = cv2.threshold(imgray_invert ,197 , 255 , cv2.THRESH_BINARY)

image, contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
roi = []

#EXTRACTING TEMPLATES

mom = []
for cnt in contours:
    approx = cv2.approxPolyDP(cnt, 0.01 * cv2.arcLength(cnt, True), True)
    x1, y1, w, h = cv2.boundingRect(cnt)
    y = cv2.moments(cnt)
    mom.append(y)
    roi.append(img[y1:(y1+h),x1:x1+w])

mom_cen = np.zeros((len(roi),2),np.uint32)

for i in range(len(roi)):
    cx = int(mom[i]['m10'] / mom[i]['m00'])
    cy = int(mom[i]['m01'] / mom[i]['m00'])
    mom_cen[i][0] = cx
    mom_cen[i][1] = cy

#SEQUENCING THEM ACCORDING TO SERIAL NUMBER

e_roi=[]
for u in range(len(col)):
    for i in range(len(roi)):
        color = img[mom_cen[i][1],mom_cen[i][0]]
        if (col[u][0] == color[0] and col[u][1] == color[1] and col[u][2] == color[2]):
            e_roi.append(roi[i])
            break

f_roi = []

for i in range(len(e_roi)):
    img_roi = cv2.cvtColor(e_roi[i], cv2.COLOR_BGR2GRAY)
    img_roi_invert = cv2.bitwise_not(img_roi, dst=None)
    ret, thresh_roi = cv2.threshold(img_roi_invert, 197, 255, cv2.THRESH_BINARY)
    f_roi.append(thresh_roi)


resize_roi = []
for j in range(len(f_roi)):
    resize_roi_temp = cv2.resize(f_roi[j], (80, 80), interpolation=cv2.INTER_LINEAR)
    resize_roi.append(resize_roi_temp)

kernel = np.ones((5,5),np.uint8)
closing_original_arena = cv2.morphologyEx(arena, cv2.MORPH_CLOSE, kernel)
arena_gray = cv2.cvtColor(closing_original_arena, cv2.COLOR_BGR2GRAY)
ret,thresh_arena = cv2.threshold(arena_gray,121,255,cv2.THRESH_BINARY)
closing_arena = cv2.morphologyEx(thresh_arena, cv2.MORPH_CLOSE, kernel)

image_areana,contours_arena,hierarchy_arena = cv2.findContours(closing_arena,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

arena_roi = []
moments = []
corners_1 = np.zeros((12,2),np.int32)
corners_2 = np.zeros((12,2),np.int32)
approx_contours = []
i = 0
contours_area = []

for cnt_a in contours_arena:
    approx_a = cv2.approxPolyDP(cnt_a, 0.03 * cv2.arcLength(cnt_a, True), True)
    area = cv2.contourArea(approx_a)
    if(area>300):
        corners_1[i][0] = approx_a[0][0][0]
        corners_1[i][1] = approx_a[0][0][1]
        corners_2[i][0] = approx_a[1][0][0]
        corners_2[i][1] = approx_a[1][0][1]
        approx_contours.append(approx_a)
        y = cv2.moments(approx_a)
        moments.append(y)
        i = i + 1
        x1,y1,w1,h1 = cv2.boundingRect(approx_a)
        arena_roi.append(closing_arena[y1:(y1 + h1), x1:(x1 + w1)])
        arena = cv2.rectangle(arena,(x1,y1),((x1+w1),(y1+h1)),(255,0,0),thickness=2)

def angle():
    x = corners_1[i][0] - corners_2[i][0]
    y = corners_1[i][1] - corners_2[i][1]
    if (x == 0):
        m = 0
        ang = 0
    else:
        m = y/float(x)
        ang = (math.atan(m) * 180 / np.pi)
    return ang

rotated_image = []
for i in range(len(arena_roi)):
    rot = angle()
    rotate_1 = ndimage.rotate(arena_roi[i],rot)
    blur = cv2.medianBlur(rotate_1, 15)
    ima, cont, hierarchy_1 = cv2.findContours(blur, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    roi_1 = []
    for cnt in cont:
        approx = cv2.approxPolyDP(cnt, 0.01 * cv2.arcLength(cnt, True), True)
        x, y, w, h = cv2.boundingRect(approx)
        rotate_1 = cv2.rectangle(rotate_1, (x, y), (x + w, y + h), (0, 255, 255), 2)
        roi_1.append(rotate_1[y:y + h, x:x + w])

    roi_1[0] = cv2.resize(roi_1[0], (80, 80), interpolation=cv2.INTER_LINEAR)
    rotated_image.append(roi_1[0])

mat_roi = []

for r in range(len(resize_roi)):
    mat = []
    sum = 0
    for i in range(20):
        for j in range(20):
            sum = 0
            for m in range(4):
                y = i * 4 + m
                for n in range(4):
                    x = j * 4 + n
                    px = resize_roi[r][y, x]
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

final_sequence = []
for r in range(len(resize_roi)):
    final_zeros = []
    for v in range(len(rotated_image)):
        zeros = []
        temp = rotated_image[v]
        for p in range(4):
            temp = ndimage.rotate(temp,90)
            blur = cv2.medianBlur(temp, 15)
            ima, cont, hierarchy_1 = cv2.findContours(blur, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            roi_2 = []
            for cnt in cont:
                approx = cv2.approxPolyDP(cnt, 0.01 * cv2.arcLength(cnt, True), True)
                x, y, w, h = cv2.boundingRect(approx)
                rotate = cv2.rectangle(temp, (x, y), (x + w, y + h), (0, 255, 255), 2)
                roi_2.append(temp [y:y + h, x:x + w])

            roi_2[0] = cv2.resize(roi_2[0],(80,80), interpolation=cv2.INTER_LINEAR)

            mat_temp = []
            sum = 0
            for i in range(20):
                for j in range(20):
                    sum = 0
                    for m in range(4):
                        y = i * 4 + m
                        for n in range(4):
                            x = j * 4 + n
                            px = roi_2[0][y, x]
                            if px == 255:
                                px = 1
                            else:
                                px = -1
                            sum = sum + px

                    if (sum < 0):
                        sum = -1
                    elif (sum > 0):
                        sum = 1
                    mat_temp.append(sum)

            a = []
            for k in range(400):
                y = mat_roi[r][k] - mat_temp[k]
                a.append(y)

            k = 0
            for i in range(400):
                if (a[i] == 0):
                    k = k + 1
                else:
                    continue
            zeros.append(k)

        y = len(zeros)
        max_zero = zeros[0]
        q=0
        for t in range(y):
            if(max_zero < zeros[t]):
                max_zero = zeros[t]
                q = t
            else:
                continue
        final_zeros.append(zeros[q])

    max_zero = final_zeros[0]
    w = 0
    for t in range(len(final_zeros)):
        if (max_zero < final_zeros[t]):
            max_zero = final_zeros[t]
            w = t
        else:
            continue
    final_sequence.append(w)


goal = np.zeros((7,2),np.uint32)

for i in range(len(final_sequence)):
    cx = int(moments[final_sequence[i]]['m10'] / moments[final_sequence[i]]['m00'])
    cy = int(moments[final_sequence[i]]['m01'] / moments[final_sequence[i]]['m00'])
    goal[i][0] = cx
    goal[i][1] = cy

print final_sequence


templates_contours = []
obstacles_contours = []


for i in final_sequence:
    tem = arena_roi[i]
    templates_contours.append(tem)

for i in range(len(arena_roi)):
    t = 0
    for j in final_sequence:

        if (i == j):
            t = 1
    if t == 0:
        obstacles_contours.append(approx_contours[i])


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


#############################################################################
for i in range(20):
    _ , arena = cap.read()
Y,X = arena.shape[:2]
blocksize = 40
grid_Y = int(Y/float(blocksize))
grid_X = int(X/float(blocksize))
grid = np.zeros((grid_Y,grid_X),np.int32)

grid[:] = -1
goal_grid = []
G_cent = []
neigh = []
prev_neigh = []
path = []
temp_neigh = []
cent =np.zeros((1,2),np.int32)
#cent[0][0] = 300
#cent[0][1] = 48
#prev_neigh.append([cent[0][0],cent[0][1]])
grid_ind = 0


def grid_cent(ind):
    G_cent = []
    G_X = (ind[0]+1) * blocksize
    G_Y = (ind[1]+1) * blocksize
    G_x = ind[0] * blocksize
    G_y = ind[1] * blocksize

    G_cent.append((G_X + G_x) / 2)
    G_cent.append((G_y + G_Y) / 2)
    return G_cent

def goalgrid(goal):
    for target in goal:
        goal_grid.append([(target[0] / blocksize),(target[1] / blocksize)])

def start_grid(curr_loc):
    start.append(curr_loc[0][0] / blocksize)
    start.append(curr_loc[0][1] / blocksize)

def obstacle(obs):
    for k in obs:
        for m in k:
            for n in m:
                o_gx = n[0] / blocksize
                o_gy = n[1] / blocksize
                if (o_gx < grid_X and o_gy < grid_Y):
                    grid[o_gy][o_gx] = -2

def draw_grid():
    #vertical lines
    for i in range(grid_X):
        cv2.line(arena,(i*blocksize,0),(i*blocksize,Y),(255,255,255),1)
    #horizontal lines
    for j in range (grid_Y):
        cv2.line(arena, (0,j * blocksize), (X,j * blocksize), (255, 255, 255), 1)

def neighbours(curr_loc, neigh_arr, ind):
    adj = [[0, 1], [1, 0], [-1, 0], [0, -1],[1, 1], [-1, -1], [1, -1], [-1, 1]]

    for i in adj:
        N_x = curr_loc[0] + i[0]
        N_y = curr_loc[1] + i[1]
        if (N_x < grid_X and N_y < grid_Y and N_x >= 0 and N_y >= 0 ):
            if(grid[N_y][N_x] == ind):
                neigh_arr.append([N_x, N_y])
###########################################################
#grid make
#######################################################################
#execution starts here
#######################################################################

for i in range(10):
    cent_blue, cent_pink = marker_loc()
    centroid(cent_pink, cent_blue)
    cv2.circle(arena,(cent[0][0],cent[0][1]),3,(255,255,255),thickness=2)
    cv2.imshow('arena',arena)
    if cv2.waitKey(1) & 0xFF == 27:
        break

goal[5][0] = cent[0][0]
goal[5][1] = cent[0][1]


ser = serial.Serial('COM40', 9600)
time.sleep(1)
draw_grid()
blink = 0
goalgrid(goal)
print 'goal =',goal

#print 'grid = ', grid
for targ in goal_grid:
    _ , fortress = cap.read()
    grid[:] = -1
    obstacle(obstacles_contours)
    blink = blink + 1
    pointer = 0
    grid[targ[1]][targ[0]] = -1
    path = []
    start = []
    prev = []
    prev_neigh = []
    cent_blue, cent_pink = marker_loc()
    centroid(cent_pink, cent_blue)
    start_grid(cent)
    prev_neigh.append([start[0], start[1]])
    grid[start[1]][start[0]] = 0

    while 1:
        neigh = []
        prev = prev_neigh.pop(0)
        neighbours(prev, neigh, -1)
        grid_ind = grid[prev[1]][prev[0]] + 1
        l = len(neigh)
        for j in range(l):
            curr_block = neigh.pop(0)

            if (curr_block == targ):
                grid[curr_block[1]][curr_block[0]] = grid_ind
                prev_neigh.append(curr_block)
                pointer = 1
                break

            grid[curr_block[1]][curr_block[0]] = grid_ind
            prev_neigh.append(curr_block)
            #print grid

        if pointer == 1:
            break
    #print 'grid = '
    #print grid
    x = prev_neigh.pop()
    path.append(x)
    q = 0
    for z in range(grid_ind - 1, -1, -1):
        temp_neigh = []
        neighbours(x, temp_neigh, z)

        print 'z = ', z
        x = temp_neigh.pop()
        path.append(x)

    path.reverse()
    print 'path = ', path
    #######################################################################
    # execution starts here
    #######################################################################
    pt1 = []
    pt2 = []
    for a in range(len(path) - 1):
        i1 = path[a]
        i2 = path[a + 1]
        pt1 = grid_cent(i1)
        pt2 = grid_cent(i2)
        cv2.line(fortress, (pt1[0], pt1[1]), (pt2[0], pt2[1]), (0, 255, 255), 2)
        cv2.imshow('path',fortress)
    #cv2.circle(arena, (goal[0], goal[1]), 10, (0, 0, 0), -1)
    #cv2.circle(arena, (cent[0], cent[1]), 10, (123, 255, 0), -1)
    #cv2.waitKey(0)
    cv2.destroyAllWindows

    for guide in path:
        while True:
            t = 0
            point = grid_cent(guide)
            cent_blue, cent_pink = marker_loc()
            _, fortress = cap.read()
            centroid(cent_pink, cent_blue)
            dis1 = distance(cent_pink, point)
            dis2 = distance(cent_blue, point)
            dis = distance(cent, point)
            ang = angle(cent_pink, cent_blue, point)
            dir = direction(ang, point)
            if dir == 'clockwise':
                dir = 'o'
            if dir == 'anticlockwise':
                dir = 'p'
            cv2.line(fortress, (cent_pink[0][0], cent_pink[0][1]), (cent_blue[0][0], cent_blue[0][1]), (0, 255, 255), 2)
            cv2.line(fortress, (cent[0][0], cent[0][1]), (point[0], point[1]), (0, 255, 255), 2)
            print 'dis =', dis
            print 'direction = ', dir
            print 'angle = ', ang

            while (math.fabs(ang) < 30):
                _, fortress = cap.read()
                cent_blue, cent_pink = marker_loc()
                centroid(cent_pink, cent_blue)
                dis1 = distance(cent_pink, point)
                dis2 = distance(cent_blue, point)
                ang = angle(cent_pink, cent_blue, point)
                dir = direction(ang, point)
                dist = distance(cent, point)
                ser.write('m')
                print 'movebot'
                print ang

                cv2.line(fortress, (cent_pink[0][0], cent_pink[0][1]), (cent_blue[0][0], cent_blue[0][1]), (255, 255, 255),thickness=3)
                cv2.line(fortress, (point[0], point[1]), (cent[0][0], cent[0][1]), (255, 255, 255), thickness=3)
                print 'distance= ', dist
                cv2.imshow('path',fortress)
                if (dist < 20):
                    t=1
                    ser.write('s')
                    break
            if t == 1:
                break
            ser.write(dir)
            cv2.line(fortress, (cent_pink[0][0], cent_pink[0][1]), (cent_blue[0][0], cent_blue[0][1]), (255, 255, 255),thickness=3)
            cv2.line(fortress, (point[0], point[1]), (cent[0][0], cent[0][1]), (255, 255, 255), thickness=3)
            cv2.imshow('path',fortress)
    ser.write('s')
    time.sleep(1)
    print('stopped for blinking')
    if (blink == 1):
        ser.write('1')
    if (blink == 2):
        ser.write('2')
    if (blink == 3):
        ser.write('3')
    if (blink == 4):
        ser.write('4')
    if (blink == 5):
        ser.write('b')


ser.write('s')
ser.close()