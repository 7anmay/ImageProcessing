import cv2
import numpy as np
import math


color = [(0,242,255),(204,72,63),(76,177,34),(36,28,237)]

img = cv2.imread('image_2.png')
imgray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
imgray_invert = cv2.bitwise_not(imgray, dst=None)
ret,thresh = cv2.threshold(imgray_invert ,10 , 255 , cv2.THRESH_BINARY)

image, contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
roi = []

for cnt in contours:
    approx = cv2.approxPolyDP(cnt, 0.01 * cv2.arcLength(cnt, True), True)
    x, y, w, h = cv2.boundingRect(cnt)
    img = cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 255), 2)
    roi.append(img[y:(y+h),x:x+w])

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

resize_roi = []
for j in range(4):
    resize_roi_temp = cv2.resize(f_roi[j], (80, 80), interpolation=cv2.INTER_LINEAR)
    resize_roi.append(resize_roi_temp)


kernel = np.ones((5,5),np.uint8)
arena = cv2.imread('image_3.png')
closing_original_arena = cv2.morphologyEx(arena, cv2.MORPH_CLOSE, kernel)
arena_gray = cv2.cvtColor(closing_original_arena, cv2.COLOR_BGR2GRAY)
ret,thresh_arena = cv2.threshold(arena_gray,100,255,cv2.THRESH_BINARY)
closing_arena = cv2.morphologyEx(thresh_arena, cv2.MORPH_CLOSE, kernel)


image_areana,contours_arena,hierarchy_arena = cv2.findContours(closing_arena,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
loc = []
moments = []
arena_roi = []

approx_r = []

for cnt_a in contours_arena:
    approx_a = cv2.approxPolyDP(cnt_a, 0.03 * cv2.arcLength(cnt_a, True), True)
    approx_r.append(approx_a)
    x1,y1,w1,h1 = cv2.boundingRect(cnt_a)
    arena_roi.append(closing_arena[y1:(y1+h1),x1:(x1+w1)])
    rect = cv2.minAreaRect(cnt_a)
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    loc.append(box)

#print approx_a[6]


'''
for i in range(9):

    print loc[i][3][0] , loc[i][3][1]
    print loc[i][0][0] , loc[i][0][1]
'''
for l in range(4):
    print loc[6][l][0] , loc[l][3][1]

x = loc[5][3][0] - loc[5][0][0]
y = -(loc[5][3][1] - loc[5][0][1])
slope = y/float(x)
print x,y ,slope
angle = math.atan(slope)

print (angle*180)/float(3.14)
#print ("%.2f" % angle)
print (1.57 - angle)*180/float(3.14)
#arena_2 = cv2.drawContours(closing_arena,[loc[6]],0,(255,255,255),2)
#cv2.imshow('ar',arena_roi[0])
#cv2.imshow('arasfena',arena_roi[1])
#cv2.imshow('arenasfa',arena_roi[2])
#cv2.imshow('arensasfa',arena_roi[3])
#v2.imshow('arehgfna',arena_roi[4])
#cv2.imshow('arewena',arena_roi[5])
#cv2.imshow('arenta',arena_roi[6])
#cv2.imshow('arenwa',arena_roi[7])
#cv2.imshow('arenua',arena_roi[8])

#cv2.imwrite('vish.png',arena_roi[6])
cv2.imshow('vissha',resize_roi[0])
cv2.imwrite('test_1.png',resize_roi[0])
#cv2.imshow('vishal',arena_2)
#cv2.imshow('vsdn',closing_arena)
cv2.waitKey(0)
cv2.destroyAllWindows()