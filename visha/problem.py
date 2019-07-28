import cv2
import numpy as np
import math
from scipy import ndimage


img = cv2.imread('image_2.png')
imgray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
imgray_invert = cv2.bitwise_not(imgray, dst=None)
ret,thresh = cv2.threshold(imgray_invert ,10 , 255 , cv2.THRESH_BINARY)

image, contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
roi = []

for cnt in contours:
    approx = cv2.approxPolyDP(cnt, 0.01 * cv2.arcLength(cnt, True), True)
    x, y, w, h = cv2.boundingRect(cnt)
    img = cv2.rectangle(thresh, (x, y), (x + w, y + h), (255, 255, 255), 2)
    roi.append(thresh[y:(y+h),x:x+w])

resize_roi = []
for j in range(4):
    resize_roi_temp = cv2.resize(roi[j], (80, 80), interpolation=cv2.INTER_LINEAR)
    resize_roi.append(resize_roi_temp)

kernel = np.ones((5,5),np.uint8)
arena = cv2.imread('image_rotate.png')
closing_original_arena = cv2.morphologyEx(arena, cv2.MORPH_CLOSE, kernel)
arena_gray = cv2.cvtColor(closing_original_arena, cv2.COLOR_BGR2GRAY)
ret,thresh_arena = cv2.threshold(arena_gray,200,255,cv2.THRESH_BINARY)
closing_arena = cv2.morphologyEx(thresh_arena, cv2.MORPH_CLOSE, kernel)


image_areana,contours_arena,hierarchy_arena = cv2.findContours(closing_arena,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

arena_roi = []

for cnt_a in contours_arena:
    approx_a = cv2.approxPolyDP(cnt_a, 0.03 * cv2.arcLength(cnt_a, True), True)
    x1,y1,w1,h1 = cv2.boundingRect(cnt_a)
    clos = cv2.rectangle(arena,(x1,y1),(x1+w,y1+h1),(255,255,255),2)
    arena_roi.append(closing_arena[y1:(y1+h1),x1:(x1+w1)])

zeros = []
for i in range(360):
    x = i+1
    rotate = ndimage.rotate(arena_roi[2],x)
    blur = cv2.medianBlur(rotate, 15)
    ima, cont, hierarchy_1 = cv2.findContours(blur, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    roi_1 = []
    for cnt in cont:
        approx = cv2.approxPolyDP(cnt, 0.01 * cv2.arcLength(cnt, True), True)
        x, y, w, h = cv2.boundingRect(approx)
        rotate = cv2.rectangle(rotate, (x, y), (x + w, y + h), (0, 255, 255), 2)
        roi_1.append(rotate[y:y + h, x:x + w])
    roi_1[0] = cv2.resize(roi_1[0], (80, 80), interpolation=cv2.INTER_LINEAR)

    mat = []
    sum = 0
    for i in range(20):
        for j in range(20):
            sum = 0
            for m in range(4):
                y = i * 4 + m
                for n in range(4):
                    x = j * 4 + n
                    px = resize_roi[1][y, x]
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

    mat_temp = []
    sum = 0
    for i in range(20):
        for j in range(20):

            sum = 0

            for m in range(4):
                y = i * 4 + m
                for n in range(4):
                    x = j * 4 + n
                    px = roi_1[0][y, x]
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
        y = mat[k] - mat_temp[k]
        a.append(y)

    k = 0
    for i in range(400):
        if (a[i] == 0):
            k = k + 1
        else:
            continue
    zeros.append(k)


y =  len(zeros)
max_zero = zeros[0]
q=0
for t in range(y):
    if(max_zero < zeros[t]):
        max_zero = zeros[t]
        q = t
    else:
        continue

print zeros[q]
print q


cv2.imshow('csd',roi_1[0])
cv2.imshow('test',resize_roi[1])
cv2.waitKey(0)
cv2.destroyAllWindows()
