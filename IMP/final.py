import cv2
import numpy as np
from scipy import ndimage

cv2.useOptimized()
color = [(0,242,255),(204,72,63),(76,177,34),(36,28,237)]


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


resize_roi = []
for j in range(len(f_roi)):
    resize_roi_temp = cv2.resize(f_roi[j], (80, 80), interpolation=cv2.INTER_LINEAR)
    resize_roi.append(resize_roi_temp)

kernel = np.ones((5,5),np.uint8)
arena = cv2.imread('image_rotate.png')
closing_original_arena = cv2.morphologyEx(arena, cv2.MORPH_CLOSE, kernel)
arena_gray = cv2.cvtColor(closing_original_arena, cv2.COLOR_BGR2GRAY)
ret,thresh_arena = cv2.threshold(arena_gray,200,255,cv2.THRESH_BINARY)
closing_arena = cv2.morphologyEx(thresh_arena, cv2.MORPH_CLOSE, kernel)

image_areana,contours_arena,hierarchy_arena = cv2.findContours(closing_arena,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
arr = []
h_w = []
arena_roi = []
moments = []
for cnt_a in contours_arena:
    approx_a = cv2.approxPolyDP(cnt_a, 0.03 * cv2.arcLength(cnt_a, True), True)
    x1,y1,w1,h1 = cv2.boundingRect(cnt_a)
    arr.append([x1,y1])
    h_w.append([h1,w1])
    y = cv2.moments(cnt_a)
    moments.append(y)
    arena_roi.append(closing_arena[y1:(y1+h1),x1:(x1+w1)])

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
for r in range(len(f_roi)):
    final_zeros = []
    for v in range(len(arena_roi)):
        zeros = []
        for p in range(360):
            e = p+1
            rotate = ndimage.rotate(arena_roi[v],e)
            blur = cv2.medianBlur(rotate, 15)
            ima, cont, hierarchy_1 = cv2.findContours(blur, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            roi_1 = []
            for cnt in cont:
                approx = cv2.approxPolyDP(cnt, 0.01 * cv2.arcLength(cnt, True), True)
                x, y, w, h = cv2.boundingRect(approx)
                rotate = cv2.rectangle(rotate, (x, y), (x + w, y + h), (0, 255, 255), 2)
                roi_1.append(rotate[y:y + h, x:x + w])
            roi_1[0] = cv2.resize(roi_1[0], (80, 80), interpolation=cv2.INTER_LINEAR)

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

goal = np.zeros((len(final_sequence),2),np.uint32)

for i in range(len(final_sequence)):
    cx = int(moments[final_sequence[i]]['m10'] / moments[final_sequence[i]]['m00'])
    cy = int(moments[final_sequence[i]]['m01'] / moments[final_sequence[i]]['m00'])
    goal[i][0] = cx
    goal[i][1] = cy

print goal

cv2.imshow('test1',arena_roi[final_sequence[0]])
cv2.imshow('test2',arena_roi[final_sequence[1]])
cv2.imshow('test3',arena_roi[final_sequence[2]])
cv2.imshow('test4',arena_roi[final_sequence[3]])
cv2.waitKey(0)
cv2.destroyAllWindows()