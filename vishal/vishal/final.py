import cv2
import numpy as np

color = [(0,242,255),(204,72,63),(76,177,34),(36,28,237)]

img = cv2.imread('image_2.png')



cv2.namedWindow('img',cv2.WINDOW_NORMAL)
imgray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
imgray_invert = cv2.bitwise_not(imgray, dst=None)
ret,thresh = cv2.threshold(imgray_invert ,10 , 255 , cv2.THRESH_BINARY)

image, contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
roi = []

for cnt in contours:
    approx = cv2.approxPolyDP(cnt, 0.01 * cv2.arcLength(cnt, True), True)
    x1, y1, w, h = cv2.boundingRect(cnt)
    img = cv2.rectangle(img, (x1, y1), (x1 + w, y1 + h), (255, 255, 255), 2)
    roi.append(img[y1:(y1+h),x1:x1+w])

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


kernel = np.ones((5,5),np.uint8)
arena = cv2.imread('image_1.png')
closing_original_arena = cv2.morphologyEx(arena, cv2.MORPH_CLOSE, kernel)
arena_gray = cv2.cvtColor(closing_original_arena, cv2.COLOR_BGR2GRAY)
ret,thresh_arena = cv2.threshold(arena_gray,100,255,cv2.THRESH_BINARY)
closing_arena = cv2.morphologyEx(thresh_arena, cv2.MORPH_CLOSE, kernel)

image_areana,contours_arena,hierarchy_arena = cv2.findContours(closing_arena,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
loc = []
moments = []
arena_roi = []
for cnt_a in contours_arena:
    approx_a = cv2.approxPolyDP(cnt_a, 0.05 * cv2.arcLength(cnt_a, True), True)
    x2,y2,w1,h1 = cv2.boundingRect(cnt_a)
    y = cv2.moments(cnt_a)
    moments.append(y)
    arena = cv2.rectangle(closing_arena, (x2, y2), (x2 + w1, y2 + h1), (0, 255, 0), 2)
    loc.append(x2)
    loc.append(y2)
    arena_roi.append(closing_arena[y2:(y2+h1),x2:(x2+w1)])

print len(moments)
resize_arena = []
for i in range(4):
    resize_arena_temp = cv2.resize(arena_roi[i+2], (80, 80), interpolation=cv2.INTER_LINEAR)
    resize_arena.append(resize_arena_temp)

resize_roi = []
for j in range(4):
    resize_roi_temp = cv2.resize(f_roi[j], (80, 80), interpolation=cv2.INTER_LINEAR)
    resize_roi.append(resize_roi_temp)




mat_roi = []
for k in range(4):
    vishal = resize_roi[k]
    mat = []
    sum = 0
    for i in range(20):
        for j in range(20):
            sum = 0
            for m in range(4):
                y = i * 4 + m
                for n in range(4):
                    x = j * 4 + n
                    px = vishal[y, x]
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


mat_arena = []
for k in range(4):
    vishal = resize_arena[k]
    mat = []
    sum = 0
    for i in range(20):
        for j in range(20):
            sum = 0
            for m in range(4):
                y = i * 4 + m
                for n in range(4):
                    x = j * 4 + n
                    px = vishal[y, x]
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
for i in range(4):
    diff = []
    for l in range(4):
        temp = []
        for k in range(400):
            y = mat_roi[i][k] - mat_arena[l][k]
            temp.append(y)
        diff.append(temp)

    zeros = []
    for m in range(4):
        p= 0
        for b in range(400):
            if(diff[m][b] == 0):
                p=p+1
            else:
                continue
        zeros.append(p)
    max_zero = zeros[0]
    q=0
    for t in range(4):
        if(max_zero < zeros[t]):
            max_zero = zeros[t]
            q = t
        else:
            continue
    finalsequence.append(q)

x_loc = []
y_loc = []

for i in range(4):
    cx = int(moments[finalsequence[i]+2]['m10'] / moments[finalsequence[i]+2]['m00'])
    cy = int(moments[finalsequence[i]+2]['m01'] / moments[finalsequence[i]+2]['m00'])
    x_loc.append(cx)
    y_loc.append(cy)

for m in range(4):
    print  m+1 ,'match', x_loc[m] , y_loc[m]

cv2.imshow('img',img)
cv2.imshow('arena',arena)
cv2.imshow('match1',resize_arena[finalsequence[0]])
cv2.imshow('match2',resize_arena[finalsequence[1]])
cv2.imshow('match3',resize_arena[finalsequence[2]])
cv2.imshow('match4',resize_arena[finalsequence[3]])
cv2.imshow('match_roi1',resize_roi[0])
cv2.imshow('match_roi2',resize_roi[1])
cv2.imshow('match_roi3',resize_roi[2])
cv2.imshow('match_roi4',resize_roi[3])

cv2.waitKey(0)
cv2.destroyAllWindows()




