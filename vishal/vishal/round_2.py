import cv2
import numpy as np

color = [(0,242,255),(204,72,63),(76,177,34),(36,28,237)]


img = cv2.imread('image_2.png')

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

arena = cv2.imread('image_rotate.png')
kernel = np.ones((5,5),np.uint8)
closing_original_arena = cv2.morphologyEx(arena, cv2.MORPH_CLOSE, kernel)
arena_gray = cv2.cvtColor(closing_original_arena, cv2.COLOR_BGR2GRAY)
ret,thresh_arena = cv2.threshold(arena_gray,200,255,cv2.THRESH_BINARY)
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
    closin = cv2.rectangle(closing_original_arena, (x2, y2), (x2 + w1, y2 + h1), (0, 255, 0), 2)
    loc.append(x2)
    loc.append(y2)
    arena_roi.append(closing_arena[y2:(y2+h1),x2:(x2+w1)])

resize_roi = []
for j in range(4):
    resize_roi_temp = cv2.resize(f_roi[j], (80, 80), interpolation=cv2.INTER_LINEAR)
    resize_roi.append(resize_roi_temp)


for g in range(4)
    mat = []
    for i in range(20):
        for j in range(20):
            sum = 0
            for m in range(4):
                y = i * 4 + m
                for n in range(4):
                    x = j * 4 + n
                    px = resize_roi[g][y, x]
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

    for v in range(len(arena_roi))
        zeros = []
        for b in range(360):
            h = b+1
            rotate = ndimage.rotate(arena_roi[v],h)
            blur = cv2.medianBlur(rotate, 15)
            ima, cont, hierarchy_1 = cv2.findContours(blur, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            roi_1 = []
                approx = cv2.approxPolyDP(cntr, 0.01 * cv2.arc
            for cntr in cont:Length(cnt, True), True)
                x, y, w, h = cv2.boundingRect(cntr)
                roi_1.append(rotate[y:y + h, x:x + w])

            roi_1[0] = cv2.resize(roi_1[0], (80, 80), interpolation=cv2.INTER_LINEAR)
s
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
            for o in range(400):
                if (a[o] == 0):
                    k = k + 1
                else:
                    continue
            zeros.append(k)


            max_zero = zeros[0]
            q=0
            for t in range(y):
                if(max_zero < zeros[t]):
                    max_zero = zeros[t]
                    q = t
                else:
                    continue
            max_zeros.append(q)






x_loc = []
y_loc = []







for i in range(len(finalsequence)):
    cx = int(moments[finalsequence[i]]['m10'] / moments[finalsequence[i]]['m00'])
    cy = int(moments[finalsequence[i]]['m01'] / moments[finalsequence[i]]['m00'])
    x_loc.append(cx)
    y_loc.append(cy)

for m in range(len(finalsequence)):
    print  m+1 ,'match', x_loc[m] , y_loc[m]

print finalsequence

cv2.imshow('arena',arena)
cv2.imshow('match1',resize_arena[finalsequence[0]])
cv2.imshow('match2',resize_arena[finalsequence[1]])
cv2.imshow('match3',resize_arena[finalsequence[2]])
cv2.imshow('match4',resize_arena[finalsequence[3]])
cv2.imshow('match_roi1',resize_roi[0])
cv2.imshow('match_roi2',resize_roi[1])
cv2.imshow('match_roi3',resize_roi[2])
cv2.imshow('match_roi4',resize_roi[3])

#cv2.imshow('arena_gray',arena_gray)
#cv2.imshow('clos',clos)
#cv2.imshow('closin',closin)
cv2.waitKey(0)
cv2.destroyAllWindows()




