import cv2
import numpy as np

##############################################################################################
# TEMPLATE MATCHING

color = [(0,242,255),(76,177,34),(36,28,237),(204,72,63)]


cap = cv2.VideoCapture(0)

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

print finalsequence

goal = np.zeros((len(finalsequence),2),np.uint32)

for i in range(len(finalsequence)):
    cx = int(moments[finalsequence[i]]['m10'] / moments[finalsequence[i]]['m00'])
    cy = int(moments[finalsequence[i]]['m01'] / moments[finalsequence[i]]['m00'])
    #print cx,cy
    goal[i][0] = cx
    goal[i][1] = cy
print goal[0]

cv2.circle(arena,(goal[0][0],goal[0][1]),0,(255,255,255),thickness = 2)
cv2.circle(arena,(goal[1][0],goal[1][1]),0,(255,255,255),thickness = 2)
cv2.circle(arena,(goal[2][0],goal[2][1]),0,(255,255,255),thickness = 2)
cv2.circle(arena,(goal[3][0],goal[3][1]),0,(255,255,255),thickness = 2)

'''
for j in goal:
    print j[0]
'''
print moments[0]
cv2.imshow('closing',arena)
cv2.imshow('test1',resize_roi[0])
cv2.imshow('test2',resize_roi[1])
cv2.imshow('test3',resize_roi[2])
cv2.imshow('test4',resize_roi[3])
cv2.imshow('tested1',resize_arena[finalsequence[0]])
cv2.imshow('tested2',resize_arena[finalsequence[1]])
cv2.imshow('tested3',resize_arena[finalsequence[2]])
cv2.imshow('tested4',resize_arena[finalsequence[3]])
cv2.waitKey(0)
cv2.destroyAllWindows()
cap.release()

# TEMPLATE END
#############################################################################################
