import cv2
import numpy as np

##############################################################################################
# TEMPLATE MATCHING

#GIVEN IMAGE
img = cv2.imread('IMG_1.jpg')

col = [(255,255,1),(0,0,254),(1,255,1),(254,0,0),(249,7,246)]

img = cv2.resize(img, (500, 500), interpolation=cv2.INTER_LINEAR)
imgray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
imgray_invert = cv2.bitwise_not(imgray, dst=None)
ret,thresh = cv2.threshold(imgray_invert ,50, 255 , cv2.THRESH_BINARY)


image, contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
roi = []
moment = []
#EXTRACTING TEMPLATES

for cnt in contours:
    approx = cv2.approxPolyDP(cnt, 0.01 * cv2.arcLength(cnt, True), True)
    x1, y1, w, h = cv2.boundingRect(cnt)
    y = cv2.moments(cnt)
    moment.append(y)
    img = cv2.rectangle(img, (x1, y1), (x1 + w, y1 + h), (255, 0, 0), 2)
    roi.append(img[y1:(y1+h),x1:x1+w])

goal = np.zeros((len(roi),2),np.uint32)

for i in range(len(roi)):
    cx = int(moment[i]['m10'] / moment[i]['m00'])
    cy = int(moment[i]['m01'] / moment[i]['m00'])
    goal[i][0] = cx
    goal[i][1] = cy

e_roi = []
for u in range(len(col)):
    for i in range(len(roi)):
        color = img[goal[i][1],goal[i][0]]
        if (col[u][0] == color[0] and col[u][1] == color[1] and col[u][2] == color[2]):
            e_roi.append(roi[i])
            break

cv2.imshow('er',e_roi[3])
cv2.waitKey(0)
cv2.destroyAllWindows()

