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

f_roi=[]
for u in range(4):
    for k in range(4):
        t = 0
        rows, cols = roi[k].shape[:2]
        for i in range(rows):
            for j in range(cols):
                if (color[u][0] == roi[k][i][j][0] and color[u][1] == roi[k][i][j][1] and color[u][2] == roi[k][i][j][2]) :
                    f_roi.append(roi[k])
                    print "asd"
                    t=1
                    break
            if t==1:
                break
            else:
                continue

print len(f_roi)
cv2.imshow('s1',f_roi[0])
cv2.imshow('s2',f_roi[1])
cv2.imshow('s3',f_roi[2])
cv2.imshow('s4',f_roi[3])
cv2.waitKey(0)
cv2.destroyAllWindows()
