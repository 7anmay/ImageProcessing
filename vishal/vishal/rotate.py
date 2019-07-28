import cv2
from scipy import ndimage
img = cv2.imread('vishal.png',0)
img2 = cv2.imread('vishal_1.png',0)
rotate = ndimage.rotate(img, 38.8713002745)
blur = cv2.medianBlur(rotate,15)
image, contours, hierarchy = cv2.findContours(blur,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)


img3 = cv2.imread('vish.png',0)
rotate_1 = ndimage.rotate(img3, 75.2952260268)
blur_1 = cv2.medianBlur(rotate_1,15)
ima, cont, hierarchy_1 = cv2.findContours(blur_1,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)


roi = []
for cnt in contours:
    approx = cv2.approxPolyDP(cnt, 0.01 * cv2.arcLength(cnt, True), True)
    x, y, w, h = cv2.boundingRect(approx)
    rotate = cv2.rectangle(rotate, (x, y), (x + w, y + h), (0, 255, 255), 2)
    roi.append(rotate[y:y+h,x:x+w])

roi[0] = cv2.resize(roi[0],(80,80), interpolation=cv2.INTER_LINEAR)

roi_1 = []
for cnt_1 in cont:
    approx_1 = cv2.approxPolyDP(cnt_1, 0.01 * cv2.arcLength(cnt, True), True)
    x, y, w, h = cv2.boundingRect(approx_1)
    rotate_1 = cv2.rectangle(rotate_1, (x, y), (x + w, y + h), (0, 255, 255), 2)
    roi_1.append(rotate_1[y:y+h,x:x+w])

roi_1[0] = cv2.resize(roi_1[0],(80,80), interpolation=cv2.INTER_LINEAR)



mat = []
sum = 0
for i in range(20):
    for j in range(20):
        sum = 0
        for  m in range(4):
            y = i*4 + m
            for n in range(4):
                x = j*4 + n
                px = roi[0][y,x]
                if px==255 :
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

        for  m in range(4):
            y = i*4 + m
            for n in range(4):
                x = j*4 + n
                px = img2[y,x]
                if px==255 :
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

print len(a)
k = 0
for i in range(400):
    if( a[i] == 0):
        k=k+1
    else:
        continue
print k

roi_1[0] = ndimage.rotate(roi_1[0],90)
cv2.imshow('sdk',roi_1[0])
cv2.imshow('vis',roi[0])
cv2.imshow('djf',img2)

#cv2.imshow('vihsal',rotate)
cv2.waitKey(0)
cv2.destroyAllWindows()