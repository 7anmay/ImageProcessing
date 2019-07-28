import cv2

cap = cv2.VideoCapture(1)

for i in range(20):
    ret,arena = cap.read()


arena_gray = cv2.cvtColor(arena,cv2.COLOR_BGR2GRAY)
ret,threshhold = cv2.threshold(arena_gray,150,255,cv2.THRESH_BINARY)

image, contours, hierarchy = cv2.findContours(threshhold,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
roi = []

for cnt in contours:
    approx = cv2.approxPolyDP(cnt, 0.01 * cv2.arcLength(cnt, True), True)
    x1, y1, w, h = cv2.boundingRect(cnt)
    area = cv2.contourArea(cnt)
    print area

cv2.imshow('arena',threshhold)
cv2.waitKey(0)
cv2.destroyAllWindows()
