import cv2

cap = cv2.VideoCapture(0)

while True:
    ret,frame = cap.read()
    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF ==27:
        break
#cv2.imwrite('round1arena.jpg',frame)

cap.release()
cv2.destroyAllWindows()