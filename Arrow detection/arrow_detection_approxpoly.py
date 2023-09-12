import numpy as np
import time
import cv2
import cv2 as cv
import imutils
import math
cap = cv2.VideoCapture(0)
wii=400
hii = 300
while(1):
    # a+=1
    _ , image = cap.read() 
    # image = cv2.imread('./customimages/arrow'+'12'+'.jpg')
    # image=cv2.resize(image, (400,400))
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # gray = cv2.bilateralFilter(gray, 11, 17, 17)
    # edged = cv2.Canny(gray, 30, 200)
    kernel = np.ones((5,5),np.uint8)
    cv2.namedWindow("No arrow", cv2.WINDOW_KEEPRATIO)
    cv2.imshow("No arrow", gray)
    cv2.resizeWindow('No arrow', wii,hii)
    # cv2.waitKey(0)
    # gray = cv2.threshold(gray,cv2.THRESH_OTSU,cv2.THRESH_BINARY,)
    # blur = cv.GaussianBlur(gray,(5,5),0)
    blur=gray
    ret3,gray = cv.threshold(blur,0,230,cv.THRESH_BINARY+cv.THRESH_OTSU)
    # opening = cv2.morphologyEx(gray, cv2.MORPH_OPEN,kernel=kernel)
    # opening = gray
    # opening = blur
    dst = cv.Canny(gray, 50, 200)
    cdst = cv.cvtColor(dst, cv.COLOR_GRAY2BGR)
    # gray = cdst
    # opening = cv2.morphologyEx(opening, cv2.MORPH_CLOSE,kernel=kernel)
    # blur = cv.GaussianBlur(opening,(7,7),0)
    # ret3,gray = cv.threshold(blur,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)
    opening = cv2.morphologyEx(gray, cv2.MORPH_OPEN,kernel=kernel)
    opening = cv2.morphologyEx(opening, cv2.MORPH_CLOSE,kernel=kernel)
    
    edged = opening

    cv2.namedWindow("arrow", cv2.WINDOW_KEEPRATIO)
    cv2.imshow("arrow", edged)
    cv2.resizeWindow('arrow', wii,hii)
    # cv2.waitKey(0)
    if True: # HoughLinesP
        lines = cv.HoughLinesP(dst, 1, math.pi/180.0, 40, np.array([]), 50, 10)
        
        a, b, _c = lines.shape
        for i in range(a):
            cv.line(cdst, (lines[i][0][0], lines[i][0][1]), (lines[i][0][2], lines[i][0][3]), (0, 0, 255), 3, cv.LINE_AA)
    cv.namedWindow("detected lines", cv2.WINDOW_KEEPRATIO)
    cv.imshow("detected lines", cdst)
    cv2.resizeWindow('detected lines', wii, hii)

    cv.namedWindow("source", cv2.WINDOW_KEEPRATIO)
    cv.imshow("source", gray)
    cv2.resizeWindow('source', wii, hii)
    cnts = cv2.findContours(edged, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
    screenCnt = None
    for c in cnts:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.03*peri, True)

        if len(approx) == 7:
            screenCnt = approx
            break
    if screenCnt is not None:
        cv2.drawContours(image, [screenCnt], -1, (0, 255, 0), 3)
        cv2.namedWindow("Arrow", cv2.WINDOW_KEEPRATIO)
        cv2.imshow("Arrow", image)
        cv2.resizeWindow('Arrow', wii, hii)

        # cv2.waitKey(0)
    else:
        cv2.namedWindow("No arrow", cv2.WINDOW_KEEPRATIO)
        cv2.imshow("No arrow", image)
        cv2.resizeWindow('No arrow', wii, hii)

        # cv2.waitKey(0)
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

cap.release()    
cv2.destroyWindow()