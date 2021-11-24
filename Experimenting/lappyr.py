#!/usr/bin/env python

''' An example of Laplacian Pyramid construction and merging.

Level : Intermediate

Usage : python lappyr.py [<video source>]

References:
  http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.54.299

Alexander Mordvintsev 6/10/12
'''

# Python 2/3 compatibility
from __future__ import print_function
import sys
from PIL import Image
import imutils
PY3 = sys.version_info[0] == 3

if PY3:
    xrange = range

import numpy as np
import cv2 as cv

import video
from common import nothing, getsize

def build_lappyr(img, leveln=6, dtype=np.int16):
    img = dtype(img)
    levels = []
    for _i in xrange(leveln-1):
        next_img = cv.pyrDown(img)
        img1 = cv.pyrUp(next_img, dstsize=getsize(img))
        levels.append(img-img1)
        img = next_img
    levels.append(img)
    return levels

def merge_lappyr(levels):
    img = levels[-1]
    for lev_img in levels[-2::-1]:
        img = cv.pyrUp(img, dstsize=getsize(lev_img))
        img += lev_img
    return np.uint8(np.clip(img, 0, 255))


def main():
    import sys

    try:
        fn = sys.argv[1]
    except:
        fn = 0
    cap = video.create_capture(fn)

    leveln = 6
    # trackbarptr = [5 for i in range(leveln)]
    trackbarptr = [0,15,22,50,16,8]
    cv.namedWindow('level control')
    for i in xrange(leveln):
        cv.createTrackbar('%d'%i, 'level control',trackbarptr[i], 50, nothing)

    arrow_pts = np.array([[255,27],[364,143],[224,260],[225,172],[19,172][19,114],[225,114]])

    while True:
        _ret, frame = cap.read()
        frame2 = frame
        pyr = build_lappyr(frame, leveln)
        for i in xrange(leveln):
            v = int(cv.getTrackbarPos('%d'%i, 'level control') / 5)
            pyr[i] *= v
        res = merge_lappyr(pyr)

        cv.imshow('laplacian pyramid filter', res)
        
        ###modified begins
        pil_image = Image.fromarray(res.astype(np.uint8))
        # pil_image = Image.open('Image.jpg').convert('RGB') 
        open_cv_image = np.array(pil_image)
        open_cv_image = open_cv_image[:, :, ::-1].copy()
        edged = cv.cvtColor(open_cv_image, cv.COLOR_BGR2GRAY)
        kernel = np.ones((3,3),np.uint8)
        for i in range(1):
            pass
        
        # cv.imshow("edged1", edged)
        # cv.imshow("edged3", edged)
        kernels = [np.ones((2*i+1,2*i+1),np.uint8) for i in range (5)]
        for i in range(5):
            edged = cv.morphologyEx(edged, cv.MORPH_OPEN,kernel=kernels[i])

        edged = cv.GaussianBlur(edged,(3,3),cv.BORDER_DEFAULT)
        edged = cv.GaussianBlur(edged,(5,5),cv.BORDER_DEFAULT)

        kernel2 = np.array([[-1,-1,-1], 
                       [-1, 9,-1],
                       [-1,-1,-1]])
        edged = cv.filter2D(edged, -1, kernel2) # applying the sharpening kernel to the input image & displaying it.
        
        cnts = cv.findContours(edged, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        cnts = sorted(cnts, key=cv.contourArea, reverse=True)
        screenCnt = []
        homograph = []
        for c in cnts and i in range(5):
            peri = cv.arcLength(c, True)
            area = cv.contourArea(c)
            approx = cv.approxPolyDP(c, 0.03*peri, True)

            if len(approx) == 7:# and area/peri < 3.4 and area/peri > 2.2 :
                screenCnt.append(approx)
                homograph.append(cv.findHomography(approx, arrow_pts)
)
                
        if screenCnt is not []:
            cv.drawContours(frame, screenCnt, -1, (0, 255, 0), 3)
            cv.imshow("Arrow", frame)
            cv.imshow("edged", edged)

            # k = cv.waitKey(30)
            # if k == 27:
            #     return
            # cv.waitKey(0)
        # else:
        #     cv.imshow("No arrow", res)
        #     cv.waitKey(0)
        
        # Display Output
        cv.imshow("Laplace of Image", res)
        k = cv.waitKey(15)
        if k == 27:
            return
        elif k==ord('y'):
            print(k)
            if trackbarptr[0] < 50:
                trackbarptr[0]+=1
                cv.setTrackbarPos('%d'%0,'level control', trackbarptr[0])
        elif k==ord('Y'):
            print
            if trackbarptr[0] > 0:
                trackbarptr[0]-=1
                cv.setTrackbarPos('%d'%0,'level control', trackbarptr[0])
        elif k==ord('u'):
            print(k)
            if trackbarptr[1] < 50:
                trackbarptr[1]+=1
                cv.setTrackbarPos('%d'%1,'level control', trackbarptr[1])
        elif k==ord('U'):
            print
            if trackbarptr[1] > 0:
                trackbarptr[1]-=1
                cv.setTrackbarPos('%d'%1,'level control', trackbarptr[1])
        elif k==ord('i'):
            print(k)
            if trackbarptr[2] < 50:
                trackbarptr[2]+=1
                cv.setTrackbarPos('%d'%2,'level control', trackbarptr[2])
        elif k==ord('I'):
            print
            if trackbarptr[2] > 0:
                trackbarptr[2]-=1
                cv.setTrackbarPos('%d'%2,'level control', trackbarptr[2])
        elif k==ord('o'):
            print(k)
            if trackbarptr[3] < 50:
                trackbarptr[3]+=1
                cv.setTrackbarPos('%d'%3,'level control', trackbarptr[3])
        elif k==ord('O'):
            print
            if trackbarptr[3] > 0:
                trackbarptr[3]-=1
                cv.setTrackbarPos('%d'%3,'level control', trackbarptr[3])
        elif k==ord('p'):
            print(k)
            if trackbarptr[4] < 50:
                trackbarptr[4]+=1
                cv.setTrackbarPos('%d'%4,'level control', trackbarptr[4])
        elif k==ord('P'):
            print
            if trackbarptr[4] > 0:
                trackbarptr[4]-=1
                cv.setTrackbarPos('%d'%4,'level control', trackbarptr[4])
        elif k==ord('['):
            print(k)
            if trackbarptr[5] < 50:
                trackbarptr[5]+=1
                cv.setTrackbarPos('%d'%5,'level control', trackbarptr[5])
        elif k==ord('{'):
            print
            if trackbarptr[5] > 0:
                trackbarptr[5]-=1
                cv.setTrackbarPos('%d'%5,'level control', trackbarptr[5])
        else:
            # print(k)
            pass
        ###modified ends
        
        '''
        ###modified 2
        
        pyr2 = build_lappyr(edged, leveln)
        for i in xrange(leveln):
            v = int(cv.getTrackbarPos('%d'%i, 'level control') / 5)
            pyr2[i] *= v
        res2 = merge_lappyr(pyr2)
        cv.imshow('laplacian pyramid filter2', res2)

        pil_image2 = Image.fromarray(res2.astype(np.uint8))
        # pil_image = Image.open('Image.jpg').convert('RGB') 
        open_cv_image2 = np.array(pil_image2)
        open_cv_image2 = open_cv_image2[:, ::-1].copy()
        # edged2 = cv.cvtColor(open_cv_image2, cv.COLOR_BGR2GRAY)
        edged2 = open_cv_image2
        cnts2 = cv.findContours(edged2, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        cnts2 = imutils.grab_contours(cnts2)
        cnts2 = sorted(cnts2, key=cv.contourArea, reverse=True)
        screenCnt2 = []
        for c in cnts:
            peri = cv.arcLength(c, True)
            approx = cv.approxPolyDP(c, 0.03*peri, True)

            if len(approx) == 7:
                screenCnt.append(approx)
                
        if screenCnt is not []:
            cv.drawContours(frame2, screenCnt2, -1, (0, 255, 0), 3)
            cv.imshow("Arrow2", frame)
            cv.imshow("edged2", edged2)

        ###modified 2 ends
        '''
        if cv.waitKey(1) == 27:
            break

    print('Done')


if __name__ == '__main__':
    print(__doc__)
    main()
    cv.destroyAllWindows()

'''
arrow pointing right of viewer
top acute vertex: (255,27)
front tip: (364,143)
lower acute vertex: (224,260)
lower join of rectangle with triangle: (225,172)
lower vertex of tail: (19,172)
upper vertex of tail: (19,114)
lower join of rectangle with triangle: (225,114)

All the vertices are in clockwise orientation
'''