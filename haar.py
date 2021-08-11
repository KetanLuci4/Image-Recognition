#pylint:disable=no-member

import cv2 as cv
import numpy as np

# cv.waitKey(0)

def path(imgPath):
    global faces
    # img = cv.imdecode(np.fromfile(path,dtype=np.uint8),-1)
    img = cv.imread(imgPath)
    # cv.imshow('mypic', img)

    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # cv.imshow('Gray People', gray)

    haar_cascade = cv.CascadeClassifier('haar_face.xml')

    faces_rect = haar_cascade.detectMultiScale(gray,  scaleFactor=1.1, minNeighbors=1)

    # print(f'Number of faces found = {len(faces_rect)}')

    for (x,y,w,h) in faces_rect:
        cv.rectangle(img, (x,y), (x+w,y+h), (0,255,0), thickness=2)
        faces = gray[y:y + h, x:x + w]
        # img2 = cv.imshow("face",faces)

    return faces

# path(r"C:\Users\ketan\Pictures\mypic1.jpg")
# cv.waitKey(0)
