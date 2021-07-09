import cv2 as cv
import numpy as np

img=cv.imread('facee.jpg') #image path
cv.imshow('image', img)

#converts image to grayscale
gray=cv.cvtColor(img, cv.COLOR_BGR2GRAY)

haar_cascade=cv.CascadeClassifier('haar_face.xml')

#compares the haar template with out image
face_rect=haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3)

print(f'Number of people found: {len(face_rect)}')

#makes a rectangle over the detected face
for(x, y, w, h) in face_rect:
    cv.rectangle(img, (x,y), (x+w, y+h), (0,255,0), thickness=2)

cv.imshow('Detected Face', img)

cv.waitKey(0)
