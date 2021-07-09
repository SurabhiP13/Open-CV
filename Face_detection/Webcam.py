import cv2 as cv
import numpy as np


vid=cv.VideoCapture(0)
def rescaleFrame(frame, scale=1.5):
    width=int(frame.shape[1]*scale)
    height=int(frame.shape[0]*scale)
    dimension=(width, height)
    return cv.resize(frame, dimension, interpolation=cv.INTER_AREA)

haar_cascade=cv.CascadeClassifier('haar_face.xml')
    
while True:
    isTrue, frame=vid.read()
    frame2=rescaleFrame(frame)
    gray=cv.cvtColor(frame2, cv.COLOR_BGR2GRAY)
    face_rect=haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=7)
    for(x, y, w, h) in face_rect:
        cv.rectangle(frame2, (x,y), (x+w, y+h), (0,255,0), thickness=2)
        cv.imshow('Detected Face', frame2)
    
    if cv.waitKey(20)&0xFF==ord('d'):
        break
vid.release()
cv.destroyAllWindows()



# print(f'Number of people found: {len(face_rect)}')



cv.waitKey(0)
