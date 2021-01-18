# OpenCV program to detect face in real time
# import libraries of python OpenCV 
# where its functionality resides
import cv2 
import argparse 
import os
  
 # parse arguments
parser = argparse.ArgumentParser(description='OpenCV Face Detection')
parser.add_argument('--src', action='store', default=1, nargs='?', help='Set video source; Mac Camera')
parser.add_argument('--w', action='store', default=320, nargs='?', help='Set video width')
parser.add_argument('--h', action='store', default=240, nargs='?', help='Set video height')
args = parser.parse_args()
 
# load the required trained XML classifiers
# https://github.com/Itseez/opencv/blob/master/
# data/haarcascades/haarcascade_frontalface_default.xml
# Trained XML classifiers describes some features of some
# object we want to detect a cascade function is trained
# from a lot of positive(faces) and negative(non-faces)
# images.
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
 
# capture frames from a camera
cap = cv2.VideoCapture(args.src)
 
while 1:            
    # reads frames from a camera
    ret, img = cap.read() 
 
    # convert to gray scale of each frames
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  
    # Detects faces of different sizes in the input image
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in faces:
        # To draw a circle in a face 
        cv2.circle(img, (int(x+w/2),int(y+h/2)),int(max(w,h)/2),(255,255,0),2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
  
  
    # Display an image in a window
    cv2.imshow('opencv face detection', img)
 
    # Wait for Esc key to stop    
    if cv2.waitKey(10) & 0xFF == 27:
        break
  
# Close the window
cap.release()
  
# De-allocate any associated memory usage
cv2.destroyAllWindows()