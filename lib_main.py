# import the necessary packages
from imutils.video import VideoStream
from imutils import face_utils
import datetime
import argparse
import imutils
import time
import cv2
import picamera
import dlib
import numpy as np
# import lib create
import dlib_landmark as dl

def _init():
	#Detector dos eyes
	eye_cascade = cv2.CascadeClassifier("haarcascade_eye.xml")
	# Detector da face
	detector = dlib.get_frontal_face_detector()
	#landmark, 69 points
	predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
	return detector,predictor,eye_cascade

def name_img(qtd):
	buf = []
	for i in range(qtd):
		name_order = "image%02d.jpg" %(i+1)
		buf.append(name_order)
	return buf    
    
def face_detector(frame,detector,eye_cascade):
	gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
	faces = detector(gray,1)
	 #Desenha um retângulo nas faces detectadas
	if len(faces) > 0:
		text = "{} face(s) found".format(len(faces))
		cv2.putText(frame, text, (10, 20), cv2.FONT_HERSHEY_SIMPLEX,
			0.5, (0, 0, 255), 2)
	for (i,rect) in enumerate(faces):
		#Definindo retangulo ao redor da face
		(x,y,w,h) = face_utils.rect_to_bb(rect)
		cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
		cv2.putText(frame, "Face #{}".format(i + 1), (x - 10, y - 10),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
		roi_gray = gray[y:y+h, x:x+w]
		roi_color = frame[y:y+h, x:x+w]
		eyes = eye_cascade.detectMultiScale(roi_gray,minNeighbors=15)
		for (ex,ey,ew,eh) in eyes:
			cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
	return frame
	
def landmark_detector(frame,detector,predictor):
	gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
	faces = detector(gray,0)
	 #Desenha um retângulo nas faces detectadas
	if len(faces) > 0:
		text = "{} face(s) found".format(len(faces))
		cv2.putText(frame, text, (10, 20), cv2.FONT_HERSHEY_SIMPLEX,
			0.5, (0, 0, 255), 2)
			
	for (i,rect) in enumerate(faces):
		#Definindo retangulo ao redor da face
		(x,y,w,h) = face_utils.rect_to_bb(rect)
		cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
		cv2.putText(frame, "Face #{}".format(i + 1), (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
		
		shape = predictor(gray,rect)
		shape = face_utils.shape_to_np(shape)
        
		for (i, (x, y)) in enumerate(shape):
			cv2.circle(frame, (x, y), 1, (0, 0, 255), -1)
			#cv2.putText(frame, str(i + 1), (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)
       
	return frame

def landmark_predictor(vec, frame):
	img_test = []
	img_test.append(frame)
	out = dl.face_landmark(vec, img_test)
	return out
	
def landmark_predictor_test(name_train, qtd):
	imgs_train = dl.create_read(name_train, qtd)
	vec_train_mean = dl.face_landmark_train(imgs_train)
	return vec_train_mean
	
	
def cap_photos(frame,cont):
	name = 'image%02d.jpg' %cont
	crop = frame[120:120+200,235:235+165]
	cv2.imwrite(name,crop)


def _read_img(name,qtd):
    buf=[]
    i=0
    for i in range(qtd):
        i = name+'%02d.jpg' %i
        img = cv2.imread(i)
        gray = cv2.equalizeHist(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
        buf.append(gray)
    return buf
    			

def mace_train(buf):
    mace = cv2.face.MACE_create(200)
    print('Training...')
    mace.train(buf)
    return mace


def mace_test(mace, frame):
    gray = cv2.equalizeHist(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
    ans = mace.same(gray)
    return ans

def sel(fcn,key):
	if key == ord("1"):
		fcn = 1
	elif key == ord("2"):
		fcn = 2
	elif key == ord("0"):
		fcn = -1
	elif key == ord("3"):
		fcn = 3
	elif key == ord("4"):
		fcn = 4
	elif key == ord("5"):
		fcn = 5
	elif key == ord("6"):
		fcn = 6
	elif key == ord("7"):
		fcn = 7
	return fcn

def inicio(frame):
	text = '1:Landmark  2:Face  3:Capture Photos  4:MACE Train 5:MACE Test'
	text1 = '6:Landmark Train  7:Landmark Test 8:Confirm 0:Return  q:Exit' 
	cv2.putText(frame, text,(10, frame.shape[0] - 30), cv2.FONT_HERSHEY_SIMPLEX,
	    0.55,(255,255,255),1, cv2.LINE_AA)
	cv2.putText(frame, text1,(30, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX,
	    0.55,(255,255,255),1, cv2.LINE_AA)
	

	return frame
