# import the necessary packages
from picamera.array import PiRGBArray
from picamera import PiCamera
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
# import libs create
import lib_main as lm
import dlib_landmark as dl
	

print('Opening video stream...')
cam = PiCamera()
cam.resolution = (640, 480)
cam.framerate = 5
cam.vflip = True
cap = PiRGBArray(cam, size=(640, 480))
time.sleep(1)
#Inicializando
detector,predictor,eye_cascade = lm._init()
cont = 0
fcn = -1
conttext = 0
qtd = 30
cnt = 0
buf = []
acc = 0 
up = 8 
for frame in cam.capture_continuous(cap, format="bgr", use_video_port=True):
	frame = frame.array
	
	if fcn == -1:
		frame = lm.inicio(frame)
		if conttext == 4:
			cv2.putText(frame, 'Treino MACE...',(10, 45), cv2.FONT_HERSHEY_SIMPLEX,
				0.5,(0,0,255),1, cv2.LINE_AA)
		elif conttext == 5:
			cv2.putText(frame, 'Teste MACE...',(10, 45), cv2.FONT_HERSHEY_SIMPLEX,
				0.5,(0,0,255),1, cv2.LINE_AA)
			text = 'MACE:' + str(ok)	
			cv2.putText(frame, text,(10, 25), cv2.FONT_HERSHEY_SIMPLEX,
				0.5,(0,0,255),1, cv2.LINE_AA)
		elif conttext == 6:
			cv2.putText(frame, 'Treino Landmark...',(10, 45), cv2.FONT_HERSHEY_SIMPLEX,
				0.5,(0,0,255),1, cv2.LINE_AA)
		elif conttext == 7:
			cv2.putText(frame, 'Teste Landmark...',(10, 45), cv2.FONT_HERSHEY_SIMPLEX,
				0.5,(0,0,255),1, cv2.LINE_AA)
			text = 'Landmark predictor:' + str(ok_)	
			cv2.putText(frame, text,(10, 25), cv2.FONT_HERSHEY_SIMPLEX,
				0.5,(0,0,255),1, cv2.LINE_AA)
		else:
			frame = lm.inicio(frame)
		up = 0			
			
	elif fcn == 1:
		frame = lm.landmark_detector(frame,detector,predictor)
	elif fcn == 2:
		frame = lm.face_detector(frame,detector,eye_cascade)
	elif fcn == 3:
		cv2.rectangle(frame,(235,120),(235+165,120+200),(0,255,0),2)
		if fcn == 3 and up == 8:
			lm.cap_photos(frame,cont)
			cont +=1
			fcn = 3
			up = 0
		text = "%d photos captured" %cont
		cv2.putText(frame, text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
			0.5, (0, 0, 255), 1, cv2.LINE_AA)
		
	elif fcn == 4:
		buf = lm._read_img('image',qtd)
		mace = lm.mace_train(buf)
		fcn = -1
		conttext = 4
		
	elif fcn == 5:
		cv2.rectangle(frame,(235,120),(235+165,120+200),(0,255,0),2)
		crop = frame[120:120+200,235:235+165]
		if fcn==5 and up==8:
			cv2.imwrite('test_image.jpg',crop)
			ok = lm.mace_test(mace,crop)
			print('Mace:', ok)
			fcn = -1
			cont = 0
			up = 0
		conttext = 5
				
	elif fcn == 6:
		print("Treinando...")
		vec = lm.landmark_predictor_test('image', 15)
		fcn = -1
		conttext = 6
	elif fcn == 7:
		print("Testando...")
		ok_ = lm.landmark_predictor(vec, frame)
		fcn = -1
		conttext = 7
	else:
		pass


	# show the frame
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(100) & 0xFF

	if key == ord("q"):
		break
	elif key == ord("1"):
		fcn = lm.sel(fcn,key)
		if 'mace' in locals():
			del mace
	elif key == ord("2"):
		fcn = lm.sel(fcn,key)
	elif key == ord("0"):
		fcn = lm.sel(fcn,key)
		cont = 0
		conttext = 0
	elif key == ord("3"):
		fcn = lm.sel(fcn,key)
	elif key == ord("4"):
		fcn = lm.sel(fcn,key)
		if 'mace' in locals():
			del mace
	elif key == ord("5"):
		fcn = lm.sel(fcn,key)
	elif key == ord("6"):
		fcn = lm.sel(fcn,key)
	elif key == ord("7"):
		fcn = lm.sel(fcn,key)
	elif key == ord("8"):
		up = 8
	cap.truncate(0)
print('Closing stream...')
cam.close()
cv2.destroyAllWindows()
	
