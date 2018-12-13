
# coding: utf-8

# In[1]:


import sys
import os
import dlib
import glob
import cv2
#from scipy.spatial import distance
#from scipy import spatial
import numpy as np


# Load all the models we need: a detector to find the faces, a shape predictor
# to find face landmarks so we can precisely localize the face, and finally the
# face recognition model.
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
facerec = dlib.face_recognition_model_v1('dlib_face_recognition_resnet_model_v1.dat')

def create_read(name,qtd):
    buf =[]
    for i in range(qtd):
        i = name + '%02d.jpg' %i
        img = cv2.imread(i)
        buf.append(img)
    return buf

imgs_list = create_read('image',4)



def vectors_128(imgs_list):
    vectors_list = []
    # Now process all the images
    for img in imgs_list:
        #print("Processing file: {}".format(f))

        #win.clear_overlay()
        #win.set_image(img)

        # Ask the detector to find the bounding boxes of each face. The 1 in the
        # second argument indicates that we should upsample the image 1 time. This
        # will make everything bigger and allow us to detect more faces.
        dets = detector(img, 1)
        if dets is None:
            print('Nenhuma face detectada')
            break
        #print("Number of faces detected: {}".format(len(dets)))

        # Now process each face we found.
        for k, d in enumerate(dets):
            #print("Detection {}: Left: {} Top: {} Right: {} Bottom: {}".format(
            #    k, d.left(), d.top(), d.right(), d.bottom()))
            # Get the landmarks/parts for the face in box d.
            shape = predictor(img, d)
            # Draw the face landmarks on the screen so we can see what face is currently being processed.
            #win.clear_overlay()
            #win.add_overlay(d)
            #win.add_overlay(shape)

            # Compute the 128D vector that describes the face in img identified by
            # shape.  In general, if two face descriptor vectors have a Euclidean
            # distance between them less than 0.6 then they are from the same
            # person, otherwise they are from different people. Here we just print
            # the vector to the screen.
            face_descriptor = facerec.compute_face_descriptor(img, shape)
            #print('face descriptor: '+ str(face_descriptor))
            # It should also be noted that you can also call this function like this:
            #  face_descriptor = facerec.compute_face_descriptor(img, shape, 100)
            # The version of the call without the 100 gets 99.13% accuracy on LFW
            # while the version with 100 gets 99.38%.  However, the 100 makes the
            # call 100x slower to execute, so choose whatever version you like.  To
            # explain a little, the 3rd argument tells the code how many times to
            # jitter/resample the image.  When you set it to 100 it executes the
            # face descriptor extraction 100 times on slightly modified versions of
            # the face and returns the average result.  You could also pick a more
            # middle value, such as 10, which is only 10x slower but still gets an
            # LFW accuracy of 99.3%.
            
            vectors_list.append(np.array(face_descriptor))
        
            #dlib.hit_enter_to_continue()
            
    return vectors_list


def mean_vectors128(vectors128):
    return np.mean(np.array(vectors128), axis=0)


def check_landmark(vector_a, vector_b):
    out = np.sqrt(np.sum(vector_a-vector_b, axis=1)**2)[0] 
    print(out)
    return out < 0.4


def face_landmark(vec_train, img_test):
    
    vec_test = vectors_128(img_test)   
   
    return check_landmark(vec_train, vec_test)
    
def face_landmark_train(imgs_train):
    vec_train = vectors_128(imgs_train)
    vec_mean_train = mean_vectors128(vec_train)
    return vec_mean_train
