#!/usr/bin/env python3
"""Play a sine signal."""
import argparse
import sys

import numpy as np
import sounddevice as sd
import cv2

import ctypes
user32 = ctypes.windll.user32
screen_width = user32.GetSystemMetrics(0)
screen_height = user32.GetSystemMetrics(1)

num_notes = 50

start_idx = 0
frequency=1000
amplitude = 0.1

note_count = 1
offset = 0

cap = cv2.VideoCapture(0)
if not cap.isOpened:
    print('--(!)Error opening video capture')
    exit(0)



samplerate = sd.query_devices(sd.default.device, 'output')['default_samplerate']
def callback(outdata, frames, time, status):
    frames  = frames * 1
    global frequency
    """ global note_count
    note_count +=1
    if(note_count > num_notes):
        note_count = 1 """
    frequency = 440*pow(2,note_count/12) + offset
    #print(frequency)
    if(frequency>18000):
        frequency = 100
    if status:
        print(status, file=sys.stderr)
    global start_idx
    t = (start_idx + np.arange(frames)) / samplerate
    t = t.reshape(-1, 1)
    outdata_temp = amplitude * np.sin(2 * np.pi * frequency * t) +  np.sin(2 * np.pi * 3* frequency * t)
    outdata_temp = 0
    for i in range(5):
        outdata_temp += amplitude * np.sin(2 * np.pi * (i+1)* frequency * t)

    hanning_array_shape = np.array( np.shape(outdata_temp[:,0]) )[0]
    #print("hanning_array_shape",hanning_array_shape)
    array_shape = np.shape(outdata_temp)[0:1]
    #print("array_shape",array_shape )
    #print(type(hanning_array_shape))
    #np.hanning(hanning_array_shape)
    outdata_temp1 = np.multiply(outdata_temp , np.ones(hanning_array_shape) )
    
    outdata[:] = outdata_temp
    #print(np.shape(outdata_temp1))
    
    
    start_idx += frames

def detectAndDisplay(frame):
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame_gray = cv2.equalizeHist(frame_gray)
    #-- Detect faces
    faces = face_cascade.detectMultiScale(frame_gray)
    center = [0,0]
    size = [0,0]
    for (x,y,w,h) in faces:
        center = (x + w//2, y + h//2)
        size = [w,h]
        #print(w,h)
        frame = cv2.ellipse(frame, center, (w//2, h//2), 0, 0, 360, (255, 0, 255), 4)
        faceROI = frame_gray[y:y+h,x:x+w]
        #-- In each face, detect eyes
        eyes = eyes_cascade.detectMultiScale(faceROI)
        for (x2,y2,w2,h2) in eyes:
            eye_center = (x + x2 + w2//2, y + y2 + h2//2)
            radius = int(round((w2 + h2)*0.25))
            frame = cv2.circle(frame, eye_center, radius, (255, 0, 0 ), 4)
    #cv2.imshow('Capture - Face detection', frame)
    return center, size

parser = argparse.ArgumentParser(description='Code for Cascade Classifier tutorial.')
parser.add_argument('--face_cascade', help='Path to face cascade.', default='data/haarcascades/haarcascade_frontalface_alt.xml')
parser.add_argument('--eyes_cascade', help='Path to eyes cascade.', default='data/haarcascades/haarcascade_eye_tree_eyeglasses.xml')
parser.add_argument('--camera', help='Camera divide number.', type=int, default=0)
args = parser.parse_args()
face_cascade_name = args.face_cascade
eyes_cascade_name = args.eyes_cascade
face_cascade = cv2.CascadeClassifier()
eyes_cascade = cv2.CascadeClassifier()
#-- 1. Load the cascades
if not face_cascade.load(cv2.samples.findFile(face_cascade_name)):
    print('--(!)Error loading face cascade')
    exit(0)
if not eyes_cascade.load(cv2.samples.findFile(eyes_cascade_name)):
    print('--(!)Error loading eyes cascade')
    exit(0)


with sd.OutputStream(device=sd.default.device, channels=1, callback=callback,samplerate=samplerate):
    while(True):
        ##capture image
        ret, frame = cap.read()
        if frame is None:
            print('--(!) No captured frame -- Break!')
            break
        frame = cv2.flip(frame,1)
        cv2.imshow('Capture', frame)
        center, face_size = detectAndDisplay(frame)
        if(center != [0,0]):#detected
            print(center)
        note_count = center[0]*center[1]*num_notes/(frame.shape[0] * frame.shape[1])
        #offset = frame.shape[0]

        if cv2.waitKey(20) == 27:#esc button
            break
        



cap.release()
cv2.destroyAllWindows()



