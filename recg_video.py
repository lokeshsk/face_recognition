
import face_recognition
from sklearn import svm
import os
import cv2
import numpy
import joblib
def recg():
	video_capture = cv2.VideoCapture(0)
	ret, frame = video_capture.read()
	#imgpath = "train_dir\john wick\john_wick_face-3.jpeg" #pass image path here
	#frame = img = cv2.imread(imgpath) 
	small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

	# Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
	#rgb_small_frame = small_frame[:, :, ::-1]
	rgb_small_frame = numpy.ascontiguousarray(small_frame[:, :, ::-1])
	# Find all the faces and face encodings in the current frame of video
	face_locations = face_recognition.face_locations(rgb_small_frame)
	face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
	clf = joblib.load("face_recogniser.sav")
	try:
		name = clf.predict(face_encodings)
		print(name[0])
		#return name[0]
	except:
		print ("Unknown user")
		#return "Unknown User"

recg()

         
