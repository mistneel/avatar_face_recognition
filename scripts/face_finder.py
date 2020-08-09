#!/usr/bin/env python3
#
# This node subscribes to a video/webcam stream topic, detects and recognizes the face(s), and publishes an array of face(s) to the publisher topic.
# Publishing array contains header, name, row, col, height, width, and compressed image of faces. 
#
# Version 1.3 Jul 2020 - Neel Mistry (chnaged the face detection method to use dnn neural network to detect faces with/without mask)
# Version 1.2 Jun 2020 - Neel Mistry (few formatting improvments)
# Version 1.1 Jun 2020 - Neel Mistry (added iou calculation for better predicition accuracy)
# Version 1.0 Jun 2020 - Neel Mistry
# Author - Neel Mistry (mistry.neel92@gmail.com)
#

import os.path
import argparse
import os.path
import numpy as np
import cv2
import face_recognition
import pickle
import imutils
import rospy
import json as JSON
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import String, Bool, Header
from avatar_face_recognition.msg import Face, FaceArray
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils.video import VideoStream


class FaceFinder:
    _seq = 0

    def __init__(self, image_stream, prototxt_file, weights_file , pickled, face_stream, json_file, confidence_param, face_score_param, verbose, buff_size):
        '''Initialize ros subscriber'''


        try:
            self._data = pickle.loads(open(pickled, "rb").read())
            self._json = JSON.load(open(json_file))
            self._faceNet = cv2.dnn.readNet(prototxt_file, weights_file)

        except Exception as e:
            if verbose:
                rospy.loginfo(f"face_finder: critical error {e}")
            rospy.signal_shutdown(f"face_finder: critical exception during initialization {e}")
            return

        #make sure no name contains ":" in it 
        for key, val in self._json.items():
            for subkey, name in val.items():
                if ":" in name:
                    rospy.loginfo(f"face_finder: invalid char ':' found in json names - '{name}'")
                    rospy.signal_shutdown(f"face_finder: invalid char ':' found in json names - '{name}'")

        self._json["Unknown"] = {"name": "Unknown"}
        self._subscriber = rospy.Subscriber(image_stream, CompressedImage, self.callback,  queue_size = 1, buff_size=buff_size)
        self._publisher = rospy.Publisher(face_stream, FaceArray, queue_size = 1)
        self.__conf = confidence_param
        self.__max_score = face_score_param
        self.__VERBOSE = verbose


    def callback(self, ros_data):
        '''Callback function of subscribed topic'''
        rospy.loginfo(f"face_finder: iprocessing image {ros_data.header.seq}")


        np_arr = np.fromstring(ros_data.data, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR) # OpenCV >= 3.0
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

	#resize frame to have a maximum width of 400 pixels
        frame = imutils.resize(frame, width=400)

	# detect faces in the frame
        (locs) = self.detect_and_predict_mask(frame, self._faceNet)


        boxes = [(y, xw, yh, x) for (x, y, xw, yh) in locs]
        

        encodings = face_recognition.face_encodings(rgb, boxes)
        
        names = []
        

        for pos, encoding in enumerate(encodings):
            name = "Unknown"
            Z = list(face_recognition.face_distance(self._data["encodings"], encoding))
                        
            if min(Z) < self.__max_score:
                name = self._data["names"][Z.index(min(Z))]
              
                if (name != "Unknown") and (name in names):
                    boxes.pop(pos)
                    continue
                    
            names.append(name)
        

        if self.__VERBOSE:
            rospy.loginfo(f"face_finder: we found {len(boxes)} faces: {names}")  
        
       
        #generate image parmeter array and publish
        
        msg_faces_array = FaceArray()
        offset = 10 
        for i, (top, right, bottom, left) in enumerate(boxes):
            if(top>0 and right>0 and bottom>0 and left>0):
                msg_raw_image = Face()

                c_top = max(0, top-offset)
                c_bottom = min(bottom+offset, frame.shape[0])
                c_left = max(0, left-offset)
                c_right = min(right+offset, frame.shape[1])
                cropped_face = frame[c_top:c_bottom, c_left:c_right]
                
                msg_raw_image.FaceImg.format = "jpeg"
                msg_raw_image.FaceImg.data = np.array(cv2.imencode('.jpg', cropped_face)[1]).tostring()
                
                msg_raw_image.json.data = JSON.dumps((self._json[names[i]])) # must be a string
                msg_raw_image.row.data = bottom
                msg_raw_image.col.data = right
                msg_raw_image.height.data = top
                msg_raw_image.width.data = left
                
                msg_faces_array.faces.append(msg_raw_image)

        msg_faces_array.header.stamp = rospy.Time.now()
        msg_faces_array.header.seq = self._seq
        self._seq = self._seq + 1
        print(self._seq)
        rospy.loginfo(f"face_finder: Transmitting {len(msg_faces_array.faces)} faces")
        self._publisher.publish(msg_faces_array)
 

    def detect_and_predict_mask(self, image, faceNetwork):
	# grab the dimensions of the frame and then construct a blob
	# from it
        (h, w) = image.shape[:2]
        blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300), (104.0, 177.0, 123.0))

	# pass the blob through the network and obtain the face detections
        faceNetwork.setInput(blob)
        detections = faceNetwork.forward()

	# initialize our list of face locations
        locs = []

	# loop over the detections
        for i in range(0, detections.shape[2]):
	    # extract the confidence (i.e., probability) associated with
	    # the detection
            confidence = detections[0, 0, i, 2]

	    # filter out weak detections by ensuring the confidence is
	    # greater than the minimum confidence
            if confidence > self.__conf:
                # compute the (x, y)-coordinates of the bounding box for the object
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")

		# ensure the bounding boxes fall within the dimensions of the frame
                (startX, startY) = (max(0, startX), max(0, startY))
                (endX, endY) = (min(w - 1, endX), min(h - 1, endY))

		# add the bounding boxes to their respective lists
                locs.append((startX, startY, endX, endY))

	
        return (locs)

        

if __name__ == '__main__':
    '''Initializes and cleanup ros node'''

    rospy.init_node('face_finder', anonymous=False)    

    image = rospy.get_param('~image', '/camera/image_raw/compressed')
    faces = rospy.get_param('~faces', '/face_finder/faces')
    prototxt = rospy.get_param('~prototxt', 'deploy.prototxt')
    weights = rospy.get_param('~weights', 'res10_300x300_ssd_iter_140000.caffemodel')
    pickled = rospy.get_param('~pickle', 'faces.pickle')
    jsonFile = rospy.get_param('~json', 'info.json')
    confidence_val = rospy.get_param('~confidence', 0.35)
    face_score_val = rospy.get_param('~face_score', 0.3)
    verbose = rospy.get_param('~verbose', True)
    buff_size = int(rospy.get_param('~buffsize', str(1024*1024*20))) #

    ic = FaceFinder(image, prototxt, weights, pickled, faces, jsonFile, confidence_val, face_score_val, verbose, buff_size)

    try:
        rospy.spin()
    except KeyboardInterrupt:
        rospy.loginfo("face_finder: shutting down")

