#!/usr/bin/env python3
#
# This node subscribes to a video/webcam stream topic, detects and recognizes the face(s), and publishes an array of face(s) to the publisher topic.
# Publishing array contains header, name, row, col, height, width, and compressed image of faces. 
#
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

class FaceFinder:
    _seq = 0

    def __init__(self, image_stream, haar, pickled, face_stream, json_file, iou_param, verbose, buff_size):
        '''Initialize ros subscriber'''


        try:
            self._data = pickle.loads(open(pickled, "rb").read())
            self._json = JSON.load(open(json_file))
            self._detector = cv2.CascadeClassifier(haar)
        except Exception as e:
            if verbose:
                rospy.loginfo(f"face_finder: critical error {e}")
            rospy.signal_shutdown(f"face_finder: critical exception during initialization {e}")
            return
        self._json["Unknown"] = {"name": "Unknown"}

        self._subscriber = rospy.Subscriber(image_stream, CompressedImage, self.callback,  queue_size = 1, buff_size=buff_size)
        self._publisher = rospy.Publisher(face_stream, FaceArray, queue_size = 1)
        self._detector = cv2.CascadeClassifier(haar)
        self.__iou = iou_param
        self.__VERBOSE = verbose


    def callback(self, ros_data):
        '''Callback function of subscribed topic'''
        rospy.loginfo(f"face_finder: iprocessing image {ros_data.header.seq}")
        np_arr = np.fromstring(ros_data.data, np.uint8)
        image_np = cv2.imdecode(np_arr, cv2.IMREAD_COLOR) # OpenCV >= 3.0:
       
        gray = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)
        rgb = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)

        rects = self._detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)     
        
        boxes = [(y, x + w, y + h, x) for (x, y, w, h) in rects]
        boxes.sort(key=lambda x: (x[2]-x[0]) * (x[1]-x[3]), reverse=True)
        
        faceboxes = []

        if len(boxes) > 0:
            faceboxes.append(boxes[0])

        for i, box in enumerate(boxes[:-1]):
            if self.bb_intersection_over_union(box, boxes[i+1]) <= self.__iou:
                faceboxes.append(boxes[i+1])

        encodings = face_recognition.face_encodings(rgb, faceboxes)
        
        names = []
        
        max_score = 0.45
        score = max_score

        for pos, encoding in enumerate(encodings):
            name = "Unknown"
            Z = list(face_recognition.face_distance(self._data["encodings"], encoding))
                        
            if min(Z) < score:
                score = min(Z)
                name = self._data["names"][Z.index(min(Z))]
                
                if (name != "Unknown") and (name in names):
                    faceboxes.pop(pos)
                    continue
                    
            names.append(name)
        

        if self.__VERBOSE:
            rospy.loginfo(f"face_finder: we found {len(faceboxes)} faces: {names}")  
        
       
        #generate image parmeter array and publish
        
        msg_faces_array = FaceArray()
        offset = 10 
        for i, (top, right, bottom, left) in enumerate(faceboxes):
            if(top>0 and right>0 and bottom>0 and left>0):
                msg_raw_image = Face()

                c_top = max(0, top-offset)
                c_bottom = min(bottom+offset, image_np.shape[0])
                c_left = max(0, left-offset)
                c_right = min(right+offset, image_np.shape[1])
                cropped_face = image_np[c_top:c_bottom, c_left:c_right]
                
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
        rospy.loginfo(f"face_finder: Transmitting {len(msg_faces_array.faces)} faces")
        self._publisher.publish(msg_faces_array)
	
        
    def bb_intersection_over_union(self, boxA, boxB):
	# determine the (x, y)-coordinates of the intersection rectangle
        xA = max(boxA[3], boxB[3])
        yA = max(boxA[0], boxB[0])
        xB = min(boxA[1], boxB[1])
        yB = min(boxA[2], boxB[2])

	# compute the area of intersection rectangle
        interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1) 

	# compute the area of both the prediction and ground-truth
	# rectangles
        boxAArea = (boxA[1] - boxA[3]) * (boxA[2] - boxA[0])
        boxBArea = (boxB[1] - boxB[3]) * (boxB[2] - boxB[0])

	# compute the intersection over union by taking the intersection
	# area and dividing it by the sum of prediction + ground-truth
	# areas - the interesection area
        iou = interArea / float(boxAArea + boxBArea - interArea)

	# return the intersection over union value
        return iou
            

if __name__ == '__main__':
    '''Initializes and cleanup ros node'''

    rospy.init_node('face_finder', anonymous=False)    

    image = rospy.get_param('~image', '/camera/image_raw/compressed')
    faces = rospy.get_param('~faces', '/face_finder/faces')
    haar = rospy.get_param('~haar', 'haarcascade_frontalface_default.xml')
    pickled = rospy.get_param('~pickle', 'faces.pickle')
    jsonFile = rospy.get_param('~json', 'info.json')
    iou_val = rospy.get_param('~iou_val', 0.5)
    verbose = rospy.get_param('~verbose', True)
    buff_size = int(rospy.get_param('~buffsize', str(1024*1024*20))) #

    ic = FaceFinder(image, haar, pickled, faces, jsonFile, iou_val, verbose, buff_size)

    try:
        rospy.spin()
    except KeyboardInterrupt:
        rospy.loginfo("face_finder: shutting down")

