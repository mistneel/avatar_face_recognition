#!/usr/bin/env python3
#
# This node subscribes to the topic face_finer.py publishes to. It takes all the faces produced by face_finder to see if the faces has mask on them. Finally, it appends mask info to exisiting json string and publishes to publisher.
#
# Version 1.0 Jun 2020 - Neel Mistry
# Author - Neel Mistry (mistry.neel92@gmail.com)
#

import os.path
import rospy
import json as JSON
import numpy as np
import argparse
import imutils
import cv2
import os
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import String, Bool, Header
from avatar_face_recognition.msg import Face, FaceArray
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils.video import VideoStream



class MaskDetector:
    _seq = 0

    def __init__(self, face_stream, model_file, results_stream, VERBOSE_param):
        '''Initialize ros subscriber'''

        try:
            self._maskNet = load_model(model_file)  
        except Exception as e:
            rospy.signal_shutdown(f"mask_detector: critical exception during initialization {e}")
            return

        self._subscriber = rospy.Subscriber(face_stream, FaceArray, self.callback,  queue_size = 10)
        self._publisher = rospy.Publisher(results_stream, FaceArray, queue_size = 10)
        self.__VERBOSE = VERBOSE_param

    def callback(self, ros_data):
        '''Callback function of subscribed topic'''
      
        rospy.loginfo(f"mask_detector: subscribing to {faces}")

        allFaces = []
        allPreds = []

        for faceData in ros_data.faces:


            face = np.fromstring(faceData.FaceImg.data, np.uint8)
            face = cv2.imdecode(face, cv2.IMREAD_COLOR)
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face = cv2.resize(face, (224, 224))
            face = img_to_array(face)
            face = preprocess_input(face)

            allFaces.append(face)

        if len(allFaces) > 0:
            allFaces = np.array(allFaces, dtype = "float32")
            allPreds = self._maskNet.predict(allFaces, batch_size = 32)

        allRes = []
        
        for pred in allPreds:

            (mask, withoutMask) = pred

            res = "Mask" if mask > withoutMask else "No-Mask"
		# include the probability in the result
            res = "{} {:.2f}%".format(res, max(mask, withoutMask) * 100)
            allRes.append(res)

        if self.__VERBOSE:
            rospy.loginfo(f"mask_detector: we found {len(allPreds)} results : {allRes}")
        
        #add mask results to existing json string and publish

        for i, res in enumerate(allRes):
                ros_data.faces[i].json.data 

                s = JSON.loads(ros_data.faces[i].json.data)
                s["maskVal"] = allRes[i]
                ros_data.faces[i].json.data = str(s)
                #print(ros_data.faces[i].json.data)
        
        if len(allFaces) > 0:
            ros_data.header.stamp = rospy.Time.now()
            ros_data.header.seq = self._seq
            self._seq = self._seq + 1
            rospy.loginfo(f"mask_detector: Transmitting {len(ros_data.faces)} results")
            self._publisher.publish(ros_data)
            

if __name__ == '__main__':
    '''Initializes and cleanup ros node'''
    rospy.init_node('mask_detector', anonymous=False)

    faces = rospy.get_param('~faces', '/face_finder/faces/')
    modelFile = rospy.get_param('~mask_detector', 'mask_detector.model')
    results = rospy.get_param('~results', '/avatar/results/')
    VERBOSE_val = rospy.get_param('~VERBOSE_val', True)
    
    ic = MaskDetector(faces, modelFile, results, VERBOSE_val)

    try:
        rospy.spin()
    except KeyboardInterrupt:
        rospy.loginfo("mask_detector: shutting down")
