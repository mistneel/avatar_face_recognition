#!/usr/bin/env python3
#
# This node subscribes to face_array message topic. From the arrays, it takes all the images and concatenate them horizontally. It then displays concatenated images on screen.
#
# Version 1.0 Jun 2020 - Neel Mistry
# Author - Neel Mistry (mistry.neel92@gmail.com)
#

import argparse

import numpy as np
import json as JSON
import rospy
from sensor_msgs.msg import CompressedImage
from avatar_face_recognition.msg import Face, FaceArray
import cv2


class face2screen:
  __SIZE = 200 # all images fit in a box this size

  def __init__(self, image):
      '''Initialize ros subscriber'''
      self.subscriber = rospy.Subscriber(image, FaceArray, self.callback,  queue_size = 1)
      rospy.loginfo(f"face_2_screen: subscribing to: {image}")

  def callback(self, ros_data):
      '''Callback function of subscribed topic'''
      
      allFaces = []
      rospy.loginfo(f"face_2_screen: We received {len(ros_data.faces)} faces")
      for faceData in ros_data.faces:

          np_arr = np.fromstring(faceData.FaceImg.data, np.uint8)
          image_np = cv2.imdecode(np_arr, cv2.IMREAD_COLOR) # OpenCV >= 3.0:
          scale = float(self.__SIZE) / max(image_np.shape[0], image_np.shape[1])
          print(f' scale {scale} shape {image_np.shape}')
          fit = cv2.resize(image_np, (int(scale*image_np.shape[0]), int(scale*image_np.shape[1])))
          print(f' after scale shape {fit.shape}')
          holder = np.pad(fit, ((0, self.__SIZE-fit.shape[0]), (0, self.__SIZE-fit.shape[1]), (0,0)), mode='constant')

          top = (JSON.loads(faceData.json.data.replace("\'", "\"")))["name"]
          cv2.putText(holder, top, (2, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 140, 255), 2)

          bottom = (JSON.loads(faceData.json.data.replace("\'", "\"")))["maskVal"]
          cv2.putText(holder, bottom, (2, 195), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 140, 255), 2)

          allFaces.append(holder)
          #rospy.loginfo(f"Face db {faceData.json}")
      
      if len(allFaces) > 0:
          allFaces = np.hstack(allFaces)
          #cv2.putText(allFaces, "NEEL", (10, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
          #aa = JSON.loads(ros_data.faces.json.data)
      else:
          allFaces = np.zeros((self.__SIZE,self.__SIZE,3),dtype=np.uint8)
              
      cv2.imshow('faces', allFaces)
      cv2.waitKey(2)


if __name__ == '__main__':
    '''Initializes and cleanup ros node'''
    rospy.init_node('face2screen', anonymous=False)
    image = rospy.get_param('~image', '/mask_detector/faces_with_mask/')
    rospy.loginfo(f"face_2_screen: starting on {image}")
    ic = face2screen(image)
    try:
        rospy.spin()
    except Exception as inst:
        rospy.loginfo("face_2_screen: {inst} shutting down")
    cv2.destroyAllWindows()
