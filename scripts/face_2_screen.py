#!/usr/bin/env python3
#
# This node subscribes to face_array message topic. From the arrays, it takes all the images and concatenate them horizontally. It then displays concatenated images on screen.
#
# Version 1.0 Jun 2020 - Neel Mistry
# Author - Neel Mistry (mistry.neel92@gmail.com)
#

import argparse

import numpy as np
import rospy
from sensor_msgs.msg import CompressedImage
from avatar_face_recognition.msg import Face, FaceArray
import cv2


class face2screen:
  _HEIGHT = 64 # make all images this high

  def __init__(self, image):
      '''Initialize ros subscriber'''
      self.subscriber = rospy.Subscriber(image, FaceArray, self.callback,  queue_size = 10)
      rospy.loginfo(f"face_2_screen: subscribing to: {image}")

  def callback(self, ros_data):
      '''Callback function of subscribed topic'''
      
      allFaces = []
      rospy.loginfo(f"face_2_screen: We received {len(ros_data.faces)} faces")
      for faceData in ros_data.faces:

          np_arr = np.fromstring(faceData.FaceImg.data, np.uint8)
          image_np = cv2.imdecode(np_arr, cv2.IMREAD_COLOR) # OpenCV >= 3.0:
          scale = float(self._HEIGHT) / image_np.shape[0]
          allFaces.append(cv2.resize(image_np, (self._HEIGHT, int(scale * image_np.shape[1]))))
          #rospy.loginfo(f"Face db {faceData.json}")
      
      allFaces = np.hstack(allFaces)
        
      cv2.imshow('faces', allFaces)
      cv2.waitKey(2)


if __name__ == '__main__':
    '''Initializes and cleanup ros node'''
    rospy.init_node('face2screen', anonymous=False)
    print(rospy.get_param_names())
    image = rospy.get_param('~image', '/face_finder/faces')
    rospy.loginfo(f"face_2_screen: starting on {image}")
    ic = face2screen(image)
    try:
        print(rospy.get_param_names())
        rospy.spin()
    except Exception as inst:
        rospy.loginfo("face_2_screen: {inst} shutting down")
    cv2.destroyAllWindows()
