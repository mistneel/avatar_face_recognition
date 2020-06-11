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
from avatar.msg import Face, FacesArray
import cv2


class image2screen:

  def __init__(self, image):
      '''Initialize ros subscriber'''
      self.subscriber = rospy.Subscriber(image, FacesArray, self.callback,  queue_size = 10)
      rospy.loginfo(f"image_2_screen: subscribing to: {image}")

  def callback(self, ros_data):
      '''Callback function of subscribed topic'''
      
      AllFaces = []

      for FaceData in ros_data.FaceArray:

          np_arr = np.fromstring(FaceData.FaceImg.data, np.uint8)
          image_np = cv2.imdecode(np_arr, cv2.IMREAD_COLOR) # OpenCV >= 3.0:
          AllFaces.append(image_np)
      
      FaceView = np.concatenate(AllFaces, axis = 1)

      
      rospy.loginfo(FaceView)
      w, h, d = image_np.shape;
      tmp = cv2.resize(FaceView,(int(h),int(w)))        
        
      cv2.imshow('cv_img', tmp)
      cv2.waitKey(2)


if __name__ == '__main__':
    '''Initializes and cleanup ros node'''
    rospy.init_node('image2screen', anonymous=False)
    print(rospy.get_param_names())
    image = rospy.get_param('~image', '/face_finder/preview')
    rospy.loginfo(f"image_2_screen: starting on {image}")
    ic = image2screen(image)
    try:
        print(rospy.get_param_names())
        rospy.spin()
    except Exception as inst:
        rospy.loginfo("image_2_screen: {inst} shutting down")
    cv2.destroyAllWindows()
