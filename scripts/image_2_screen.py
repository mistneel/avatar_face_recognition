#!/usr/bin/env python3

import argparse

import numpy as np
import rospy
from sensor_msgs.msg import CompressedImage
import cv2


class image2screen:

  def __init__(self, image):
      '''Initialize ros subscriber'''
      self.subscriber = rospy.Subscriber(image, CompressedImage, self.callback,  queue_size = 1)
      rospy.loginfo(f"image_2_screen: subscribing to: {image}")

  def callback(self, ros_data):
      '''Callback function of subscribed topic'''

      np_arr = np.fromstring(ros_data.data, np.uint8)
      image_np = cv2.imdecode(np_arr, cv2.IMREAD_COLOR) # OpenCV >= 3.0:
      w, h, d = image_np.shape;
      tmp = image_np
        
      cv2.imshow('cv_img', tmp)
      cv2.waitKey(2)


if __name__ == '__main__':
    '''Initializes and cleanup ros node'''
    rospy.init_node('image2screen', anonymous=False)
    image = rospy.get_param('image', '/camera/image_raw/compressed')
    rospy.loginfo(f"image_2_screen: starting on {image}")
    ic = image2screen(image)
    try:
        rospy.spin()
    except Exception as inst:
        rospy.loginfo("image_2_screen: {inst} shutting down")
    cv2.destroyAllWindows()

