#!/usr/bin/env python

import argparse

import numpy as np
import rospy
from sensor_msgs.msg import CompressedImage
import cv2
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image


class movie2image:
    _VERBOSE=False

    def __init__(self):##
        '''Initialize ros publisher'''
        self._subscriber = rospy.Subscriber(webcam, Image, self.callback,  queue_size=10)
        self._image_pub = rospy.Publisher("/camera/image_raw/compressed", CompressedImage, queue_size=10) ##/image/compressed
        ##self._vs = cv2.VideoCapture(source)
        self.bridge = CvBridge()

    def callback(self, img):
        cv_img = self.convert_image(img)
        msg = self.spew(cv_img)
        self._image_pub.publish(msg)
                   

    def convert_image(self, img):
        try:
          return self.bridge.imgmsg_to_cv2(img, "rgb8")
        except CvBridgeError as e:
          print(e)

    def spew(self,cv_img) :
        if self._VERBOSE :
            rospy.loginfo('movie_2_image: spewing an image')

        ##ret, frame = self.cv_img.read()
        ##if ret == False :
        ##    return False

        msg = CompressedImage()
        msg.header.stamp = rospy.Time.now()
        msg.format = "jpeg"
        msg.data = np.array(cv2.imencode('.jpg', cv_img)[1]).tostring()
        return msg
        

if __name__ == '__main__':
    rospy.loginfo('movie_2_image: "Starting up')
    rospy.init_node('movie_2_image', anonymous=False)
    webcam = rospy.get_param('~webcam', '/webcam/image_raw')
    updateRate = float(rospy.get_param('rate', '10'))
    mti = movie2image()##
    rate = rospy.Rate(updateRate) # hz
    
    try:
        rospy.spin()

    except rospy.ROSInterruptException as e:
        print(e)

