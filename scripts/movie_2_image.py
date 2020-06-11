#!/usr/bin/env python3
#
# This node subscribes to a video stream topic, produces compressed images, and publish them to the publisher topic.
#
# Version 1.0 Jun 2020 - Neel Mistry
# Author - Neel Mistry (mistry.neel92@gmail.com)
#

import argparse
import numpy as np
import rospy
from sensor_msgs.msg import CompressedImage
import cv2


class movie2image:
    _VERBOSE=False

    def __init__(self, source):
        '''Initialize ros publisher'''
        self._image_pub = rospy.Publisher("/camera/image_raw/compressed", CompressedImage, queue_size=10)
        self._vs = cv2.VideoCapture(source)

    def spew(self):
        if self._VERBOSE:
          rospy.loginfo('movie_2_image: spewing an image')

        ret, frame = self._vs.read()
        if ret == False:
            return False

        msg = CompressedImage()
        msg.header.stamp = rospy.Time.now()
        msg.format = "jpeg"
        msg.data = np.array(cv2.imencode('.jpg', frame)[1]).tostring()
        self._image_pub.publish(msg)
        return True
        

if __name__ == '__main__':
    rospy.loginfo('movie_2_image: "Starting up')
    rospy.init_node('movie_2_image', anonymous=False)
    movie = rospy.get_param('movie', '/home/neel/catkin_ws/src/avatar/sample/sample.mov')
    updateRate = float(rospy.get_param('rate', '30'))
    mti = movie2image(movie)
    rate = rospy.Rate(updateRate) # hz
    try:
        while not rospy.is_shutdown():
            if not mti.spew():
                exit()
            rate.sleep()
    except KeyboardInterrupt:
        rospy.loginfo('movie_2_image: "Shutting down Image spew"')

