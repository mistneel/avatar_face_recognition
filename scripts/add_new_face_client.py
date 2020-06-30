#!/usr/bin/env python3
#
# Version 1.0 Jun 2020 - Neel Mistry
# Author - Neel Mistry (mistry.neel92@gmail.com)
#
#
#


import sys
import rospy
from avatar_face_recognition.srv import *

def recognize_faces_client():
    rospy.wait_for_service('rebuild_database')
    print("Found server service")
    try:
        register_new_face = rospy.ServiceProxy('rebuild_database', UpdateDatabase)
        resp1 = register_new_face()
        return resp1.response
    except rospy.ServiceException as e:
        print("Service call failed: " + str(e))


if __name__ == "__main__":
    
    print("Response: " + str(recognize_faces_client()))
