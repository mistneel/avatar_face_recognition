#!/usr/bin/env python3

import sys
import rospy
from avatar.srv import *

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
