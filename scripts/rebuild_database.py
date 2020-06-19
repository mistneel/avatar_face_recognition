#!/usr/bin/env python3
#
# Update the face tracking database. This node does nothing until prodded and if so,
# does so.
#
# Note: There are two critical parameters:
# dataset - where the dataset lives (absolute path)
# encodings - where to put the encoded dataset (absolute path)
# 
#
# Version 1.0 Jun 2020 - Neel Mistry
# Author - Neel Mistry (mistry.neel92@gmail.com)
#
#
from avatar_face_recognition.srv import UpdateDatabase, UpdateDatabaseResponse
from std_msgs.msg import String
import rospy
from imutils import paths
import pickle
import cv2
import face_recognition
import os
import json

def rebuild_database(req):
    '''Rebuild the dataset'''
    dataset = rospy.get_param('~dataset', '/home/neel/catkin_ws/src/avatar_face_recognition/faces')
    encodingFile = rospy.get_param('~encodings', '/home/neel/catkin_ws/src/avatar_face_recognition/faces.pickle')
    method = rospy.get_param('~method', 'hog')
    info = rospy.get_param('~info', '/home/neel/catkin_ws/src/avatar_face_recognition/info.json')
    rospy.loginfo("avatar: rebuild_database_server called with dataset {dataset} {info} {encodingFile} method {method}")
    

    # grab the paths to the input images in our dataset
    imagePaths = list(paths.list_images(dataset))

    # initialize the list of known encodings and known names
    knownEncodings = []
    knownNames = []

    # read in the info file (a json dictionary) 
    with open(info) as json_file:
        data = json.load(json_file)
    rospy.loginfo("avatar: rebuild_database_server loaded the info file from {info}")
    print(data)

    # loop over the image paths
    for (i, imagePath) in enumerate(imagePaths):
        # extract the person name from the image path
        rospy.loginfo("avatar: rebuild_database_server Processing image {i+1}/{len(imagePaths)}")
        name = imagePath.split(os.path.sep)[-2]

        # load the input image and convert it from RGB (OpenCV ordering)
        # to dlib ordering (RGB)
        image = cv2.imread(imagePath)
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # detect the (x, y)-coordinates of the bounding boxes
        # corresponding to each face in the input image
        boxes = face_recognition.face_locations(rgb, model=method)

        # compute the facial embedding for the face
        encodings = face_recognition.face_encodings(rgb, boxes)

        # loop over the encodings
        for encoding in encodings:
            # add each encoding + name to our set of known names and  encodings
            knownEncodings.append(encoding)
            knownNames.append(name)

    # dump the facial encodings + names to disk
    rospy.loginfo("Serializing encodings...")
    data = {"encodings": knownEncodings, "names": knownNames, "info": data}
    rospy.loginfo("Writing to {encodingFile}")
    f = open(encodingFile, "wb")
    f.write(pickle.dumps(data))
    f.close()

    return UpdateDatabaseResponse("rebuilt with {len(knownEncodings)} individual faces")

def warn():
    rospy.loginfo("avatar: rebuild_database_server shutting down (killed)")

def rebuild_database_server():
    rospy.init_node('rebuild_database_server')

    s = rospy.Service('rebuild_database', UpdateDatabase, rebuild_database)
    rospy.loginfo("avatar: rebuild_database_server is running")
    rospy.on_shutdown(warn)
    try :
        rospy.spin()
    except rospy.ROSInterruptException:
        pass

if __name__ == "__main__":
    rebuild_database_server()
