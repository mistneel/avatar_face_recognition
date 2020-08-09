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
# Version 1.1 Jul 2020 - Neel Mistry (chnaged the face detection method to use dnn neural network to detect faces with/without mask)
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
import imutils
import numpy as np



def rebuild_database(req):
    '''Rebuild the dataset'''
    dataset = rospy.get_param('~dataset', 'faces')
    encodingFile = rospy.get_param('~encodings', 'faces.pickle')
    method = rospy.get_param('~method', 'hog')
    info = rospy.get_param('~info', 'info.json')
    prototxt = rospy.get_param('~prototxt', 'deploy.prototxt')
    weights = rospy.get_param('~weights', 'res10_300x300_ssd_iter_140000.caffemodel')


    rospy.loginfo("avatar: rebuild_database_server called with dataset {dataset} {info} {encodingFile} method {method}")



    _faceNet = cv2.dnn.readNet(prototxt, weights)
    __conf = 0.35
    

    # grab the paths to the input images in our dataset
    imagePaths = list(paths.list_images(dataset))

    # initialize the list of known encodings and known names
    knownEncodings = []
    knownNames = []

    # read in the info file (a json dictionary) 
    with open(info) as json_file:
        data = json.load(json_file)
    rospy.loginfo(f"avatar: rebuild_database_server loaded the info file from {info}")

    # loop over the image paths
    for (i, imagePath) in enumerate(imagePaths):

        # extract the person name from the image path
        rospy.loginfo(f"avatar: rebuild_database_server Processing image {i+1}/{len(imagePaths)}")
        name = imagePath.split(os.path.sep)[-2]

        image = cv2.imread(imagePath)
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        image = imutils.resize(image, width=400)

        (h, w) = image.shape[:2]
        blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300), (104.0, 177.0, 123.0))

	# pass the blob through the network and obtain the face detections
        _faceNet.setInput(blob)
        detections = _faceNet.forward()

	# initialize our list of face locations
        locs = []

	# loop over the detections
        for i in range(0, detections.shape[2]):
	    # extract the confidence (i.e., probability) associated with
	    # the detection
            confidence = detections[0, 0, i, 2]

	    # filter out weak detections by ensuring the confidence is
	    # greater than the minimum confidence
            if confidence > __conf:
                # compute the (x, y)-coordinates of the bounding box for the object
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")

		# ensure the bounding boxes fall within the dimensions of the frame
                (startX, startY) = (max(0, startX), max(0, startY))
                (endX, endY) = (min(w - 1, endX), min(h - 1, endY))

		# add the face and bounding boxes to their respective lists
                #faces.append(face)
                locs.append((startX, startY, endX, endY))

	# loop over the detected face locations and their corresponding
	# locations
        for box in locs:
            (startX, startY, endX, endY) = box
       
        ##for test only
        cv2.rectangle(image, (startX, startY), (endX, endY), (0, 0, 255), 2)
        cv2.imshow('faces', image)
        cv2.waitKey(3)


        boxes = [(y, xw, yh, x) for (x, y, xw, yh) in locs]

        # compute the facial embedding for the face
        encodings = face_recognition.face_encodings(rgb, boxes, num_jitters = 25)

        # loop over the encodings
        for encoding in encodings:
            # add each encoding + name to our set of known names and  encodings
            knownEncodings.append(encoding)
            knownNames.append(name)

    # dump the facial encodings + names to disk
    rospy.loginfo("Serializing encodings...")
    data = {"encodings": knownEncodings, "names": knownNames, "info": data}
    rospy.loginfo(f"Writing to {encodingFile}")
    f = open(encodingFile, "wb")
    f.write(pickle.dumps(data))
    f.close()

    return UpdateDatabaseResponse(f"rebuilt with {len(knownEncodings)} individual faces")




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
