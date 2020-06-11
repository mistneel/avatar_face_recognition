#!/usr/bin/env python3
#
# This node subscribes to a video/webcam stream topic, detects and recognizes the face(s), and publishes an array of face(s) to the publisher topic.
# Publishing array contains header, name, row, col, height, width, and compressed image of faces. 
#
# Version 1.0 Jun 2020 - Neel Mistry
# Author - Neel Mistry (mistry.neel92@gmail.com)
#


import argparse
import numpy as np
import cv2
import face_recognition
import pickle
import imutils
import rospy
from avatar.msg import Face, FacesArray
from sensor_msgs.msg import CompressedImage
from avatar.srv import SetImagePreview, SetImagePreviewResponse
from std_msgs.msg import String, Bool

class FaceFinder:
    _VERBOSE = False

    def __init__(self, image_stream, haar, pickled, preview_stream, resize=500):
        '''Initialize ros subscriber'''
        self._subscriber = rospy.Subscriber(image_stream, CompressedImage, self.callback,  queue_size = 10)
        self._publisher = rospy.Publisher(preview_stream, FacesArray, queue_size = 10)
        self._data = pickle.loads(open(pickled, "rb").read())
        self._detector = cv2.CascadeClassifier(haar)
        self._resize = resize
        self._publishPic = True
        rospy.Service('set_preview', SetImagePreview, self.set_preview)
        rospy.loginfo(f"face_finder: subscribing to {image}")

    def set_preview(self, preview) :
        if self._VERBOSE:
            rospy.loginfo("face_finder: got a message {preview}")
        self._publishPic = preview
        return SetImagePreviewResponse("Preview set to {self._publishPic}")


    def callback(self, ros_data):
        '''Callback function of subscribed topic'''

        np_arr = np.fromstring(ros_data.data, np.uint8)
        image_np = cv2.imdecode(np_arr, cv2.IMREAD_COLOR) # OpenCV >= 3.0:
        if self._resize > 0 :
            frame = imutils.resize(image_np, width=self._resize)
        else :
            frame = image_np
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        rects = self._detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)
        if self._VERBOSE:
            rospy.loginfo("face_finder: we found {len(rects)} faces")

        # compute the facial embeddings for each face bounding box
        boxes = [(y, x + w, y + h, x) for (x, y, w, h) in rects]
        encodings = face_recognition.face_encodings(rgb, boxes)
        names = []
        matches = []

        for encoding in encodings:
            matches = face_recognition.compare_faces(self._data["encodings"], encoding)

        name = "Unknown"
        if True in matches:
            matchedIdxs = [i for (i, b) in enumerate(matches) if b]
            counts = {}

            for i in matchedIdxs:
                name = self._data["names"][i]
                counts[name] = counts.get(name, 0) + 1

            name = max(counts, key=counts.get)

        names.append(name)

        #generate image parmeter array and publish
        
        if self._publishPic :
            msg_faces_array = FacesArray()
            offset = 10
            for (top, right, bottom, left) in boxes:
                if(top>0 and right>0 and bottom>0 and left>0):
                    msg_raw_image = Face()

                    cropped_face = frame[top-offset:bottom+offset, left-offset:right+offset]
                    cropped_face = cv2.resize(cropped_face,(100,100))
                
                    msg_raw_image.FaceImg.header.stamp = rospy.Time.now()
                    msg_raw_image.FaceImg.format = "jpeg"
                    msg_raw_image.FaceImg.data = np.array(cv2.imencode('.jpg', cropped_face)[1]).tostring()
                
                    msg_raw_image.name = name
                    msg_raw_image.row = bottom
                    msg_raw_image.col = right
                    msg_raw_image.height = top
                    msg_raw_image.width = left
                    print("z")
                
                
                    msg_faces_array.FaceArray.append(msg_raw_image)
                    msg_faces_array.header.stamp = rospy.Time.now()
            
            if len(msg_faces_array.FaceArray) > 0:
                self._publisher.publish(msg_faces_array)
                rospy.loginfo(msg_faces_array)    
            

if __name__ == '__main__':
    '''Initializes and cleanup ros node'''
    image = rospy.get_param('~image', '/camera/image_raw/compressed')
    haar = rospy.get_param('~haar', '/home/neel/catkin_ws/src/avatar/haarcascade_frontalface_default.xml')
    pickled = rospy.get_param('~pickle', '/home/neel/catkin_ws/src/avatar/faces.pickle')
    preview = rospy.get_param('~preview', '/face_finder/preview')

    rospy.init_node('face_finder', anonymous=False)
    ic = FaceFinder(image, haar, pickled, preview, resize=500)

    try:
        rospy.spin()
    except KeyboardInterrupt:
        rospy.loginfo("face_finder: shutting down")
#    cv2.destroyAllWindows()
