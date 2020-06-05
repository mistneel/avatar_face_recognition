#!/usr/bin/env python3
#
# This is a wrapper for standard face detection and identification tools in opencv
#
# Version 1.1 Feb 2020 - now with bells and whistles
# Version 1.0 July 2019 - basic functionality
#


import argparse
import numpy as np
import cv2
import face_recognition
import pickle
import imutils
import rospy
from avatar.msg import facial_box_dim, bounding_box_array
from sensor_msgs.msg import CompressedImage
from avatar.srv import SetImagePreview, SetImagePreviewResponse
from std_msgs.msg import String, Bool

class FaceFinder:
    _VERBOSE = False

    def __init__(self, image_stream, haar, pickled, preview_stream, face_preview_stream, face_box_dimensions, resize=500):
        '''Initialize ros subscriber'''
        self._subscriber = rospy.Subscriber(image_stream, CompressedImage, self.callback,  queue_size = 10)
        self._publisher1 = rospy.Publisher(preview_stream, CompressedImage, queue_size = 10)
        self._publisher2 = rospy.Publisher(face_preview_stream, CompressedImage, queue_size = 10)
        self._publisher3 = rospy.Publisher(face_box_dimensions, bounding_box_array, queue_size = 10)
        self._data = pickle.loads(open(pickled, "rb").read())
        self._detector = cv2.CascadeClassifier(haar)
        self._resize = resize
        self._publishArray= True
        self._publishPic = True
        self._face_publishPic = True
        rospy.Service('set_preview', SetImagePreview, self.set_preview)
        rospy.loginfo(f"face_finder: subscribing to {image}")

    def set_preview(self, preview) :
        if self._VERBOSE:
            rospy.loginfo("face_finder: got a message {preview}")
        self._publishPic = preview
        return SetImagePreviewResponse("Preview set to {self._publishPic}")

    def set_preview(self, face_preview) :
        if self._VERBOSE:
            rospy.loginfo("face_finder: got a message {face_preview}")
        self._face_publishPic = face_preview
        return SetImagePreviewResponse("Preview set to {self._face_publishPic}")

    def set_preview(self, face_box_dim) :
        if self._VERBOSE:
            rospy.loginfo("face_finder: got a message {face_box_dim}")
        self._publishArray = face_box_dim
        return SetArrayPreviewResponse("Preview set to {self._publishArray}")

    def callback(self, ros_data):
        '''Callback function of subscribed topic'''

        np_arr = np.fromstring(ros_data.data, np.uint8)
        image_np = cv2.imdecode(np_arr, cv2.IMREAD_COLOR) # OpenCV >= 3.0:
        #if self._resize > 0 :
        #frame = imutils.resize(image_np, width=self._resize)
        #else :
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
        
        if self._face_publishPic :
            cropped_faces = []
            offset = 10
            for (top, right, bottom, left) in boxes:
                cropped_face = frame[top-offset:bottom+offset, left-offset:right+offset]
                cropped_face = cv2.resize(cropped_face,(100,100))
                cropped_faces.append(cropped_face)
                print("z")
            
            all_faces = np.concatenate(cropped_faces, axis = 1, out = None)
            
            msg_raw_image2 = CompressedImage()
            msg_raw_image2.header.stamp = rospy.Time.now()
            msg_raw_image2.format = "jpeg"
            msg_raw_image2.data = np.array(cv2.imencode('.jpg', all_faces)[1]).tostring()
            self._publisher2.publish(msg_raw_image2)

        if self._publishPic :
            for (top, right, bottom, left) in boxes:
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                y = top - 15 if top - 15 > 15 else top + 15
                cv2.putText(frame, name, (left, y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
            msg_raw_image1 = CompressedImage()
            msg_raw_image1.header.stamp = rospy.Time.now()
            msg_raw_image1.format = "jpeg"
            msg_raw_image1.data = np.array(cv2.imencode('.jpg', frame)[1]).tostring()
            self._publisher1.publish(msg_raw_image1)
#            cv2.imshow("Frame", frame)
#            cv2.waitKey(10)

        if self._publishArray :
            msg_bounding_box_array = bounding_box_array()
            for (x, y, w, h) in rects:
                msg_facial_box_dim = facial_box_dim()
                msg_facial_box_dim.name = name
                msg_facial_box_dim.llc_row = y
                msg_facial_box_dim.llc_col = x
                msg_facial_box_dim.height = h
                msg_facial_box_dim.width = w
                msg_bounding_box_array.bounding_box_array.append(msg_facial_box_dim)
            self._publisher3.publish(msg_bounding_box_array)
            rospy.loginfo(msg_bounding_box_array)

if __name__ == '__main__':
    '''Initializes and cleanup ros node'''
    image = rospy.get_param('~image', '/camera/image_raw/compressed')##'/camera/image_raw/compressed')
    haar = rospy.get_param('~haar', '/home/neel/catkin_ws/src/avatar/haarcascade_frontalface_default.xml')
    pickled = rospy.get_param('~pickle', '/home/neel/catkin_ws/src/avatar/faces.pickle')
    preview = rospy.get_param('_preview', '/face_finder/preview')
    face_preview = rospy.get_param('_face_preview', '/face_finder/face_preview')
    box_dimensions = rospy.get_param('_box_dimensions', '/face_finder/face_box_dimensions')

    rospy.init_node('face_finder', anonymous=False)
    ic = FaceFinder(image, haar, pickled, preview, face_preview, box_dimensions, resize=500)

    try:
        rospy.spin()
    except KeyboardInterrupt:
        rospy.loginfo("face_finder: shutting down")
#    cv2.destroyAllWindows()
