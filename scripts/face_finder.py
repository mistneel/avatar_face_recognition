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
from sensor_msgs.msg import CompressedImage
from avatar.srv import SetImagePreview, SetImagePreviewResponse
from std_msgs.msg import String, Bool

class FaceFinder:
    _VERBOSE = False

    def __init__(self, image_stream, haar, pickled, preview_stream, resize=500):
        '''Initialize ros subscriber'''
        self._subscriber = rospy.Subscriber(image_stream, CompressedImage, self.callback,  queue_size = 10)
        self._publisher = rospy.Publisher(preview_stream, CompressedImage, queue_size = 10)
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
            rospy.loginfo(f"face_finder: we found {len(rects)} faces")

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

        if self._publishPic :
            for (top, right, bottom, left) in boxes:
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                y = top - 15 if top - 15 > 15 else top + 15
                cv2.putText(frame, name, (left, y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
            msg_raw_image = CompressedImage()
            msg_raw_image.header.stamp = rospy.Time.now()
            msg_raw_image.format = "jpeg"
            msg_raw_image.data = np.array(cv2.imencode('.jpg', frame)[1]).tostring()
            self._publisher.publish(msg_raw_image)
#            cv2.imshow("Frame", frame)
#            cv2.waitKey(10)


if __name__ == '__main__':
    '''Initializes and cleanup ros node'''
    image = rospy.get_param('~image', '/camera/image_raw/compressed')##'/camera/image_raw/compressed')
    haar = rospy.get_param('~haar', '/home/neel/catkin_ws/src/avatar/haarcascade_frontalface_default.xml')
    pickled = rospy.get_param('~pickle', '/home/neel/catkin_ws/src/avatar/faces.pickle')
    preview = rospy.get_param('_preview', '/face_finder/preview')

    rospy.init_node('face_finder', anonymous=False)
    ic = FaceFinder(image, haar, pickled, preview, resize=0)

    try:
        rospy.spin()
    except KeyboardInterrupt:
        rospy.loginfo("face_finder: shutting down")
#    cv2.destroyAllWindows()
