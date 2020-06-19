#!/usr/bin/env python3
#
# This node subscribes to a video/webcam stream topic, detects and recognizes the face(s), and publishes an array of face(s) to the publisher topic.
# Publishing array contains header, name, row, col, height, width, and compressed image of faces. 
#
# Version 1.0 Jun 2020 - Neel Mistry
# Author - Neel Mistry (mistry.neel92@gmail.com)
#

import os.path
import argparse
import os.path
import numpy as np
import cv2
import face_recognition
import pickle
import imutils
import rospy
import json as JSON
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import String, Bool, Header
from avatar_face_recognition.msg import Face, FaceArray
from avatar_face_recognition.srv import SetImagePreview, SetImagePreviewResponse
from shapely.geometry import box as SHAPELYBOX

class FaceFinder:
    _VERBOSE = True
    _seq = 0

    def __init__(self, image_stream, haar, pickled, face_stream, json_file, resize=0):
        '''Initialize ros subscriber'''
        
        try:
            if not os.path.exists(haar):
                raise FileNotFoundError("No such file or directory: "+haar)
            
            if not os.path.exists(json_file):
                raise FileNotFoundError("No such file or directory: "+json_file)

            self._data = pickle.loads(open(pickled, "rb").read())
            
        except FileNotFoundError as e:
            print(e)
            rospy.signal_shutdown(e)
        except Exception as e:
            print(e)
            rospy.signal_shutdown(e)
            pass
        else:
            rospy.loginfo("checking info.json for entry named \"Unknown\"")
            if "Unknown" in open(json_file, "r"):
                rospy.loginfo("found info.json")
                self._json = JSON.load(open(json_file))
                
            else:
                rospy.loginfo("writing an \"Unknown\" entry in info.json")
                entry={"Unknown": {"name" : "Unknown"}}
                with open(json_file, "r+") as file:
                    data=JSON.load(file)
                    data.update(entry)
                    file.seek(0)
                    JSON.dump(data, file, indent=4)
                    file.close()
                    self._json = JSON.load(open(json_file))

        self._subscriber = rospy.Subscriber(image_stream, CompressedImage, self.callback,  queue_size = 10)
        self._publisher = rospy.Publisher(face_stream, FaceArray, queue_size = 10)
        self._resize = resize
        self._publishPic = True    
        rospy.Service('set_preview', SetImagePreview, self.set_preview)

    def set_preview(self, preview):
        if self._VERBOSE:
            rospy.loginfo("face_finder: got a message {preview}")
        self._publishPic = preview
        return SetImagePreviewResponse("Preview set to {self._publishPic}")



    def callback(self, ros_data):
        '''Callback function of subscribed topic'''
        rospy.loginfo(f"face_finder: subscribing to {image}")
        np_arr = np.fromstring(ros_data.data, np.uint8)
        image_np = cv2.imdecode(np_arr, cv2.IMREAD_COLOR) # OpenCV >= 3.0:
        #if self._resize > 0 :
        #    frame = imutils.resize(image_np, width=self._resize)
        #else :
        frame = image_np
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        rects = cv2.CascadeClassifier(haar).detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)
        

        # compute the facial embeddings for each face bounding box
        boxes = [(y, x + w, y + h, x) for (x, y, w, h) in rects]
        encodings = face_recognition.face_encodings(rgb, boxes)

        names = []     
        EuDistanceTemp = []
        EuDistanceStored = []

        for encoding in encodings:
            EuDistanceTemp = face_recognition.face_distance(self._data["encodings"], encoding)
            
            name = "Unknown"

            if min(EuDistanceTemp) < 0.5: #threshold
                i = list(EuDistanceTemp).index(min(EuDistanceTemp))
                EuDistanceStored.append(min(EuDistanceTemp))
                name = self._data["names"][i]

            names.append(name)

        SHAPELYBOXLIST = []

        for box in boxes:
            
            obj=SHAPELYBOX(box[3], box[0], box[1], box[2])
            SHAPELYBOXLIST.append(obj)

        OverlapTol = 0.65            

        if len(SHAPELYBOXLIST) > 1:
            for i, sbox1 in enumerate(SHAPELYBOXLIST[:-1]):
                for j, sbox2 in enumerate(SHAPELYBOXLIST[+1:]):
                    intersection = sbox1.intersection(sbox2) #for test (470,180,760,490)) on sample.mov    
                    if intersection.area > (OverlapTol * (min(sbox1.area, sbox2.area))):
                        if EuDistanceStored[i] > EuDistanceStored[j]:
                            SHAPELYBOXLIST.pop(i)
                            boxes.pop(i)
                            encodings.pop(i)
                            EuDistanceStored.pop(i)
                            names.pop(i)
                        else:
                            SHAPELYBOXLIST.pop(j)
                            boxes.pop(j)
                            encodings.pop(j)
                            EuDistanceStored.pop(j)
                            names.pop(j)

        if len(names) > 1:
            for i, name1 in enumerate(names[:-1]):
                for j, name2 in enumerate(names[+1:]):
                    if name1 == name2 and name1 != "Unknown":
                        if EuDistanceStored[i] > EuDistanceStored[j]:
                            SHAPELYBOXLIST.pop(i)
                            boxes.pop(i)
                            encodings.pop(i)
                            EuDistanceStored.pop(i)
                            names.pop(i)
                        else:
                            SHAPELYBOXLIST.pop(j)
                            boxes.pop(j)
                            encodings.pop(j)
                            EuDistanceStored.pop(j)
                            names.pop(j)                      



        
        if self._VERBOSE:
            rospy.loginfo(f"face_finder: we found {len(boxes)} faces")
        print(names)
        
       
        #generate image parmeter array and publish
        
        msg_faces_array = FaceArray()
        offset = 10 # boundardy around the face to crop
        for i, (top, right, bottom, left) in enumerate(boxes):
            if(top>0 and right>0 and bottom>0 and left>0):
                msg_raw_image = Face()

                cropped_face = frame[top-offset:bottom+offset, left-offset:right+offset]
                cropped_face = cv2.resize(cropped_face,(100,100))
                
                msg_raw_image.FaceImg.format = "jpeg"
                msg_raw_image.FaceImg.data = np.array(cv2.imencode('.jpg', cropped_face)[1]).tostring()
                
                #if self._VERBOSE:
                    #rospy.loginfo(f"Face {name[i]} mapps to {self._json[name}")
                msg_raw_image.json.data = JSON.dumps((self._json[names[i]])) # must be a string
                msg_raw_image.row.data = bottom
                msg_raw_image.col.data = right
                msg_raw_image.height.data = top
                msg_raw_image.width.data = left
                
                
                msg_faces_array.faces.append(msg_raw_image)
                print(msg_raw_image.json.data)
        if len(msg_faces_array.faces) > 0:
            msg_faces_array.header.stamp = rospy.Time.now()
            msg_faces_array.header.seq = self._seq
            self._seq = self._seq + 1
            rospy.loginfo(f"face_finder: Transmitting {len(msg_faces_array.faces)} faces")
            self._publisher.publish(msg_faces_array)
            

if __name__ == '__main__':
    '''Initializes and cleanup ros node'''
    image = rospy.get_param('~image', '/camera/image_raw/compressed')
    haar = rospy.get_param('~haar', '/home/neel/catkin_ws/src/avatar_face_recognition/haarcascade_frontalface_default.xml')
    pickled = rospy.get_param('~pickle', '/home/neel/catkin_ws/src/avatar_face_recognition/faces.pickle')
    jsonFile = rospy.get_param('~json', '/home/neel/catkin_ws/src/avatar_face_recognition/info.json')
    faces = rospy.get_param('~preview', '/face_finder/faces')

    rospy.init_node('face_finder', anonymous=False)
    ic = FaceFinder(image, haar, pickled, faces, jsonFile, resize=0)

    try:
        rospy.spin()
    except KeyboardInterrupt:
        rospy.loginfo("face_finder: shutting down")
#    cv2.destroyAllWindows()
