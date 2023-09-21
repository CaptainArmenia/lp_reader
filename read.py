import os
from glob import glob
import time

import cv2
from matplotlib import pyplot as plt
import numpy as np
from PIL import Image
import pandas as pd

def network (modelConfiguration, modelWeights):
    net = cv2.dnn.readNetFromDarknet(modelConfiguration, modelWeights)
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    return net

def split_classes(classesFile):
    classes = None
    with open(classesFile, 'rt') as f:
        classes = f.read().rstrip('\n').split('\n')
    return classes

def load_net():
    modelConfiguration_3 = "weights/char/yolov4_custom.cfg"
    modelWeights_3 = "weights/char/yolov4_custom_last.weights"
    classesFile_3 = "weights/char/obj.names"

    classes = split_classes(classesFile_3)
    net = network(modelConfiguration_3, modelWeights_3)

    print ("------------- LOADING ALL YOLO's -----------------")
    return classes, net

def getOutputsNames(net):
    # Get the names of all the layers in the network
    layersNames = net.getLayerNames()
    # Get the names of the output layers, i.e. the layers with unconnected outputs
    return [layersNames[i - 1] for i in net.getUnconnectedOutLayers()]

class Reader:
    def __init__(self):
        self.classes, self.net = load_net()
        self.confThreshold = 0.5
        self.nmsThreshold = 0.4
        self.inpWidth = 416      #Width of network's input image
        self.inpHeight = 416     #Height of network's input image

    def get_classes(self):
        return self.classes

    def detect(self, frame):
        blob = cv2.dnn.blobFromImage(frame, 1/255, (self.inpWidth, self.inpHeight), [0,0,0], 1, crop=False)

        # Sets the input to the network
        self.net.setInput(blob)

        # Runs the forward pass to get output of the output layers
        outs = self.net.forward(getOutputsNames(self.net))

        H, W = frame.shape[:2]
        boxes, confidences, class_IDs = [], [], []

        for output in outs:
            for detection in output:
                scores = detection[5:]
                classID = np.argmax(scores)
                confidence = scores[classID]
                if confidence > self.confThreshold:
                    box = detection[0:4] * np.array([W, H, W, H])
                    centerX, centerY, width, height = box.astype("int")
                    x, y = int(centerX - (width / 2)), int(centerY - (height / 2))
                    boxes.append([x, y, int(width), int(height)])
                    confidences.append(float(confidence))
                    class_IDs.append(classID)

        
        # Apply non-max suppression to identify best bounding box
        indices = cv2.dnn.NMSBoxes(boxes, confidences, self.confThreshold, self.nmsThreshold)
        xmin, xmax, ymin, ymax, label_ids = [], [], [], [], []

        if len(indices) > 0:
            for i in indices.flatten():
                x, y, w, h = boxes[i][0], boxes[i][1], boxes[i][2], boxes[i][3]
                xmin.append(x)
                ymin.append(y)
                xmax.append(x+w)
                ymax.append(y+h)
                label_ids.append(class_IDs[i])

        return xmin, ymin, xmax, ymax, label_ids, (min(confidences) if len(confidences) > 0 else 0)


    def read(self, frame):
        blob = cv2.dnn.blobFromImage(frame, 1/255, (self.inpWidth, self.inpHeight), [0,0,0], 1, crop=False)

        # Sets the input to the network
        self.net.setInput(blob)

        # Runs the forward pass to get output of the output layers
        outs = self.net.forward(getOutputsNames(self.net))

        H, W = frame.shape[:2]
        boxes, confidences, class_IDs = [], [], []
        
        for output in outs:
            for detection in output:
                scores = detection[5:]
                classID = np.argmax(scores)
                confidence = scores[classID]
                if confidence > self.confThreshold:
                    box = detection[0:4] * np.array([W, H, W, H])
                    centerX, centerY, width, height = box.astype("int")
                    x, y = int(centerX - (width / 2)), int(centerY - (height / 2))
                    boxes.append([x, y, int(width), int(height)])
                    confidences.append(float(confidence))
                    class_IDs.append(classID)


        # Apply non-max suppression to identify best bounding box
        indices = cv2.dnn.NMSBoxes(boxes, confidences, self.confThreshold, self.nmsThreshold)
        xmin, xmax, ymin, ymax, labels, confs = [], [], [], [], [], []
        xmi,xma,ymi,yma,lc=[],[],[],[],[]
        xc,xcm,yc,ycm,ln=[],[],[],[],[]
        if len(indices) > 0:
            for i in indices.flatten():
                x, y, w, h = boxes[i][0], boxes[i][1], boxes[i][2], boxes[i][3]
                xmin.append(x)
                ymin.append(y)
                xmax.append(x+w)
                ymax.append(y+h)
                label = str.upper((self.classes[class_IDs[i]]))
                labels.append(label),
                confs.append(confidences[i])

        # for each detection, check if box is considerably smaller than the others
        # if so, remove it
        if len(xmin) > 1:
            # calculate the mean box area
            mean_area = np.mean((np.array(xmax) - np.array(xmin)) * (np.array(ymax) - np.array(ymin)))
            for i in range(len(xmin)):
                # if box area is less than 1/2 of the mean area, remove it
                if (xmax[i] - xmin[i]) * (ymax[i] - ymin[i]) < mean_area / 2:
                    xmin.pop(i)
                    ymin.pop(i)
                    xmax.pop(i)
                    ymax.pop(i)
                    labels.pop(i)
                    confs.pop(i)
                    
        # sort boxes and confidences by xmin coordinate
        xmin, ymin, xmax, ymax, labels, confs = zip(*sorted(zip(xmin, ymin, xmax, ymax, labels, confs), key=lambda x: x[0]))
        # obtain the license plate number
        lp_num = ''.join(labels)
        

        return lp_num, (min(confs) if len(confs) > 0 else 0)