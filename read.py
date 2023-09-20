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
        xmin, xmax, ymin, ymax, labels = [], [], [], [], []
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
                print("-------------")

    #                 print('label ',label)
                if (label == str('A') or label == str('B')or label == str('C')
                or label == str('D')or label == str('E')or label == str('F')
                or label == str('G')or label == str('H')or label == str('I')
                or label == str('J')or label == str('K')or label == str('L')
                or label == str('M')or label == str('N')or label == str('O')
                or label == str('P')or label == str('Q')or label == str('R')
                or label == str('S')or label == str('T')or label == str('U')or label == str('V')
                or label == str('W')or label == str('X')or label == str('Y')or label == str('Z')):
                    xmi.append(x)
                    ymi.append(y)
                    xma.append(x+w)
                    yma.append(y+h)
                    lc.append(label)
    #                     print("xmi ",xmi,ymi,xma,yma)

                if (label == str('0') or label == str('1')or label == str('2')or label == str('3')
                or label == str('4')or label == str('5')or label == str('6')or label == str('7')
                or label == str('8')or label == str('9')):
                    xc.append(x)
                    yc.append(y)
                    xcm.append(x+w)
                    ycm.append(y+h)
                    ln.append(label)
    #                     print("xc ",xc,yc,xcm,ycm)

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
                    
        boxes = pd.DataFrame({"xmin": xmin, "ymin": ymin, "xmax": xmax, "ymax": ymax})
        char = pd.DataFrame({"xmi": xmi, "ymi": ymi, "xma": xma, "yma": yma, "Label": lc})
        num = pd.DataFrame({"xmi": xc, "yc": yc, "xca": xcm, "yca": ycm, "Label": ln})

        char.sort_values(by=['xmi'], inplace=True)
        num.sort_values(by=['xmi'], inplace=True)

        a=char[char.columns[4]]

        b=num[num.columns[4]]
        res = a._append(b)

        d1 = pd.DataFrame({"label":res})
        d2 = d1.transpose()
        d3 = d2.values.tolist()
        flatten_mat=[]
        for sublist in d3:
            for val in sublist:
                flatten_mat.append(val)

        lp_num = ''.join(map(str,flatten_mat))
        print("LP NUM: ",lp_num)

        return lp_num, (min(confidences) if len(confidences) > 0 else 0)