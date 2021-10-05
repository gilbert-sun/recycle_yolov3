#!/usr/bin/env python
#from cv2 \
import cv2 as cv

import numpy as np
import socket
import json
import time,os,sys
import datetime,json
#========================================================================gilbert_start

def switch(var, x=None):
    return {
        '.P': lambda x: 'P',
        '.COLOR': lambda x: 'COLOR',
        '.SOY': lambda x: 'SOY',
        '.OIL': lambda x: 'OIL',
        '.TRAY': lambda x: 'TRAY',
        '.CH': lambda x: 'CH',
        '.OTHER': lambda x: 'OTHER',
        '.PET': lambda x: 'PET',
        '.PP': lambda x: 'PP',
        '.PS': lambda x: 'PS',
        '.PLA': lambda x: 'PLA',
        '.PC': lambda x: 'PC',
        '.PVC': lambda x: 'PVC',
        '.RobotArm': lambda x: 'RobotArm',
        '.VisionSys': lambda x: 'VisionSys',
        '.ConveySys': lambda x: 'ConveySys',
        '.ControSys': lambda x: 'ControSys',
        '.good': lambda x: 'good',
        '.bad': lambda x: 'bad',
        'P': lambda x: 0,
        'Oil': lambda x:1,
        'Soy': lambda x: 2,
        'Color': lambda x: 3,
        'Other': lambda x: 4,
        'Ch': lambda x: 5,
        'Tray': lambda x: 6,
    }[str(var)](x)
# path = "/rms_root/catkin_ws/src/rms_pet_yolov3/scripts"
# import sys
# sys.path.append(path)
# import realsenseconfig as rs_config
# from Objection import Objection,Rcet

#========================================================================gilbert_end
def trackbarConfidence(x):
    '''
        Dynamically set YOLO confidence factor
        Inputs:
            x -> Trackbar position (not lower than 10)
    '''
    global conf
    conf = x/100

def trackbarThres(x):
    '''
        Dynamically set center-elimination threshold
        Inputs:
            x -> Trackbar position
    '''
    global distThres
    distThres = x

def get_unrepeated_index(centers):
    '''
        Eliminate centers with similar coords based on the threshold parameters (distThres)
        Inputs:
            centers -> Array of points
        Outputs:
            results -> Indexes of the unrepeated centers
    '''
    global distThres

    nwCenters = centers.copy()

    # X
    for center in nwCenters:
        dists = np.array([np.abs(center[0] - cnt[0]) for cnt in nwCenters])
        cond = dists < distThres
        indexes = list(np.where(cond == True))[0]
        for index in sorted(indexes[1:], reverse=True):
            del nwCenters[index]

            # Y
    for center in nwCenters:
        dists = np.array([np.abs(center[1] - cnt[1]) for cnt in nwCenters])
        cond = dists < distThres
        indexes = list(np.where(cond == True))[0]
        for index in sorted(indexes[1:], reverse=True):
            del nwCenters[index]

    results = []
    for center in nwCenters:
        results.append(centers.index(center))

    return results

def sort_indexes(nwCenters):
    '''
        Sort centers based on X location ascendently.
        Inputs:
            nwCenters: array of points
        Outputs:
            indexes: sorted indexes corresponding to the sorted nwCenters
    '''
    indexes = []

    if nwCenters:
        nwCenters = np.array(nwCenters)
        xPoses = nwCenters[:, 0]
        srtPoses = sorted(xPoses)

        for pose in srtPoses:
            indexes.append(np.where(xPoses==pose)[0][0])

    return indexes

def process(img):
    '''
        Process a image throught the YOLO net and runs the postprocess
        Inputs:
            img -> An image
    '''
    global net, ln, conf,rs_config,counter1
    # print(rs_config.D435_para.color_to_depth_translation)

    # 320, 416, 512, 608
    # blob = cv.dnn.blobFromImage(img, 1.0 / 255.0, (416, 416), None, True, False) #

    # cv.imshow('RealSenseRGB1',  img)
    blob = cv.dnn.blobFromImage(img, 1/255.0, (320, 320), swapRB=True, crop=False)
    net.setInput(blob)
    outputs = net.forward(ln)
    outputs = np.vstack(outputs)

    # IMPORTANT Run post process
    post_process(img, outputs, conf)
    cv.putText(img,str(counter1),(550,30),0,1,(0,255,255),2)
    cv.imshow('RealSenseRGB1',  img)
    #========================================================================gilbert_start
    # cv.imshow('RealSenseDepth1',rs_config.D435_para.depthmat)
    #========================================================================gilbert_end
def post_process(img, outputs, conf):
    global colors, clases, centers,rs_config

    H, W = img.shape[:2]

    boxes = []
    confidences = []
    classIDs = []
    centers = []

    # IMPORTANT From YOLO output to boxes, confidences, classes, centers
    for output in outputs:
        scores = output[5:]
        classID = np.argmax(scores)
        confidence = scores[classID]

        if confidence > conf  and classID.tolist() != 0 :#and classID.tolist() < 6  :
            x, y, w, h = output[:4] * np.array([W, H, W, H])
            # p0 = int(x - w//2), int(y - h//2)
            # boxes.append([*p0, int(w), int(h)])

            boxes.append([int(x - w//2), int(y - h//2), int(w), int(h)])
            confidences.append(float(confidence))
            classIDs.append(classID)
            centers.append([int(x),int(y)]) # IMPORTANT Getting object centers

    nwboxes = []
    nwConfidences = []
    nwClassIDs = []
    nwCenters = []
    nw2Centers = []

    indexes = get_unrepeated_index(centers)
    for i in indexes:
        nwboxes.append(boxes[i])
        nwConfidences.append(confidences[i])
        nwClassIDs.append(classIDs[i])
        nwCenters.append(centers[i])

    boxes.clear()
    confidences.clear()
    classIDs.clear()
    centers.clear()

    indexes = sort_indexes(nwCenters)
    for i in indexes:
        boxes.append(nwboxes[i])
        confidences.append(nwConfidences[i])
        classIDs.append(nwClassIDs[i])
        centers.append(nwCenters[i])
        nw2Centers.append(nwCenters[i])

    indices = cv.dnn.NMSBoxes(boxes, confidences, conf, conf-0.1)

    if len(indices) > 0:
        timestamp = time.time()
        for i in indices.flatten():
            Objection_vec=[]
            # rs_config.D435_para.refresh_mat()
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])
            # z = Objection(Rcet(x,x+w,y,y+h),classes[classIDs[i]])
            color = [int(c) for c in colors[classIDs[i]]]
            cv.rectangle(img, (x, y), (x + w, y + h), color, 2)
            text = "{}: {:.4f}".format(classes[classIDs[i]], confidences[i])
            labelSize, baseLine = cv.getTextSize(text, cv.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            top = max(y, labelSize[1])

            #========================================================================gilbert_start
            t, _ = net.getPerfProfile()
            fps = 1000 / (t * 1000.0 / cv.getTickFrequency())
            label = 'FPS: %.2f' % fps
            cv.putText(img, label, (25, 25), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0,255),2)
            #========================================================================gilbert_end

            cv.rectangle(img, (x, y - labelSize[1]), (x + labelSize[0], y + baseLine), (255, 255, 255), cv.FILLED)
            cv.putText(img, text, (x, y), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1)
            cv.circle(img, tuple(centers[i]), 5, color, -1)  # IMPORTANT Drawing centers
            # cv.imwrite('GH012601_8760.jpg_'+str(counter1)+".jpg", img)
            #========================================================================gilbert_start
            Position='(%d %d %d %s)' %(centers[i][0],centers[i][1],0 , classes[classIDs[i]] )
            cv.putText(img, Position, (int(centers[i][1]), int(centers[i][0])), cv.FONT_HERSHEY_SIMPLEX, 0.8, color,2)
            old_x = int(centers[i][0])
            old_y = int(centers[i][1])
            zDepth = 0
            centers[i].clear()
            nw2Centers[i].clear()
            centers[i]= [old_x,old_y,zDepth, str(classes[classIDs[i]]),str(('%.3f' % confidences[i]))+str("%")]
            nw2Centers[i]= [old_x,old_y,zDepth, str(classes[classIDs[i]]),str(('%.3f' % confidences[i]))+str("%")]

        nw2Centers.insert(0,timestamp)
        centers.insert(0,timestamp)
        #========================================================================gilbert_end
        # Put efficiency information.

def load_detector():
    '''
        Initializes variables and YOLO net
    '''
    global colors, classes, centers
    global ln, conf, net, distThres,rs_config

    ### YOLO ###
    conf = 0.1
    distThres = 10

    # classes = open(r'models\classes.txt').read().strip().split('\n')#win10
    classes = open(r'./yolov3_cfg/obj.names').read().strip().split('\n')#ubuntu
    # classes = open(r'/home/gilbert3/darknet/data/coco.names').read().strip().split('\n')
    #classes = open(r'/media/gilbert3/mx500_1/Downloads/yuching7petv4/obj.names').read().strip().split('\n')
    np.random.seed(42)
    colors = np.random.randint(0, 255, size=(len(classes), 3), dtype='uint8')

    # net = cv.dnn.readNetFromDarknet(r'models\yolov4-tiny.cfg', r'models\yolov4-tiny.weights')
    # net = cv.dnn.readNetFromDarknet(r'models\yolov3.cfg', r'models\yolov3.weights')#win10

    net = cv.dnn.readNetFromDarknet(r'./yolov3_cfg/yolov3.cfg', r'./yolov3_cfg/weights/yolov3_final_0915.weights')#custom-yolov4-detector.weights')# ubuntu

    # net = cv.dnn.readNetFromDarknet(r'./models/yolov4-custom.cfg', r'./models/yolov4-custom_best.weights')# ubuntu

    # net = cv.dnn.readNetFromDarknet(r'/home/gilbert3/darknet/cfg/yolov3.cfg', r'/home/gilbert3/darknet/yolov3.weights')
    #net = cv.dnn.readNetFromDarknet(r'/media/gilbert3/mx500_1/Downloads/yuching7petv4/yolov4-custom.cfg', r'/media/gilbert3/mx500_1/Downloads/yuching7petv4/weights/yolov4-custom_best.weights')
    net.setPreferableTarget(cv.dnn.DNN_TARGET_OPENCL)
    net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)#DNN_BACKEND_CUDA)#CUDA / OPENCV DNN_BACKEND_INFERENCE_ENGINE  on Openvino
    net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU) #CUDA / CPU


    ln = net.getLayerNames()
    ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]
def readSink_confidence():
    global old_timetag_conf,conf1
    now_timetag1 = int(datetime.datetime.now().timestamp() * 1000) - 3000
    if (now_timetag1 > old_timetag_conf):
        try:
            with open("C:/Users/e300/PetSinkSetting.json") as fs:
                data = json.load(fs)
                conf1[switch(data["SinkA_class"])] = 0.01 * data["SinkA_conf"]
                conf1[switch(data["SinkB_class"])] = 0.01 * data["SinkB_conf"]
                conf1[switch(data["SinkC_class"])] = 0.01 * data["SinkC_conf"]
                conf1[switch(data["SinkD_class"])] = 0.01 * data["SinkD_conf"]
                print(conf1)
        except:
            print("File not Found or Other Error!\n")
            conf1 = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
            print(">",conf1)
        old_timetag_conf = int(datetime.datetime.now().timestamp() * 1000)

def run():
    '''
        Main function
    '''

    #========================================================================gilbert_start
    global rs_config,counter1,centers

    counter1 = 0
    centers = 0
    # rs_config.pipeline.start(rs_config.config)
    #========================================================================gilbert_end
    load_detector()

    # IMPORTANT TCP IP connection
    HOST = '127.0.0.1'  # The server's hostname or IP address
    PORT = 21           # The port used by the server
    #========================================================================gilbert_start
    # cv.namedWindow('RealSenseDepth1')
    #========================================================================gilbert_end
    cv.namedWindow('RealSenseRGB1')
    cv.createTrackbar('confidence', 'RealSenseRGB1', 50, 100, trackbarConfidence)
    cv.createTrackbar('distance threshold', 'RealSenseRGB1', 50, 100, trackbarThres)


    if(True):
        # s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        # s.connect((HOST, PORT))
        # print('Server connected')
        # cap = cv.VideoCapture(6)



        if not os.path.isfile('./test_case_0924-C.avi'):
            print("\n-------Err: no such file or sd-card not found!!\n")
            sys.exit(1)#test_case_0924-D
        cap = cv.VideoCapture('./test_case_0924-C.avi')
        # while(True):
        while(cap.isOpened()):
            # _, img = cap.read()
            ret,img = cap.read()
            counter1 +=1
            print('\nFrame_ID: {} :'.format(counter1))
            if counter1 > 200:
                process(img)
            data = json.dumps(centers).encode('utf-8') # IMPORTANT Prepare data to send

            if data != b'[]':
                print("1.(time,x,y,z,class,confidence):> ",data)
            else:
                print ("2.(x,y,z):> ",data)
            # ch ==> 240
            # oil==> 40
            # if counter1 > 35:
            #     cv.imwrite('oil_'+str(counter1)+".jpg",img)
            #========================================================================gilbert_start
            Keyvalue = cv.waitKey(1)

            if Keyvalue==27:
                cap.release()
                cv.destroyAllWindows()
                break
            os.system("pause")
        # break
        #========================================================================gilbert_end
if __name__ == "__main__":
    run()