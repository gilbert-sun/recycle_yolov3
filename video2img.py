#!/usr/bin/env python
import cv2 as cv
import numpy as np
import socket
import json
import time,os,sys
import datetime,json
import parameters,glob
from subprocess import call
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
 
    blob = cv.dnn.blobFromImage(img, 1/255.0, (320, 320), swapRB=True, crop=False)
    net.setInput(blob)
    outputs = net.forward(ln)
    outputs = np.vstack(outputs)

    # IMPORTANT Run post process
    post_process(img, outputs, conf)
    cv.putText(img,str(counter1),(550,30),0,1,(0,255,255),2)
    cv.imshow('RealSenseRGB1',  img) 

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

        if confidence > conf:
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
    distThres = 50

    # classes = open(r'models\classes.txt').read().strip().split('\n')#win10
    classes = open(r'./yolov3_cfg/obj.names').read().strip().split('\n')#ubuntu
    # classes = open(r'/home/gilbert3/darknet/data/coco.names').read().strip().split('\n')
    #classes = open(r'/media/gilbert3/mx500_1/Downloads/yuching7petv4/obj.names').read().strip().split('\n')
    np.random.seed(42)
    colors = np.random.randint(0, 255, size=(len(classes), 3), dtype='uint8')

    net = cv.dnn.readNetFromDarknet(r'./yolov3_cfg/yolov3.cfg', r'./yolov3_cfg/weights/yolov3_final.weights')# ubuntu

    net.setPreferableTarget(cv.dnn.DNN_TARGET_OPENCL)
    net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)#DNN_BACKEND_CUDA)#CUDA / OPENCV DNN_BACKEND_INFERENCE_ENGINE  on Openvino
    net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU) #CUDA / CPU


    ln = net.getLayerNames()
    ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]


def cp_labelTxt_toLabel_folder():

    if not os.path.exists(parameters.video2img_out_labelPath):
            os.makedirs(parameters.video2img_out_labelPath)

    executeCmd1 = "cp -r {}/*.txt  {}/ ".format(
        parameters.video2img_out_predictPath,
        parameters.video2img_out_labelPath )

    executeCmd2 = "rm -fr {}/yolov3_final.txt".format(parameters.video2img_out_labelPath)
    print("\n===>",executeCmd1,"<----\n")
    print("\n===>",executeCmd2,"<----\n")
    os.system(executeCmd1)
    # call(executeCmd1.split())
    time.sleep(1)

    os.system(executeCmd2)
    time.sleep(1)

def parserPredict_body():
    print("-------------------Entering parser yolo testing file!\n")
    tagFile = ""
    tmpbuf = ""
    Wbase = 639
    Hbase = 479
    #
    with open( parameters.video2img_out_predictPath+"/yolov3_final.txt") as fs:
        tmpbuf = fs.readlines()
        count =0
        for fs_idx in tmpbuf:
            # print(fs_idx)
            if fs_idx.find("/media") == 0:
                fn = fs_idx.split("/")[-1].split(".jpg")[0]
                # print(count,":",fn, type(fn))
                # count=count+1
                fn = os.path.join(parameters.video2img_out_predictPath , fn + ".txt")
                open(fn,"w")
            else:
                # print(fs_idx.split(" "))
                fc = fs_idx.split(" ")
                with open(fn,"a") as fss:
                    # C ,x1,y1 x2,y2,conf ==> 5 , abs(x1-x2)/2/ ,abs(y1-y2)/2 ,abs((x2-x1)/1920) , abs((y2-y1)/1080)
                    cl = parameters.classType
                    x0 = str( (int(fc[1]) + (abs( int(fc[3])-int(fc[1])) /2))/Wbase ) + " "
                    y0 = str( (int(fc[2]) + (abs( int(fc[4])-int(fc[2])) /2))/Hbase) + " "
                    w0 = str( abs(int(fc[3])-int(fc[1]))/Wbase) + " "
                    h0 = str( abs(int(fc[4])-int(fc[2]))/Hbase) + "\n"
                    fss.write(cl + x0 + y0 + w0 + h0)
    cp_labelTxt_toLabel_folder()
    print("-------------------Finished!\n")

def Out_ImgList_Predict_txt():

        print("\n----------Output video2img List.txt :-----------\n")
        folderList =glob.glob(parameters.video2img_out_imgPath+"/*.jpg")

        with open(os.path.join(parameters.cfgFolder,parameters.video2img_List),"w") as fs:
            [fs.write(idx+"\n") for idx in folderList ]

        print("\n----------Execute darknet testing command:-----------\n")
        if not os.path.exists(parameters.video2img_out_predictPath):
            os.makedirs(parameters.video2img_out_predictPath)

        executeCmd = "{} detector test_v2 {} {} {} {} {} -out {} -thresh {}".format(
                 parameters.darknetEcec,
                 (parameters.cfgFolder  + "/obj.data"),
                 (parameters.cfgFolder + '/yolov3.cfg'),
                 (parameters.cfgFolder  + "/weights/yolov3_final.weights"),
                 (parameters.cfgFolder + "/" +parameters.video2img_List),
                 (parameters.video2img_out_predictPath+"/yolov3_final.txt"),
                 (parameters.video2img_out_predictPath + "/"),
                 parameters.threshValue
                 )

        print("          " + executeCmd)

        time.sleep(3)
        call(executeCmd.split())
 
def run():
    '''
        Main function
    '''
    #========================================================================gilbert_start
    global rs_config,counter1,old_timetag_conf
    counter1 = 0
    old_timetag_conf = int(datetime.datetime.now().timestamp() * 1000)
    # rs_config.pipeline.start(rs_config.config)
    #========================================================================gilbert_end
    load_detector()

 
    cv.namedWindow('RealSenseRGB1')
    cv.createTrackbar('confidence', 'RealSenseRGB1', 50, 100, trackbarConfidence)
    cv.createTrackbar('distance threshold', 'RealSenseRGB1', 50, 100, trackbarThres)


    if(True):
 
        if not os.path.isfile(parameters.video2img_sour):
            print("\n-------Err: no such file or sd-card not found!!\n")
            sys.exit(1)

        cap = cv.VideoCapture(parameters.video2img_sour)

        if not os.path.exists(parameters.video2img_out_imgPath):
            os.makedirs(parameters.video2img_out_imgPath)

        # while(True):
        while(cap.isOpened()):
            # _, img = cap.read()
            try :
                ret,img = cap.read()
                counter1 +=1

                # print('Server connected {}:'.format(counter1))
     
                # process(img)

                # data = json.dumps(centers).encode('utf-8') # IMPORTANT Prepare data to send

                if parameters.out_frame == False:
                    cv.putText(img,str(counter1),(500,30),0,1,(0,250,220),2)

                if counter1 > parameters.out_frameID and parameters.out_frame == True:
                    cv.imwrite(parameters.video2img_out_imgKind+str(counter1)+".jpg", img)



                cv.imshow('RealSenseRGB1',  img)
             
                Keyvalue = cv.waitKey(1)
                if Keyvalue==27:
                    cap.release()
                    cv.destroyAllWindows()
                    break 
            except:
                print("--------Play Video Ending!\n")
                cap.release()
                cv.destroyAllWindows()
                break


        Out_ImgList_Predict_txt()

        cv.destroyAllWindows()

        parserPredict_body() 

if __name__ == "__main__": 
    run()
