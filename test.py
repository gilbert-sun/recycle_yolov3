import parameters
import performance
import glob, os
import random
import os.path
import time
from shutil import copyfile
from subprocess import call
import cv2
from xml.dom import minidom
from os.path import basename
import numpy as np
# from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

#--------------------------------------------------------------------
folderCharacter = "/"  # \\ is for windows




#---------------------------------------------------------------------
def CFG_2_outputfolder():
    outputPath = parameters.outputPath  # 'predict'
    if not os.path.exists(outputPath + "/yolov3_cfg"):
        os.makedirs(outputPath + "/yolov3_cfg")
        os.makedirs(outputPath + "/yolov3_cfg/weights")
    executeCmd1 = "cp -r {} {} {}  {}/ ".format(
        os.path.join(parameters.cfgFolder ,  "obj.data"),
        os.path.join(parameters.cfgFolder, "obj.names"),
        os.path.join(parameters.cfgFolder, 'yolov3.cfg'),
        outputPath + "/yolov3_cfg"
    )
    executeCmd2 = "cp -r {}  {}/ ".format(
        os.path.join(parameters.cfgFolder , "weights/yolov3_final.weights"),
        outputPath + "/yolov3_cfg/weights"
    )
    call(executeCmd1.split())
    time.sleep(3)
    call(executeCmd2.split())
    time.sleep(3)

def downloadPretrained(url):
    import wget
    print("Downloading the pretrained model darknet53.conv.74, please wait.")
    wget.download(url)

def transferYolo( xmlFilepath, imgFilepath, labelGrep=""):
    global imgFolder
    
    img_file, img_file_extension = os.path.splitext(imgFilepath)
    img_filename = basename(img_file)
    #print(imgFilepath)
    img = cv2.imread(imgFilepath)
    imgShape = img.shape
    #print (img.shape)
    img_h = imgShape[0]
    img_w = imgShape[1]

    labelXML = minidom.parse(xmlFilepath)
    labelName = []
    labelXmin = []
    labelYmin = []
    labelXmax = []
    labelYmax = []
    totalW = 0
    totalH = 0
    countLabels = 0

    # tmpArrays = labelXML.getElementsByTagName("filename")
    # for elem in tmpArrays:
    #     filenameImage = elem.firstChild.data
    #
    # tmpArrays = labelXML.getElementsByTagName("name")
    # for elem in tmpArrays:
    #     labelName.append(str(elem.firstChild.data))
    #
    # tmpArrays = labelXML.getElementsByTagName("xmin")
    # for elem in tmpArrays:
    #     labelXmin.append(int(elem.firstChild.data))
    #
    # tmpArrays = labelXML.getElementsByTagName("ymin")
    # for elem in tmpArrays:
    #     labelYmin.append(int(elem.firstChild.data))
    #
    # tmpArrays = labelXML.getElementsByTagName("xmax")
    # for elem in tmpArrays:
    #     labelXmax.append(int(elem.firstChild.data))
    #
    # tmpArrays = labelXML.getElementsByTagName("ymax")
    # for elem in tmpArrays:
    #     labelYmax.append(int(elem.firstChild.data))
    #
    # yoloFilename = saveYoloPath + folderCharacter + img_filename + ".txt"
    # #print("writeing to {}".format(yoloFilename))
    #
    # with open(yoloFilename, 'a') as the_file:
    #     i = 0
    #     for className in labelName:
    #         if(className==labelGrep or labelGrep==""):
    #             classID = classList[className]
    #             x = (labelXmin[i] + (labelXmax[i]-labelXmin[i])/2) * 1.0 / img_w
    #             y = (labelYmin[i] + (labelYmax[i]-labelYmin[i])/2) * 1.0 / img_h
    #             w = (labelXmax[i]-labelXmin[i]) * 1.0 / img_w
    #             h = (labelYmax[i]-labelYmin[i]) * 1.0 / img_h
    #
    #             the_file.write(str(classID) + ' ' + str(x) + ' ' + str(y) + ' ' + str(w) + ' ' + str(h) + '\n')
    #             i += 1
    #
    # the_file.close()

#---------------------------------------------------------------
# fileCount = 0
#
# print("Step 1. Transfer VOC dataset to YOLO dataset.")
# for file in os.listdir(imgFolder):
#     filename, file_extension = os.path.splitext(file)
#     file_extension = file_extension.lower()
#
#     if(file_extension == ".jpg" or file_extension==".png" or file_extension==".jpeg" or file_extension==".bmp"):
#         imgfile = imgFolder + folderCharacter + file
#         xmlfile = xmlFolder + folderCharacter + filename + ".xml"
#
#         if(os.path.isfile(xmlfile)):
#             #print("id:{}".format(fileCount))
#             #print("processing {}".format(imgfile))
#             #print("processing {}".format(xmlfile))
#             fileCount += 1
#
#             transferYolo( xmlfile, imgfile, "")
#             copyfile(imgfile, saveYoloPath + folderCharacter + file)
#
# print("        {} images transered.".format(fileCount))

def main():
    fileList = []
    outputTrainFile = parameters.cfgFolder + "/train.txt"
    outputTestFile = parameters.cfgFolder + "/test.txt"
    classes = len(parameters.classList)

    if not os.path.exists(os.path.join(parameters.cfgFolder , "weights")):
        os.makedirs(os.path.join(parameters.cfgFolder , "weights"))
        print("all weights will generated in here: " + parameters.cfgFolder + folderCharacter + "weights" + folderCharacter)

    with open(os.path.join(parameters.cfgFolder , parameters.cfg_obj_data), 'w') as the_file:
        the_file.write("classes= " + str(classes) + "\n")
        the_file.write("train  = " + os.path.join(parameters.cfgFolder , "train.txt\n"))
        the_file.write("valid  = " + os.path.join(parameters.cfgFolder , "test.txt\n"))
        the_file.write("names = " + os.path.join(parameters.cfgFolder , "obj.names\n"))
        the_file.write("backup = " + os.path.join(parameters.cfgFolder , "weights/"))

    with open(os.path.join(parameters.cfgFolder ,parameters.cfg_obj_names), 'w') as the_file:
        classList_sorted = {name: i for i, name in parameters.classList.items()}
        for class_id in range(classes):
            the_file.write(classList_sorted[class_id] + "\n")

        # step4 ----------------------------------------------------

        print("Step 4. Start to Test the YOLO model.")

        # if not os.path.exists("darknet53.conv.74"):
        #     downloadPretrained("https://pjreddie.com/media/files/darknet53.conv.74")

        classNum = len(parameters.classList)
        filterNum = (classNum + 5) * 3

        fileCFG = "yolov3.cfg"
        # classNum = len(parameters.classList)
        # filterNum = (classNum + 5) * 3
        # if (parameters.modelYOLO == "yolov3"):
        #     fileCFG = "yolov3.cfg"

        # else:
        #     fileCFG = "yolov3-tiny.cfg"

        with open(os.path.join(parameters.cfgFolder , fileCFG)) as file:
            file_content = file.read()

        file_updated = file_content.replace("{BATCH}", str(parameters.numBatch_test))
        file_updated = file_updated.replace("{SUBDIVISIONS}", str(parameters.numSubdivision_test))
        file_updated = file_updated.replace("{FILTERS}", str(filterNum))
        file_updated = file_updated.replace("{CLASSES}", str(classNum))
        file_updated = file_updated.replace("{max_batches}", parameters.max_batches)
        file_updated = file_updated.replace("{steps}", parameters.steps)
        file_updated = file_updated.replace("{ANCHORS}", parameters.anchors)

        file = open(os.path.join(parameters.cfgFolder , fileCFG), "w")
        file.write(file_updated)
        file.close()
###################################### yolov3_cfg/yolov3.cfg change ok __end
        # gt_txt = 'gt.txt'
        # gt_file = open(gt_txt, 'w')
        # gt_file.close()
        # gt_file = open(gt_txt, 'a')
        #
        # fp = open(outputTestFile, 'r')
        # lines = fp.readlines()
        # # read test.txt filename mapping 2 gt.txt with corresponding answer together ==> gt.txt
        # for i in range(len(lines)):
        #     gt_file.write(lines[i])
        #     image_type = lines[i].split('/')[-1].split('.')[-1]
        #     txt = lines[i].replace('images', 'labels').replace(image_type, 'txt')
        #     fp2 = open(txt, 'r')
        #     lines_2 = fp2.readlines()
        #     if lines_2 != []:
        #         [h, w, ch] = cv2.imread(lines[i].split('\n')[0]).shape
        #         for j in range(len(lines_2)):
        #             class_id = lines_2[j].split(' ')[0]
        #             x1 = round(float(lines_2[j].split(' ')[1]) * w - float(lines_2[j].split(' ')[3]) / 2 * w)
        #             y1 = round(float(lines_2[j].split(' ')[2]) * h - float(lines_2[j].split(' ')[4]) / 2 * h)
        #             x2 = round(float(lines_2[j].split(' ')[1]) * w + float(lines_2[j].split(' ')[3]) / 2 * w)
        #             y2 = round(float(lines_2[j].split(' ')[2]) * h + float(lines_2[j].split(' ')[4]) / 2 * h)
        #             gt_file.write(classList_sorted[int(class_id)] + ' ' \
        #                           + str(x1) + ' ' + str(y1) + ' ' \
        #                           + str(x2) + ' ' + str(y2) + '\n')
        # gt_file.close()
        # # ================================================
        # predict_path = 'predict'
        # if not os.path.exists(predict_path):
        #     os.makedirs(predict_path)
        #
        # fp = open(parameters.cfgFolder + folderCharacter + parameters.cfg_obj_data, 'r')
        # lines = fp.readlines()
        # for i in range(len(lines)):
        #     if lines[i].split(' = ')[0] == 'backup':  weight_path = lines[i].split(' = ')[1].split('\n')[0]
        #
        # weight_file = {}
        # for filename in os.listdir(weight_path):
        #     if filename.split('.')[-1] == 'weights':
        #         if filename.split('_')[-1].split('.')[0] != 'final':
        #             weight_file[int(filename.split('_')[-1].split('.')[0])] = filename
        #         else:
        #             weight_file[int(parameters.max_batches)] = filename
        # index = sorted(weight_file.keys())
        # print("----test.main():: {}\n : {}\n  : {}\n----".format(weight_path, index,weight_file))
        # ================================================
        # executeCmd = darknetEcec + " detector train " + cfgFolder + folderCharacter + "obj.data " + cfgFolder + folderCharacter + fileCFG + " darknet53.conv.74"
        #
        # print("        execute darknet training command:")ANCHERS
        # print("          " + executeCmd)
        # print("")
        # print("        you can find all the weights files here:" + cfgFolder + folderCharacter + "weights" + folderCharacter)
        #
        # time.sleep(3)
        # call(executeCmd.split())
        testPatten = open(parameters.cfgFolder + "/test.txt", 'w')
        testPatten.close()
        testPatten = open(parameters.cfgFolder + '/test.txt', 'a')
        for tmp in os.listdir(parameters.testPathImage):
            testPatten.write(parameters.testPathImage + "/" + tmp + '\n')
        testPatten.close()
        # ================================================
        gt_txt = parameters.gt_txt
        gt_file = open(gt_txt, 'w')
        gt_file.close()
        gt_file = open(gt_txt, 'a')

        fp = open(outputTestFile, 'r')
        lines = fp.readlines()

        for i in range(len(lines)):
            gt_file.write(lines[i])
            image_type = lines[i].split('/')[-1].split('.')[-1]
            txt = lines[i].replace('images', parameters.testPathLabel.split("/")[-1]).replace(image_type, 'txt')
            # cp test.txt all *.jpg correspond answer ("labels_xxx/*.txt") to gt.txt
            # parameters.testPathLabel.split("/")[-1] == "labels"
            # print ("\n----{} :txt :{}:\n{}:-------\n".format(i,lines[i].split('.')[0],txt))
            fp2 = open(txt, 'r')
            lines_2 = fp2.readlines()
            if lines_2 != []:
                [h, w, ch] = cv2.imread(lines[i].split('\n')[0]).shape
                for j in range(len(lines_2)):
                    class_id = lines_2[j].split(' ')[0]
                    x1 = round(float(lines_2[j].split(' ')[1]) * w - float(lines_2[j].split(' ')[3]) / 2 * w)
                    y1 = round(float(lines_2[j].split(' ')[2]) * h - float(lines_2[j].split(' ')[4]) / 2 * h)
                    x2 = round(float(lines_2[j].split(' ')[1]) * w + float(lines_2[j].split(' ')[3]) / 2 * w)
                    y2 = round(float(lines_2[j].split(' ')[2]) * h + float(lines_2[j].split(' ')[4]) / 2 * h)
                    gt_file.write(classList_sorted[int(class_id)] + ' ' \
                                  + str(x1) + ' ' + str(y1) + ' ' \
                                  + str(x2) + ' ' + str(y2) + '\n')
                    # print(
                    #     "{}\n {}\n {}\n {}\n {}\n {}\n".format(lines_2[j], classList_sorted[int(class_id)], (x1), (y1),
                    #                                            (x2), (y2)))

        gt_file.close()
        # ================================================
        predict_path = parameters.predict_txt #'predict'
        if not os.path.exists(predict_path):
            os.makedirs(predict_path)

        fp = open(os.path.join(parameters.cfgFolder , parameters.cfg_obj_data), 'r')
        lines = fp.readlines()
        for i in range(len(lines)):
            if lines[i].split(' = ')[0] == 'backup':  
                weight_path = lines[i].split(' = ')[1].split('\n')[0]

        weight_file = {}
        for filename in os.listdir(weight_path):
            if filename.split('.')[-1] == 'weights':
                if filename.split('_')[-1].split('.')[0] != 'final':
                    weight_file[int(filename.split('_')[-1].split('.')[0])] = filename
                else:
                    weight_file[int(parameters.max_batches)] = filename
        predict_txt = os.path.join(parameters.savePridictPath ,weight_file[int(parameters.max_batches)].split('.')[0] + '.txt')

        # print("\n\n---------000 entry test main----------:{}:\n\n".format(predict_txt))
        executeCmd = "{} detector test_v2 {} {} {} {} {} -out {} -thresh {}".format(
                 parameters.darknetEcec, 
                 (os.path.join(parameters.cfgFolder,"obj.data")),
                 (os.path.join(parameters.cfgFolder ,'yolov3.cfg')),
                 weight_path + weight_file[int(parameters.max_batches)], 
                 outputTestFile, 
                 predict_txt,
                 (parameters.savePridictPath + "/"),
                 parameters.threshValue
                 )

        print("        execute darknet testing command:")
        print("          " + executeCmd)
        print("")
        print(
            "        you can find all the weights files here:" + parameters.cfgFolder + folderCharacter + "weights" + folderCharacter)

        time.sleep(3)
        CFG_2_outputfolder()
        return executeCmd





if __name__ == '__main__':
    # step1 ---------------------------------------------------------------

    # if not os.path.exists(saveYoloPath):
    #     os.makedirs(saveYoloPath)
    #     print(" saveYolo+ img + label ")
    # for file in os.listdir(imgFolder):
    #     imgfile = imgFolder + folderCharacter + file
    #     #print (file, ":",imgfile )
    #     copyfile(imgfile, saveYoloPath + folderCharacter + file)
    # for file in os.listdir(labelFolder):
    #     labelfile = labelFolder + folderCharacter + file
    #     copyfile(labelfile, saveYoloPath + folderCharacter + file)

    # step 1.4
    testPatten = open(parameters.cfgFolder + "/test.txt",'w')
    testPatten.close()
    testPatten = open(parameters.cfgFolder+ '/test.txt','a')
    for tmp in os.listdir(parameters.testPathImage):
        testPatten.write(parameters.testPathImage+"/"+tmp+'\n')
    testPatten.close()

    # step2 ---------------------------------------------------------------
    fileList = []
    outputTrainFile = parameters.cfgFolder + "/train.txt"
    outputTestFile = parameters.cfgFolder + "/test.txt"

    # print("Step 2. Create YOLO cfg folder and split dataset to train and test datasets.")
    # if not os.path.exists(cfgFolder):
    #     os.makedirs(cfgFolder)
    #
    # for file in os.listdir(saveYoloPath):
    #     filename, file_extension = os.path.splitext(file)
    #     file_extension = file_extension.lower()

    #     if(file_extension == ".jpg" or file_extension==".jpeg" or file_extension==".png" or file_extension==".bmp"):
    #         fileList.append(saveYoloPath + folderCharacter + file)

    # testCount = int(len(fileList) * testRatio)
    # trainCount = len(fileList) - testCount
    #
    # a = range(len(fileList))
    # test_data = random.sample(a, testCount)
    # train_data = random.sample(a, trainCount)
    #
    # f = open(outputTrainFile, 'w')
    # f.close()
    # with open(outputTrainFile, 'a') as the_file:
    #     for i in train_data:
    #         the_file.write(fileList[i] + "\n")
    #
    # f = open(outputTestFile, 'w')
    # f.close()
    # with open(outputTestFile, 'a') as the_file:
    #     for i in test_data:
    #         the_file.write(fileList[i] + "\n")
    #
    # print("        Train dataset:{} images".format(len(train_data)))
    # print("        Test dataset:{} images".format(len(test_data)))

    # step2 -------------------------------------------

    # print("Step 3. Generate data & names files under "+cfgFolder+ " folder, and update YOLO config file.")

    classes = len(parameters.classList)

    if not os.path.exists(os.path.join(parameters.cfgFolder ,"weights")):
        os.makedirs(os.path.join(parameters.cfgFolder,"weights"))
        print("all weights will generated in here: " + parameters.cfgFolder + folderCharacter + "weights" + folderCharacter)

    with open(os.path.join(parameters.cfgFolder ,parameters.cfg_obj_data), 'w') as the_file:
        the_file.write("classes= " + str(classes) + "\n")
        the_file.write("train  = " +os.path.join( parameters.cfgFolder , "train.txt\n"))
        the_file.write("valid  = " + os.path.join(parameters.cfgFolder,"test.txt\n"))
        the_file.write("names = " + os.path.join(parameters.cfgFolder ,"obj.names\n"))
        the_file.write("backup = " +os.path.join( parameters.cfgFolder , "weights/"))



    with open(os.path.join(parameters.cfgFolder , parameters.cfg_obj_names), 'w') as the_file:
        classList_sorted = {name: i for i, name in parameters.classList.items()}
        for class_id in range(classes):
            the_file.write(classList_sorted[class_id] + "\n")



    # step4 ----------------------------------------------------

    print("Step 4. Start to Test the YOLO model.")

    # if not os.path.exists("darknet53.conv.74"):
    #     downloadPretrained("https://pjreddie.com/media/files/darknet53.conv.74")

    classNum = len(parameters.classList)
    filterNum = (classNum + 5) * 3
    print("--Step 5. Start to Test the YOLO model classNum :{}".format(classNum))
    # if(parameters.modelYOLO == "yolov3"):
    #     fileCFG = "yolov3.cfg"

    # else:
    #     fileCFG = "yolov3-tiny.cfg"
    fileCFG = "yolov3.cfg"

    with open(os.path.join(parameters.cfgFolder,fileCFG)) as file:
        file_content = file.read()



    file_updated = file_content.replace("{BATCH}", str(parameters.numBatch_test))
    file_updated = file_updated.replace("{SUBDIVISIONS}", str(parameters.numSubdivision_test))
    file_updated = file_updated.replace("{FILTERS}", str(filterNum))
    file_updated = file_updated.replace("{CLASSES}", str(classNum))
    file_updated = file_updated.replace("{max_batches}", parameters.max_batches)
    file_updated = file_updated.replace("{steps}", parameters.steps)
    file_updated = file_updated.replace("{ANCHORS}", parameters.anchors)

    file = open(os.path.join(parameters.cfgFolder,fileCFG), "w")
    file.write(file_updated)
    file.close()
    ###################################### yolov3_cfg/yolov3.cfg change ok __end
    # gt_txt = 'gt.txt'
    # gt_file = open(gt_txt, 'w')
    # gt_file.close()
    # gt_file = open(gt_txt, 'a')
    #
    # fp = open(outputTestFile, 'r')
    # lines = fp.readlines()
    # # read test.txt filename mapping 2 gt.txt with corresponding answer together ==> gt.txt
    # for i in range(len(lines)):
    #     gt_file.write(lines[i])
    #     image_type = lines[i].split('/')[-1].split('.')[-1]
    #     txt = lines[i].replace('images', 'labels').replace(image_type, 'txt')
    #     # print ("\n---txt : {} : {} ---\n ".format(txt,lines[i] ))
    #     fp2 = open(txt, 'r')
    #     lines_2 = fp2.readlines()
    #     if lines_2 != []:
    #         [h, w, ch] = cv2.imread(lines[i].split('\n')[0]).shape
    #         # print(lines[i].split('\n')[0])
    #         for j in range(len(lines_2)):
    #             if lines_2[j][0] != " ":
    #                 class_id = lines_2[j].split(' ')[0]
    #                 x1 = round(float(lines_2[j].split(' ')[1]) * w - float(lines_2[j].split(' ')[3]) / 2 * w)
    #                 y1 = round(float(lines_2[j].split(' ')[2]) * h - float(lines_2[j].split(' ')[4]) / 2 * h)
    #                 x2 = round(float(lines_2[j].split(' ')[1]) * w + float(lines_2[j].split(' ')[3]) / 2 * w)
    #                 y2 = round(float(lines_2[j].split(' ')[2]) * h + float(lines_2[j].split(' ')[4]) / 2 * h)
    #             else:
    #                 class_id = lines_2[j].split(' ')[1]
    #                 x1 = round(float(lines_2[j].split(' ')[2]) * w - float(lines_2[j].split(' ')[4]) / 2 * w)
    #                 y1 = round(float(lines_2[j].split(' ')[3]) * h - float(lines_2[j].split(' ')[5]) / 2 * h)
    #                 x2 = round(float(lines_2[j].split(' ')[2]) * w + float(lines_2[j].split(' ')[4]) / 2 * w)
    #                 y2 = round(float(lines_2[j].split(' ')[3]) * h + float(lines_2[j].split(' ')[5]) / 2 * h)
    #             # print("{} {} {} {}\n".format(lines_2[j], type(y1), type(x2), type(y2)))
    #             gt_file.write(classList_sorted[int(class_id)] + ' ' \
    #                           + str(x1) + ' ' + str(y1) + ' ' \
    #                           + str(x2) + ' ' + str(y2) + '\n')
    # gt_file.close()
    # # ======================weight_path = /home/gilbert3/train_verify_yolo/yolov3_cfg/weights/
    # predict_path = 'predict'
    # if not os.path.exists(predict_path):
    #     os.makedirs(predict_path)
    #
    # fp = open(parameters.cfgFolder + folderCharacter + parameters.cfg_obj_data, 'r')
    # lines = fp.readlines()
    # for i in range(len(lines)):
    #     if lines[i].split(' = ')[0] == 'backup':  weight_path = lines[i].split(' = ')[1].split('\n')[0]
    # # read yolov3_cfg/obj.data barckup == "/home/gilbert3/trainYolo/yolov3_cfg/weights/"
    # # weight_path = barckup = "/home/gilbert3/trainYolo/yolov3_cfg/weights/"
    # print ("\n---weight_path : {} : {} ---\n ".format(weight_path,lines ))
    # weight_file = {}
    # for filename in os.listdir(weight_path):
    #     if filename.split('.')[-1] == 'weights':
    #         if filename.split('_')[-1].split('.')[0] != 'final':
    #             weight_file[int(filename.split('_')[-1].split('.')[0])] = filename
    #         else:
    #             weight_file[int(parameters.max_batches)] = filename
    # index = sorted(weight_file.keys())
    # #weight_file = {56000: 'yolov3_final.weights'}
    # print("----test->main():: {}\n : {}\n  : {}\n----".format(weight_path, index, weight_file))


#================================================
    # executeCmd = darknetEcec + " detector train " + cfgFolder + folderCharacter + "obj.data " + cfgFolder + folderCharacter + fileCFG + " darknet53.conv.74"
    #
    # print("        execute darknet training command:")
    # print("          " + executeCmd)
    # print("")
    # print("        you can find all the weights files here:" + cfgFolder + folderCharacter + "weights" + folderCharacter)
    #
    # time.sleep(3)
    # call(executeCmd.split())

    #================================================
    # # read test.txt filename mapping 2 gt.txt with corresponding answer together ==> gt.txt
    gt_txt = parameters.gt_txt
    gt_file = open(gt_txt, 'w')
    gt_file.close()
    gt_file = open(gt_txt, 'a')

    gt_path = os.path.join(parameters.outputPath ,  "gtDir/")
    if not os.path.exists(gt_path):
        os.makedirs(gt_path)
    # outputTestFile = parameters.cfgFolder + "/test.txt"
    fp = open(outputTestFile, 'r')
    lines = fp.readlines()
    # read test.txt filename mapping 2 gt.txt with corresponding answer("./labels/*.txt")together ==> gt.txt
    for i in range(len(lines)):
        gt_file.write(lines[i])
        # image_type = lines[i].split('/')[-1].split('.')[-1]
        # txt = lines[i].replace('images', parameters.testPathLabel.split("/")[-1]).replace(image_type, 'txt')
        txt = lines[i].split('.')[0] + '.txt'
        # cp test.txt all *.jpg correspond answer ("labels_xxx/*.txt") to gt.txt
        txt = txt.replace('images', parameters.testPathLabel.split("/")[-1])
        # print ("\n----{} :txt :{}:\n{}:-------\n".format(i,lines[i].split('.')[0],txt))
        fp2 = open(txt, 'r')
        lines_2 = fp2.readlines()

        # Predict_txt = parameters.predict_txt + "/" + lines[i].split("/")[-1].strip().replace("jpg", 'txt')
        # #read corresponding predict GH012582_12400.txt
        # fp_predict = open(Predict_txt,"r")
        # lines_predict = fp_predict.readlines()
        #
        # cap = cv2.VideoCapture(lines[i].strip())
        # # ret, img = cap.read()
        # # GH012582_12400.jpg == lines[i].split('/')[-1]
        # image_type = lines[i].split('/')[-1].split('.')[-1]
        # imgFileName = lines[i].split('/')[-1].strip()

        if lines_2 != []:
            [h, w, ch] = cv2.imread(lines[i].split('\n')[0]).shape
            # ret, img = cap.read()
            for j in range(len(lines_2)):
                if lines_2[j][0] != " ":
                    class_id = lines_2[j].split(' ')[0]
                    x1 = round(float(lines_2[j].split(' ')[1]) * w - float(lines_2[j].split(' ')[3]) / 2 * w)
                    y1 = round(float(lines_2[j].split(' ')[2]) * h - float(lines_2[j].split(' ')[4]) / 2 * h)
                    x2 = round(float(lines_2[j].split(' ')[1]) * w + float(lines_2[j].split(' ')[3]) / 2 * w)
                    y2 = round(float(lines_2[j].split(' ')[2]) * h + float(lines_2[j].split(' ')[4]) / 2 * h)
                else:
                    class_id = lines_2[j].split(' ')[1]
                    x1 = round(float(lines_2[j].split(' ')[2]) * w - float(lines_2[j].split(' ')[4]) / 2 * w)
                    y1 = round(float(lines_2[j].split(' ')[3]) * h - float(lines_2[j].split(' ')[5]) / 2 * h)
                    x2 = round(float(lines_2[j].split(' ')[2]) * w + float(lines_2[j].split(' ')[4]) / 2 * w)
                    y2 = round(float(lines_2[j].split(' ')[3]) * h + float(lines_2[j].split(' ')[5]) / 2 * h)
                gt_file.write(classList_sorted[int(class_id)] + ' ' \
                              + str(x1) + ' ' + str(y1) + ' ' \
                              + str(x2) + ' ' + str(y2) + '\n')

        #         pt1 = (x1, y1)
        #         pt2 = (x2, y2)
        #         cv2.rectangle(img, pt1, pt2, (0, 255, 0), 2)
        #         cv2.putText(img, classList_sorted[int(class_id)], (pt1[0]-10, pt1[1] - 5),
        #                     cv2.FONT_HERSHEY_SIMPLEX, 1, [0, 255, 0], 2)
        #
        #     for k in range(len(lines_predict)):
        #         kk = lines_predict[k].split()
        #         pt1 = (int(kk[1]), int(kk[2]))
        #         pt2 = (int(kk[3]), int(kk[4]))
        #         cv2.rectangle(img, pt1, pt2, (255, 0, 0), 1)
        #         cv2.putText(img, kk[0], (pt2[0], pt1[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, [255, 0, 0], 2)
        #
        # cv2.imshow(imgFileName, img)
        # cv2.imwrite(gt_path+imgFileName,img)
        #
        # if cv2.waitKey(0) & 0xFF == ord('q'):
        #     break
    gt_file.close()
    # ================================================weight_path = /home/gilbert3/train_verify_yolo/yolov3_cfg/weights/
    predict_path = parameters.predict_txt #'predict'
    if not os.path.exists(predict_path):
        os.makedirs(predict_path)
    # read yolov3_cfg/obj.data barckup == "/home/gilbert3/trainYolo/yolov3_cfg/weights/"
    # weight_path = barckup = "/home/gilbert3/trainYolo/yolov3_cfg/weights/"
    fp = open(parameters.cfgFolder + folderCharacter + parameters.cfg_obj_data, 'r')
    lines = fp.readlines()
    for i in range(len(lines)):
        if lines[i].split(' = ')[0] == 'backup':  weight_path = lines[i].split(' = ')[1].split('\n')[0]

    weight_file = {}
    try:
        if parameters.dest_weight:
            weight_file[int(parameters.max_batches)] = parameters.dest_weight
            predict_txt = os.path.join(parameters.savePridictPath ,parameters.dest_weight.split('/')[-1].split('.')[0] + '.txt')
            print("\n\n---------123----------:{}:\n\n".format(predict_txt))

            executeCmd = "{} detector test_v2 {} {} {} {} {} -out {} -thresh {}".format(
                     parameters.darknetEcec,
                     (os.path.join(parameters.cfgFolder ,"obj.data")),
                     (os.path.join(parameters.cfgFolder , 'yolov3.cfg')),
                     parameters.dest_weight,
                     outputTestFile,
                     predict_txt,
                     (parameters.savePridictPath + folderCharacter),
                     parameters.threshValue
                     )

    except:

        # weight_path = barckup = "/home/gilbert3/trainYolo/yolov3_cfg/weights/"
        for filename in os.listdir(weight_path):
            if filename.split('.')[-1] == 'weights':
                if filename.split('_')[-1].split('.')[0] != 'final':
                    weight_file[int(filename.split('_')[-1].split('.')[0])] = filename
                else:
                    weight_file[int(parameters.max_batches)] = filename
        #weight_file = {"56000","yolov3_final.weights"}
        index = sorted(weight_file.keys())
        # predict_txt = "yolov3_final.txt"
        predict_txt = os.path.join(parameters.savePridictPath , weight_file[int(parameters.max_batches)].split('.')[0] + '.txt')

        executeCmd = "{} detector test_v2 {} {} {} {} {} -out {} -thresh {}".format(
                 parameters.darknetEcec,
                 (os.path.join(parameters.cfgFolder ,"obj.data")),
                 (os.path.join(parameters.cfgFolder , 'yolov3.cfg')),
                 weight_path + weight_file[int(parameters.max_batches)],
                 outputTestFile,
                 predict_txt,
                 (parameters.savePridictPath + folderCharacter),
                 parameters.threshValue
                 )
        print("\n\n---------456--------{}\n:{}:\n{}:\n{}:\n".format(weight_path,executeCmd, index,predict_txt))
    # for i in index:
    #     predict_txt = savePridictPath + folderCharacter + weight_file[i].split('.')[0] + '.txt'
    #     executeCmd = darknetEcec + " detector test_v2 " + cfgFolder + folderCharacter + "obj.data " \
    #                                                     + cfgFolder + folderCharacter + fileCFG + ' ' \
    #                                                     + weight_path + weight_file[i] + ' ' \
    #                                                     + outputTestFile + ' ' \
    #                                                     + predict_txt
    #
    #
    #     print("        execute darknet testing command:")
    #     print("          " + executeCmd)
    #     print("")
    #     print("        you can find all the weights files here:" + cfgFolder + folderCharacter + "weights" + folderCharacter)
    #
    #     time.sleep(3)
    #     call(executeCmd.split())


    # predict_txt = parameters.savePridictPath + folderCharacter + weight_file[int(parameters.max_batches)].split('.')[0] + '.txt'
    #
    # fp = open(parameters.savePridictPath + folderCharacter +"yolov3_final.txt", 'r')
    # lines = fp.readlines()
    # tmp = ""
    # for i in range(len(lines)):
    #
    #             if lines[i][0] == '/':
    #                 print(i,lines[i].split("/")[-1].split(".")[0]+".txt")
    #                 tmp = lines[i].split("/")[-1].split(".")[0]+".txt"
    #             else:
    #                 print(lines[i].strip())
    #                 with open(os.getcwd()+"/predict/"+tmp,"a") as fd:
    #                         fd.write(lines[i].strip() + "\n")

    print("        execute darknet testing command:")
    print("          " + executeCmd)
    print("")
    print("        you can find all the weights files here:" + parameters.cfgFolder + folderCharacter + "weights" + folderCharacter)

    time.sleep(3)
    call(executeCmd.split())
    #
    time.sleep(3)
    performance.perform()

    time.sleep(3)
    CFG_2_outputfolder()
