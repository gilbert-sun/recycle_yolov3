import parameters
import test
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
# from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

#--------------------------------------------------------------------
folderCharacter = "/"  # \\ is for windows
# xmlFolder = "/media/e200/BackupUbuntu/Documents/makeYOLOv3/cloth5/labels"
# labelFolder = "/media/e200/BackupUbuntu/Documents/makeYOLOv3/cloth5/labels"
# imgFolder = "/media/e200/BackupUbuntu/Documents/makeYOLOv3/cloth5/images"
# saveYoloPath = "/media/e200/BackupUbuntu/Documents/makeYOLOv3/cloth5/yolov4"
# classList = { "0_Deep_ST":0, "1_Light_ST": 1 , "2_Deep_LT":2, "3_Light_LT": 3 ,"4_Deep_SU":4, "5_Light_SU": 5 ,"6_Deep_LU":6, "7_Light_LU": 7  }

# labelFolder = "/media/e200/新增磁碟區/scene/labels"
# imgFolder = "/media/e200/新增磁碟區/scene/images"

# classList = {"oven": 0, "sink": 1, "refrigerator": 2, "sofa": 3,"TV_monitor": 4, "chair": 5,"bed": 6,
#              "counter_top": 7, "range_hood": 8, "gas_stove": 9, "electric_pot": 10, "electric_kettle": 11,
#              "coffee_table": 12, "dvd_player": 13, "TV_remote": 14, "vanity": 15, "pillow":16, "quilt":17,
#              "closet": 18, "nightstand": 19}


# Deep_ST
# Light_ST
# Deep_LT
# Light_LT
# Deep_SU
# Light_SU
# Deep_LU
# Light_LU

# #---------------------------------------------------------------------
#
# if not os.path.exists(saveYoloPath):
#     os.makedirs(saveYoloPath)

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


if __name__ == '__main__':
    # step1 ---------------------------------------------------------------

    # if not os.path.exists(saveYoloPath):
        # os.makedirs(saveYoloPath)
        # print(" saveYolo+ img + label ")
    # for file in os.listdir(imgFolder):
    #     imgfile = imgFolder + folderCharacter + file
    #     #print (file, ":",imgfile )
    #     copyfile(imgfile, saveYoloPath + folderCharacter + file )
    # for file in os.listdir(labelFolder):
    #     labelfile = labelFolder + folderCharacter + file
    #     copyfile(labelfile, saveYoloPath + folderCharacter + file)

    # step2 ---------------------------------------------------------------
    fileList = []
    outputTrainFile = parameters.cfgFolder + "/train.txt"
    outputTestFile = parameters.cfgFolder + "/test.txt"
    # use_for_test = parameters.cfgFolder + "/use_for_test.txt"

    print("Step 2. Create YOLO cfg folder and split dataset to train and test datasets.")
    if not os.path.exists(parameters.cfgFolder):
        os.makedirs(parameters.cfgFolder)

    for imgfile in os.listdir(parameters.imageYoloPath):
        filename, file_extension = os.path.splitext(imgfile)
        file_extension = file_extension.lower()

        if(file_extension == ".jpg" or file_extension==".jpeg" or file_extension==".png" or file_extension==".bmp"):
            fileList.append(parameters.imageYoloPath + folderCharacter + imgfile)
    
    testCount = int(len(fileList) * parameters.testRatio)
    trainCount = len(fileList) - testCount

    a = range(len(fileList))
    all_list = list(a) 
    test_data = random.sample(a, testCount)
    for i in range(len(test_data)):
        all_list.remove(test_data[i])
    train_data = all_list   
    

    f = open(outputTrainFile, 'w')
    f.close()
    with open(outputTrainFile, 'a') as the_file:
        for i in train_data:
            the_file.write(fileList[i] + "\n")
    the_file.close()



    f = open(outputTestFile, 'w')
    f.close()
    with open(outputTestFile, 'a') as the_file:
        for i in test_data:
            the_file.write(fileList[i] + "\n")
    the_file.close()

    print("        Train dataset:{} images".format(len(train_data)))
    print("        Test dataset:{} images".format(len(test_data)))

    # step2 -------------------------------------------

    print("Step 3. Generate data & names files under "+parameters.cfgFolder+ " folder, and update YOLO config file.")

    classes = len(parameters.classList)

    if not os.path.exists(parameters.cfgFolder + folderCharacter + "weights"):
        os.makedirs(parameters.cfgFolder + folderCharacter + "weights")
        print("all weights will generated in here: " + parameters.cfgFolder + folderCharacter + "weights" + folderCharacter)

    with open(parameters.cfgFolder + folderCharacter + parameters.cfg_obj_data, 'w') as the_file:
        the_file.write("classes= " + str(classes) + "\n")
        the_file.write("train  = " + parameters.cfgFolder + folderCharacter + "train.txt\n")
        the_file.write("valid  = " + parameters.cfgFolder + folderCharacter + "test.txt\n")
        the_file.write("names = " + parameters.cfgFolder + folderCharacter + "obj.names\n")
        the_file.write("backup = " + parameters.cfgFolder + folderCharacter + "weights/")



    with open(parameters.cfgFolder + folderCharacter + parameters.cfg_obj_names, 'w') as the_file:
        classList_sorted = {name: i for i, name in parameters.classList.items()}
        for class_id in range(classes):
            the_file.write(classList_sorted[class_id] + "\n")



    # step4 ----------------------------------------------------

    print("Step 4. Start to train the YOLO model.")

    # if not os.path.exists("darknet53.conv.74"):
    #     downloadPretrained("https://pjreddie.com/media/files/darknet53.conv.74")

    # classNum = len(parameters.classList)
    # filterNum = (classNum + 5) * 3

    # if(parameters.modelYOLO == "yolov3"):
    #     fileCFG = "yolov3.cfg"

    # else:
    #     fileCFG = "yolov3-tiny.cfg"

    # with open("cfg"+folderCharacter+fileCFG) as file:
    #     file_content = file.read()

    classNum = len(parameters.classList)
    filterNum = (classNum + 5) * 3
    # if "yolov2" in parameters.cfgFolder and "tiny" in parameters.cfgFolder: 
    #     fileCFG = "yolov2-tiny.cfg"
    # elif "yolov2" in parameters.cfgFolder:
    #     fileCFG = "yolov2.cfg"
    fileCFG = "yolov3.cfg"
    

    with open("cfg"+folderCharacter+fileCFG) as file:
        file_content = file.read()



    file_updated = file_content.replace("{BATCH}", str(parameters.numBatch_train))
    file_updated = file_updated.replace("{SUBDIVISIONS}", str(parameters.numSubdivision_train))
    file_updated = file_updated.replace("{FILTERS}", str(filterNum))
    file_updated = file_updated.replace("{CLASSES}", str(classNum))
    file_updated = file_updated.replace("{max_batches}", parameters.max_batches)
    file_updated = file_updated.replace("{steps}", parameters.steps)
    file_updated = file_updated.replace("{ANCHORS}", parameters.anchors)

    file = open(parameters.cfgFolder+folderCharacter+fileCFG, "w")
    file.write(file_updated)
    file.close()



#================================================
    
    # if "yolov2" in parameters.cfgFolder and "tiny" in parameters.cfgFolder:
    #     executeCmd = parameters.darknetEcec + " detector train " + parameters.cfgFolder + folderCharacter + "obj.data " + parameters.cfgFolder + folderCharacter + 'yolov2-tiny.cfg' + " yolov2-tiny.conv.13"
    # elif "yolov2" in parameters.cfgFolder:
    #     executeCmd = parameters.darknetEcec + " detector train " + parameters.cfgFolder + folderCharacter + "obj.data " + parameters.cfgFolder + folderCharacter + 'yolov2.cfg' + " yolov2.conv.23"
    executeCmd = "{} detector train {} {} yolov3.conv.75 -gpus {}".format(parameters.darknetEcec, (parameters.cfgFolder + folderCharacter + "obj.data"), (parameters.cfgFolder + folderCharacter + 'yolov3.cfg'), parameters.num_gpu)
    #executeCmd = parameters.darknetEcec + " detector train " + parameters.cfgFolder + folderCharacter + "obj.data " + parameters.cfgFolder + folderCharacter + fileCFG + " /media/e200/新增磁碟區/Documents/makeYOLOv3_fall/fall_cfg/weights/yolov3.backup"

#
    print("        execute darknet training command:")
    print("          " + executeCmd)
    print("")
    print("        you can find all the weights files here:" + parameters.cfgFolder + folderCharacter + "weights" + folderCharacter)

    time.sleep(3)
    call(executeCmd.split())

    executeCmd = test.main()
    time.sleep(3)
    call(executeCmd.split())

    time.sleep(3)
    # performance.main()
    performance.perform()