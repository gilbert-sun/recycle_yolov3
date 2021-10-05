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


if __name__ == '__main__':
    # step1 ---------------------------------------------------------------


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


    classNum = len(parameters.classList)
    filterNum = (classNum + 5) * 3

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


    executeCmd = "{} detector train {} {} yolov3.conv.75 -gpus {}".format(parameters.darknetEcec, (parameters.cfgFolder + folderCharacter + "obj.data"), (parameters.cfgFolder + folderCharacter + 'yolov3.cfg'), parameters.num_gpu)
    #executeCmd = "{} detector train {} {} {} -gpus {}".format(parameters.darknetEcec, (parameters.cfgFolder + folderCharacter + "obj.data"), (parameters.cfgFolder + folderCharacter + 'yolov3.cfg'), (parameters.cfgFolder + folderCharacter + 'weights/yolov3.backup'),parameters.num_gpu)
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
    performance.main()
