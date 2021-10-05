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
        (parameters.cfgFolder + folderCharacter + "obj.data"),
        (parameters.cfgFolder + folderCharacter + "obj.names"),
        (parameters.cfgFolder + folderCharacter + 'yolov3.cfg'),
        outputPath + "/yolov3_cfg"
    )
    executeCmd2 = "cp -r {}  {}/ ".format(
        (parameters.cfgFolder + folderCharacter + "weights/yolov3_final.weights"),
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



def main():
    fileList = []
    outputTrainFile = parameters.cfgFolder + "/train.txt"
    outputTestFile = parameters.cfgFolder + "/test.txt"
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

        with open("cfg" + folderCharacter + fileCFG) as file:
            file_content = file.read()

        file_updated = file_content.replace("{BATCH}", str(parameters.numBatch_test))
        file_updated = file_updated.replace("{SUBDIVISIONS}", str(parameters.numSubdivision_test))
        file_updated = file_updated.replace("{FILTERS}", str(filterNum))
        file_updated = file_updated.replace("{CLASSES}", str(classNum))
        file_updated = file_updated.replace("{max_batches}", parameters.max_batches)
        file_updated = file_updated.replace("{steps}", parameters.steps)
        file_updated = file_updated.replace("{ANCHORS}", parameters.anchors)

        file = open(parameters.cfgFolder + folderCharacter + fileCFG, "w")
        file.write(file_updated)
        file.close()

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

        fp = open(parameters.cfgFolder + folderCharacter + parameters.cfg_obj_data, 'r')
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
        predict_txt = parameters.savePridictPath + folderCharacter + weight_file[int(parameters.max_batches)].split('.')[0] + '.txt'

        # print("\n\n---------000 entry test main----------:{}:\n\n".format(predict_txt))
        executeCmd = "{} detector test_v2 {} {} {} {} {} -out {} -thresh {}".format(
                 parameters.darknetEcec,
                 (parameters.cfgFolder + folderCharacter + "obj.data"),
                 (parameters.cfgFolder + folderCharacter + 'yolov3.cfg'),
                 weight_path + weight_file[int(parameters.max_batches)],
                 outputTestFile,
                 predict_txt,
                 (parameters.savePridictPath + "/"),
                 parameters.threshValue
                 )

        print("----------------------------------------------------------execute darknet testing command:")
        print("          " + executeCmd)
        print("----------------------------------------------------------")
        print("you can find all the weights files here:" + parameters.cfgFolder + folderCharacter + "weights" + folderCharacter)

        time.sleep(3)
        CFG_2_outputfolder()
        return executeCmd





if __name__ == '__main__':
    # step1 ---------------------------------------------------------------


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
    outputTestFile = parameters.cfgFolder + parameters.output_txt #"/test.txt"



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

    with open("cfg"+folderCharacter+fileCFG) as file:
        file_content = file.read()



    file_updated = file_content.replace("{BATCH}", str(parameters.numBatch_test))
    file_updated = file_updated.replace("{SUBDIVISIONS}", str(parameters.numSubdivision_test))
    file_updated = file_updated.replace("{FILTERS}", str(filterNum))
    file_updated = file_updated.replace("{CLASSES}", str(classNum))
    file_updated = file_updated.replace("{max_batches}", parameters.max_batches)
    file_updated = file_updated.replace("{steps}", parameters.steps)
    file_updated = file_updated.replace("{ANCHORS}", parameters.anchors)

    file = open(parameters.cfgFolder+folderCharacter+fileCFG, "w")
    file.write(file_updated)
    file.close()

    #================================================
    # # read test.txt filename mapping 2 gt.txt with corresponding answer together ==> gt.txt
    gt_txt = parameters.gt_txt
    gt_file = open(gt_txt, 'w')
    gt_file.close()
    gt_file = open(gt_txt, 'a')

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
        if lines_2 != []:
            [h, w, ch] = cv2.imread(lines[i].split('\n')[0]).shape
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
            predict_txt = parameters.savePridictPath + folderCharacter + parameters.dest_weight.split('/')[-1].split('.')[0] + '.txt'
            print("\n\n---------123----------:{}:\n\n".format(predict_txt))

            executeCmd = "{} detector test_v2 {} {} {} {} {} -out {} -thresh {}".format(
                     parameters.darknetEcec,
                     (parameters.cfgFolder + folderCharacter + "obj.data"),
                     (parameters.cfgFolder + folderCharacter + 'yolov3.cfg'),
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
        predict_txt = parameters.savePridictPath + folderCharacter + weight_file[int(parameters.max_batches)].split('.')[0] + '.txt'

        executeCmd = "{} detector test_v2 {} {} {} {} {} -out {} -thresh {}".format(
                 parameters.darknetEcec,
                 (parameters.cfgFolder + folderCharacter + "obj.data"),
                 (parameters.cfgFolder + folderCharacter + 'yolov3.cfg'),
                 weight_path + weight_file[int(parameters.max_batches)],
                 outputTestFile,
                 predict_txt,
                 (parameters.savePridictPath + folderCharacter),
                 parameters.threshValue
                 )
        print("\n\n---------456--------{}\n:{}:\n{}:\n{}:\n".format(weight_path,executeCmd, index,predict_txt))


    print("        execute darknet testing command:")
    print("          " + executeCmd)
    print("")
    print("        you can find all the weights files here:" + parameters.cfgFolder + folderCharacter + "weights" + folderCharacter)

    time.sleep(3)
    call(executeCmd.split())

    time.sleep(3)
    performance.main()

    time.sleep(3)
    CFG_2_outputfolder()
