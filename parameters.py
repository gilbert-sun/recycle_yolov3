import os
darknetEcec = "/home/gilbert3/darknet/darknet"
img_src = "/home/gilbert3/Documents/mnt/1000/"
imageYoloPath = img_src + "images"
labelYoloPath = img_src + "labels"
test_src = "/home/gilbert3/Documents/mnt/1000/"
testPathImage = test_src + "images"
testPathLabel = test_src + "labels"
outputPath = "/home/gilbert3/Documents/mnt/1000/"
savePridictPath = outputPath + "predict"
predict_txt = outputPath + 'predict'
wrong_txt = outputPath  + "wrong.txt"
accuracy_path = outputPath + "accuracy"
acc_txt = 'acc.txt'

# modelYOLO = "yolov2_tiny"  #yolov3 or yolov3-tiny
classList = {"P":0,"O":1, "S": 2, "C":3, "Ot":4, "Ch":5, "T":6}
testRatio = 0.1
threshValue = 0.1

#-------------- yolov3 ---------------
code_src = "/home/gilbert3/recycle_yolo/"
cfgFolder = code_src + "yolov3_cfg"
gt_txt = code_src + "gt.txt"
cfg_obj_names = "obj.names"
cfg_obj_data = "obj.data"

numBatch_train = 64
numSubdivision_train = 16
numBatch_test = 1
numSubdivision_test = 1
max_batches = "56000"
steps = "35000, 46000"
anchors = "70,123, 117, 98, 65,213, 94,164, 132,139, 99,231, 130,191, 132,242, 170,284"
num_gpu = '0'
# num_gpu = '0,1'


