import os
darknetEcec = "/home/e200/darknet/darknet" # or /home/shyechih/yolo_darknet/darknet
imageYoloPath = "/media/e200/DATA/mnt/10002/images"#"/media/e200/DATA/20190719_gilbert_full/final_test/total/images"
labelYoloPath = "/media/e200/DATA/mnt/10002/labels"#"/media/e200/DATA/20190719_gilbert_full/final_test/total/labels"
testPathImage = "/media/e200/新增磁碟區1/mnt/1000/images"
testPathLabel = "/media/e200/新增磁碟區1/mnt/1000/labels"
#"/media/e200/DATA/v3_7anchor_count_2021_0815/Oil/images"
#"/media/e200/PlextorSSD0/v3_7anchor_count_2021_0815/images"
#outputPath = "/media/e200/新增磁碟區1/mnt/v4_7anchor_resol/"
outputPath = "/media/e200/新增磁碟區1/mnt2/v3_7anchor_count_2021_1001"
#outputPath = "/media/e200/新增磁碟區1/mnt/7_15w_noAnchorl"
savePridictPath = outputPath + "/predict"
predict_txt = outputPath + '/predict'
classList = {"P":0,"O":1, "S": 2, "C":3, "Ot":4, "Ch":5, "T":6}

# modelYOLO = "yolov2_tiny"  #yolov3 or yolov3-tiny
testRatio = 0.1
threshValue = 0.7

#-------------- yolov3 ---------------
cfgFolder = "/media/e200/新增磁碟區1/Documents/makeYOLOv3_pet" + "/yolov3_cfg"

cfg_obj_names = "obj.names"
cfg_obj_data = "obj.data"

numBatch_train = 64
numSubdivision_train = 16
numBatch_test = 1
numSubdivision_test = 1
max_batches = "56020"
steps = "49000, 52000"
anchors = "10,13,  16,30,  33,23,  30,61,  62,45,  59,119,  116,90,  156,198,  373,326"
#"70,123, 117, 98, 65,213, 94,164, 132,139, 99,231, 130,191, 132,242, 170,284"
num_gpu = '0'
# num_gpu = '0,1'

output_txt = "/test.txt"
gt_txt = "/media/e200/新增磁碟區1/Documents/makeYOLOv3_pet" + "gt.txt"
wrong_txt = outputPath +"/wrong.txt"
accuracy_path = outputPath+ "/accuracy"
acc_txt = 'acc.txt'


# #---->video2img
# video2img_sour = "class_soy.avi"
# out_frameID = 550
# out_frame = True
# video2img_out_imgPath = "/media/e200/DATA/mnt/Soy/images"
# video2img_out_labelPath = "/media/e200/DATA/mnt/Soy/labels"
# video2img_out_predictPath = "/media/e200/新增磁碟區1/mnt2/soy_sour_predict/predict"
# video2img_out_imgKind = os.path.join(video2img_out_imgPath , "soy_")
# video2img_List = "iii.txt"

# #---->parser_Prediction
# classType = "2 "

# #---->rotate_dataAugment---->
# dataset_input = "/media/e200/DATA/mnt/Soy"
# dataset_output = "/media/e200/DATA/mnt/rotSoy20"
# angle_interval = 20
# show_image = False


#===================================================================================
#---->video2img
video2img_sour = "class_soy.avi"
out_frameID = 50
out_frame = True
video2img_out_imgPath = "/media/e200/DATA/mnt/Soy/images"
video2img_out_labelPath = "/media/e200/DATA/mnt/Soy/labels"
video2img_out_predictPath = "/media/e200/新增磁碟區1/mnt2/soy_sour_predict/predict"
video2img_out_imgKind = os.path.join(video2img_out_imgPath , "soy_")
video2img_List = "soy.txt"

#---->parser_Prediction
classType = "2 "

#===================================================================================
#---->rotate_dataAugment---->
dataset_input = "/media/e200/DATA/mnt/Flat"
dataset_output = "/media/e200/DATA/mnt/rotFlat20"
angle_interval = 20
show_image = False
