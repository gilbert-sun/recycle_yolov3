import parameters
import numpy as np
import os,cv2
import os
# import django
# SECRET_KEY = 'i^u9f(sbrh$%wb*c34kcpth3l2g(xw3g6+@lhk^&6_gz(uusza'
# # os.environ.setdefault('DJANGO_SETTING_MODULE', 'f1_score')
# # django.setup()

# from f1_score.mysite.models import  PetAccData
ShOW_DIFF_IMG = False
ShOW_GT_Predict_IMG = False
Pred_generate_txt = True

import glob

def count_folder(src1):
    l_all_count = []
    try:
        file_count = len(os.listdir(src1))
        for fn in glob.glob(src1+"/*"):
            with open(fn , "r") as fp:
                l_all_count.extend(list(map(lambda count: count[0], fp.readlines())))
        l_div_countPet = list(map(lambda i: l_all_count.count( str(i) ) , range(0,7)))
        # l_div_countPet = dict(map(lambda i:( str(i), l_all_count.count( str(i))  ), range(0,7)))
    except:
        print("\nErr \n---no such file or dictionary: ==> {} ---\n".format(src1))
        exit(0)
    return l_div_countPet,file_count


def woCorrespond(pred_box,j):
    # global cl, x1, y1, x2, y2, pct, pt1, pt2
    cl = pred_box[j].split(" ")[0]
    x1 = pred_box[j].split(" ")[1]
    y1 = pred_box[j].split(" ")[2]
    x2 = pred_box[j].split(" ")[3]
    y2 = pred_box[j].split(" ")[4]
    pct = str(int(float(pred_box[j].split(" ")[5]) * 100))
    pt1 = (int(x1), int(y1))
    pt2 = (int(x2), int(y2))

    return cl,pct, pt1,pt2


def predictWrong(wrong_temp,ans_temp,l):
    # global cl, x1p, y1p, x2p, y2p, pct, pt1, pt2, clg, x1g, y1g, x2g, y2g, pt3, pt4
    cl = wrong_temp[l].split(" ")[0]
    x1p = wrong_temp[l].split(" ")[1]
    y1p = wrong_temp[l].split(" ")[2]
    x2p = wrong_temp[l].split(" ")[3]
    y2p = wrong_temp[l].split(" ")[4]
    pct = str(int(float(wrong_temp[l][5]) * 100))
    pt1 = (int(x1p), int(y1p))
    pt2 = (int(x2p), int(y2p))

    clg = ans_temp[l].split(" ")[0]
    x1g = ans_temp[l].split(" ")[1]
    y1g = ans_temp[l].split(" ")[2]
    x2g = ans_temp[l].split(" ")[3]
    y2g = ans_temp[l].split(" ")[4]
    pt3 = (int(x1g), int(y1g))
    pt4 = (int(x2g), int(y2g))

    return cl,clg, pct, pt1,pt2, pt3,pt4


def iou_2small(wrong_temp,ans_temp,l):
    # global cl, pct, pt1, pt2, clg, pt3, pt4
    cl = wrong_temp[l].split(" ")[0]
    x1p = wrong_temp[l].split(" ")[1]
    y1p = wrong_temp[l].split(" ")[2]
    x2p = wrong_temp[l].split(" ")[3]
    y2p = wrong_temp[l].split(" ")[4]
    pct = str(int(float(wrong_temp[l].split(" ")[5]) * 100))
    pt1 = (int(x1p), int(y1p))
    pt2 = (int(x2p), int(y2p))

    clg = ans_temp[l].split(" ")[0]
    x1g = ans_temp[l].split(" ")[1]
    y1g = ans_temp[l].split(" ")[2]
    x2g = ans_temp[l].split(" ")[3]
    y2g = ans_temp[l].split(" ")[4]
    pt3 = (int(x1g), int(y1g))
    pt4 = (int(x2g), int(y2g))

    return cl, clg, pct, pt1, pt2,  pt3, pt4


def woPredict(gt_box_temp,l):
    # global cl, pt1, pt2
    cl = gt_box_temp[l].split(" ")[0]
    x1 = (gt_box_temp[l].split(" ")[1])
    y1 = (gt_box_temp[l].split(" ")[2])
    x2 = (gt_box_temp[l].split(" ")[3])
    y2 = (gt_box_temp[l].split(" ")[4])
    pt1 = (int(x1), int(y1))
    pt2 = (int(x2), int(y2))

    return cl, pt1, pt2


classList_sorted = {name: i for i, name in parameters.classList.items()}

def perform():
    # print GT with Predict img side by side
    if ShOW_GT_Predict_IMG:
        save_GT_Predict()
    wrong = open(parameters.wrong_txt, 'w')
    wrong.close()
    wrong = open(parameters.wrong_txt, 'a')

    if not os.path.exists(parameters.accuracy_path):
        os.makedirs(parameters.accuracy_path)
    try:
        if parameters.dest_weight:
            pred_files = [parameters.dest_weight.split('/')[-1].split('.')[0] + '.txt']

    except:
        pred_files = os.listdir(parameters.predict_txt)

    for file in pred_files:
        if file.split('_')[-1] == 'final.txt':
            gt = open(parameters.gt_txt, 'r')
            lines_gt = gt.readlines()

            pred = open(os.path.join(parameters.predict_txt, file), 'r')
            lines_pred = pred.readlines()

            acc = open(os.path.join(parameters.accuracy_path ,'acc_' + file), 'w')
            acc.close()
            acc = open(os.path.join(parameters.accuracy_path , 'acc_' + file), 'a')

            iou_thresh = 0.5
            # thresh = 0.001
            gt_data = {}
            gt_data_temp = {}
            pred_data = {}
            photo_name = []
            total_img = []
            jmax_indexes = []
            wrong_temp = []


            correct = np.zeros(len(classList_sorted))
            proposals = np.zeros(len(classList_sorted))
            total = np.zeros(len(classList_sorted))

            for i in range(len(lines_gt)):
                if '/' in lines_gt[i]:
                    if i != 0:
                        gt_data[photo_name] = data
                        gt_data_temp[photo_name] = data_temp
                    data = []
                    data_temp = []
                    photo_name = lines_gt[i].strip()
                    #(lines_gt[i].strip() == /home/gilbert3/Documents/mnt/1000/images/GH012601_9480.jpg:,
                    # (photo_name) ==/home/gilbert3/Documents/mnt/1000/images/GH012601_9480.jpg:
                    total_img.append(lines_gt[i].strip())
                else:
                    for j in range(len(classList_sorted)):
                        if classList_sorted[j] == lines_gt[i].split(' ')[0]:
                            total[j] += 1
                            break
                    data.append(lines_gt[i].strip())
                    data_temp.append(lines_gt[i].strip())
                if i == len(lines_gt) - 1:
                    gt_data[photo_name] = data
                    gt_data_temp[photo_name] = data_temp


            for i in range(len(lines_pred)):
                if '/' in lines_pred[i]:
                    if i != 0: pred_data[photo_name] = data
                    photo_name = lines_pred[i].strip()
                    data = []
                else:
                    for j in range(len(classList_sorted)):
                        if classList_sorted[j] == lines_pred[i].split(' ')[0]:
                            proposals[j] += 1
                            break
                    data.append(lines_pred[i])#.strip())
                if i == len(lines_pred) - 1:
                    pred_data[photo_name.strip()] = data
            # print ("-----{}:\n:-----{}:\n:gt_data=={}:\n".format(  len(gt_data),type(gt_data),type(gt_data) ) )
            # print ("-----{}:\n:total_img=={}:\n".format(len(total_img),total_img[:20]) )
            # print ("-----{}:\n:-----{}:\n:pred_data=={}:\n".format( len(pred_data),type(pred_data),pred_data) )
            # print ("\n---11--{}----11---{}:\n:gt_data=={}:\n".format(correct , len(classList_sorted) ,len(gt_data ) ))
            # photo_name.strip() ==/home/gilbert3/Documents/mnt/1000/images/GH012601_9480.jpg
            # print ("\n\n-444--{}-:\n--{}:\n:total_img=={}:\n".format(total_img,type(total_img),len(total_img)))#total_img[:20]) )
            #-444--['/home/gilbert3/Documents/mnt/1000/images/GH012586_2520.jpg', '/home/gilbert3/Documents/mnt/1000/images/GH012603_8040.jpg', '/home/gilbert3/Documents/mnt/1000/images/GH012601_6840.jpg',
            #:total_img==667:
            # print ("\n\n--333--{}: \n-{}:\n:pred_data=={}:\n".format( pred_data,type(pred_data),len(pred_data) ))
            #--333 - -{'/home/gilbert3/Documents/mnt/1000/images/GH012586_2520.jpg': ['T 32 16 693 416 0.986909\n','T 604 76 934 433 0.566341\n','Ot 5
            #-<class 'dict'>:
            #:pred_data==667:
            folderCharacter = "/"
            diff_path = os.path.join(parameters.outputPath , "diffDir/")
            if not os.path.exists(diff_path):
                    os.makedirs(diff_path)

            fss_all = open(os.path.join(parameters.outputPath ,"diffDir/" + "yolov3_final_diff.txt"), "w")
            ##--------------------------start
            count = 0
            for i in total_img:
                gt_box = gt_data[i]
                gt_box_temp = gt_data_temp[i]
                pred_box = pred_data[i]
                ans_temp = []
                ovmax_temp = []

                cap = cv2.VideoCapture(i.strip())
                ret, img = cap.read()
                imgFileName = i.split('/')[-1].strip()
                txtFileName = imgFileName.replace("jpg", "txt")
                fss = open(os.path.join(parameters.outputPath, "diffDir/" + txtFileName), "w")
                fss_all.write(i + "\n")
                correct_internal = 0
                for j in range(len(pred_box)):
                    # if (j < 2):
                    #     print ("---555--\ngt_box:{}\ngt_box_temp:{}\n".format( gt_box ,type(gt_box)))#pred_box[j] , pred_box[j].split(' ') ) )
                    # ret, img = cap.read()
                    if pred_box[j] != '\n':
                        overlaps = []
                        x1_pred = int(pred_box[j].split(' ')[1])
                        y1_pred = int(pred_box[j].split(' ')[2])
                        x2_pred = int(pred_box[j].split(' ')[3])
                        y2_pred = int(pred_box[j].split(' ')[4])
                        if len(gt_box) != 0:
                            for k in range(len(gt_box)):
                                x1_gt = int(gt_box[k].split(' ')[1])
                                y1_gt = int(gt_box[k].split(' ')[2])
                                x2_gt = int(gt_box[k].split(' ')[3])
                                y2_gt = int(gt_box[k].split(' ')[4])

                                inter_x1 = max(x1_gt, x1_pred)
                                inter_y1 = max(y1_gt, y1_pred)
                                inter_x2 = min(x2_gt, x2_pred)
                                inter_y2 = min(y2_gt, y2_pred)
                                inter = (max(inter_x2 - inter_x1, 0)) * (max(inter_y2 - inter_y1, 0))
                                uni = (x2_gt - x1_gt) * (y2_gt - y1_gt) + (x2_pred - x1_pred) * (y2_pred - y1_pred) - inter
                                overlaps.append(float(inter / uni))
                            if overlaps != [0.0] * len(gt_box):
                                ovmax = np.max(np.array(overlaps))
                                jmax = np.argmax(np.array(overlaps))
                            else:
                                ovmax = -1 * float("inf")
                                jmax = float("inf")
                            # gt_box["GH012582_12400.txt"][0] == pred_box["GH012582_12400.txt"][0]
                            if ovmax >= iou_thresh and gt_box[jmax].split(' ')[0] == pred_box[j].split(' ')[0]:
                                for k in range(len(classList_sorted)):
                                    if classList_sorted[k] == gt_box[jmax].split(' ')[0]:
                                        correct_internal += 1
                                        correct[k] += 1
                                        break
                                err = "4:Right "
                                fss.write(err + gt_box[jmax].strip() +" "+ pred_box[j].strip() +"\n")
                                fss_all.write(err + gt_box[jmax].strip() +" "+ pred_box[j].strip() +"\n")
                            else:
                                if ovmax != -1 * float("inf"):
                                    wrong_temp.append(pred_box[j])
                                    ans_temp.append(gt_box[jmax])
                                    ovmax_temp.append(ovmax)
                                else:
                                    # ==> Without correspond
                                    # /media/shyechih/data/stage3_mixed/mix_20percent-1rotated-10percent-10rotated-increase-thin_20200702/images/GH010227_5580_090.jpg
                                    # predict     : Ch 1517 752 1850 1079 0.881586
                                    wrong.write(' ==>Without correspond\n')
                                    wrong.write(i + '\n')
                                    wrong.write('predict     : ' + pred_box[j] + '\n')
                                    wrong.write('\n')
                                    cl,pct,pt1,pt2 = woCorrespond(pred_box,j)
                                    if ShOW_DIFF_IMG:
                                        cv2.rectangle(img, pt1, pt2, (0, 0, 255), 2)
                                        cv2.putText(img, cl + ":woCOR:"+pct, (pt1[0] - 10, pt1[1] - 5),cv2.FONT_HERSHEY_SIMPLEX, 1, [0, 0, 255], 2)
                                err = "3:woCO "
                                fss.write(err + pred_box[j].strip()+"\n" )
                                fss_all.write(err + pred_box[j].strip()+"\n" )
                                # print ("0----pred_box[j]:{} :{}\n".format(type(pred_box[j]),pred_box[j]))
                        if jmax not in jmax_indexes:
                            if ovmax != -1 * float("inf"):
                                jmax_indexes.append(jmax)
                for index in sorted(jmax_indexes, reverse=True):
                    if ovmax != -1 * float("inf"):
                        try:
                            del gt_box_temp[index]
                        except:
                            continue
                    else:
                        break
                if wrong_temp != []:
                    for l in range(len(wrong_temp)):
                        #  ==>Predict wrong
                        # /media/shyechih/data/stage3_mixed/mix_20percent-1rotated-10percent-10rotated-increase-thin_20200702/images/GH010217_5220_300.jpg
                        # predict     : P 122 926 644 1080 0.998919
                        # ground_truth: C 544 0 906 219
                        if ovmax_temp[l] >= iou_thresh:
                            wrong.write(' ==>Predict wrong\n')
                            wrong.write(i + '\n')
                            try:
                                wrong.write('predict     : ' + wrong_temp[l] + '\n')
                                wrong.write('ground_truth: ' + ans_temp[l] + '\n')
                                cl,clg,pct,pt1,pt2,pt3,pt4 = predictWrong(wrong_temp,ans_temp,l)
                                if ShOW_DIFF_IMG:
                                    cv2.rectangle(img, pt1, pt2, (255, 0, 0), 2)
                                    cv2.putText(img, cl + ":pred_F:" + pct, (pt1[0] - 10, pt1[1] - 5),cv2.FONT_HERSHEY_SIMPLEX, 1, [255, 0, 0], 2)
                                    cv2.rectangle(img, pt3, pt4, (0, 255, 0), 2)
                                    cv2.putText(img, clg + "_pred_F", (pt3[0] - 10, pt3[1] - 5),cv2.FONT_HERSHEY_SIMPLEX, 1, [0, 255, 0], 2)
                                # print("1----ans_temp[l]:{} :{}\n".format(type(ans_temp[l]), ans_temp[l]))
                                # print("1----wrong_temp[l]:{} :{}\n".format(type(wrong_temp[l]), wrong_temp[l]))
                            except:
                                continue
                            err = "0:prdW "
                            fss.write(err + ans_temp[l].strip() +" "+ wrong_temp[l].strip()+"\n")
                            fss_all.write(err + ans_temp[l].strip() +" "+ wrong_temp[l].strip()+"\n")
                        else:
                        # ==>IOU too small
                        # /media/shyechih/data/stage3_mixed/mix_20percent-1rotated-10percent-10rotated-increase-thin_20200702/images/GH010227_24540_120.jpg
                        # predict     : P 0 482 169 953 0.925573
                        # ground_truth: P 0 343 326 760
                            wrong.write(' ==>IOU too small\n')
                            wrong.write(i + '\n')
                            try:
                                wrong.write('predict     : ' + wrong_temp[l] + '\n')
                                wrong.write('ground_truth: ' + ans_temp[l] + '\n')
                                cl,clg,pct,pt1,pt2,pt3,pt4 = iou_2small(wrong_temp,ans_temp,l)
                                if ShOW_DIFF_IMG:
                                    cv2.rectangle(img, pt1, pt2, (255, 0,255), 2)
                                    cv2.putText(img, cl + ":iou_S:" + pct, (pt1[0] - 10, pt1[1] - 5),cv2.FONT_HERSHEY_SIMPLEX, 1, [255, 0, 255], 2)
                                    cv2.rectangle(img, pt3, pt4, (0, 255, 0), 2)
                                    cv2.putText(img, clg + "_iou", (pt3[0] - 10, pt3[1] - 5),cv2.FONT_HERSHEY_SIMPLEX, 1, [0, 255, 0], 2)
                                # cv2.imshow(imgFileName, img)
                                # cv2.imwrite(diff_path + imgFileName, img)
                                # print("--2--ans_temp[l]:{} :{}\n".format(type(ans_temp[l]), ans_temp[l]))
                                # print("--2--wrong_temp[l]:{} :{}\n".format(type(wrong_temp[l]), wrong_temp[l]))
                            except:
                                continue
                            err = "1:iou "
                            fss.write(err + ans_temp[l].strip() +" " +wrong_temp[l].strip()+"\n")
                            fss_all.write(err + ans_temp[l].strip() +" " +wrong_temp[l].strip()+"\n")
                        wrong.write('\n')

                if gt_box_temp != []:
                #== > Without prediction
                #/media/shyechih/data/stage3_mixed/mix_20percent-1rotated-10percent-10rotated-increase-thin_20200702/images/GH010203_3960.jpg
                # Ch 789 222 1039 662
                # C 1053 893 1467 1069
                    wrong.write(' ==>Without prediction\n')
                    wrong.write(i + '\n')
                    for l in range(len(gt_box_temp)):
                        # print ("---woPred:{}\n==:{}\n----:{}\n".format(gt_box_temp[l].split(" "),gt_box_temp[l].split(" ")[0],gt_box_temp[l].split(" ")[1]))
                        wrong.write(gt_box_temp[l] + '\n')
                        cl,pt1,pt2 = woPredict(gt_box_temp,l)
                        cv2.rectangle(img, pt1, pt2, (255, 255, 255), 2)
                        cv2.putText(img, cl + ":woPred:", (pt1[0] - 10, pt1[1] - 5),cv2.FONT_HERSHEY_SIMPLEX, 1, [255, 255, 255], 2)
                        err = "2:woP "
                        fss.write(err + gt_box_temp[l].strip())
                        fss_all.write(err + gt_box_temp[l].strip())
                        # cv2.imshow(imgFileName, img)
                        # cv2.imwrite(diff_path + imgFileName, img)
                        print("-3---gt_box_temp[l]:{} :{}\n".format(type(gt_box_temp[l]), gt_box_temp[l])+"\n")
                    wrong.write('\n')

                # print("\n\n--666--{}: \ngt_box_temp-{}:\n:pred_data=={}:\n".format(i.split("/")[-1],  (gt_box),  (pred_box)))
                #i == /home/gilbert3/Documents/mnt/1000/images/GH012601_9480.jpg
                print ("=========:{}:{}:gt_all:{} , pred_all:{} , correct_in:{}\t".format(count,"[]" ,len(gt_box), len(pred_box), correct_internal))
                if ShOW_DIFF_IMG:
                    cv2.imwrite(diff_path + "diff_"+ imgFileName, img)
                fss.write("\n* gt_all:{} , pred_all:{} , correct_in:{}".format(len(gt_box), len(pred_box), (correct_internal)))
                fss_all.write("\n* gt_all:{} , pred_all:{} , correct_in:{}\n\n".format(len(gt_box), len(pred_box),(correct_internal)))
                fss.close()
                count = count + 1
                jmax_indexes = []
                wrong_temp = []
                ans_temp = []
            ##--------------------------end
            fss_all.close()
            if proposals.any() == 0 :
                precision = 0.0
            else:
                precision = 1.0 * np.sum(correct) / np.sum(proposals)
            recall = 1.0 * np.sum(correct) / np.sum(total)
            if precision + recall != 0:
                fscore = 2.0 * precision * recall / (precision + recall)
            else:
                fscore = 0
            acc.write('Precision = ' + str(precision) + '\n')
            acc.write('Recall = ' + str(recall) + '\n')
            acc.write('Fscore = ' + str(fscore) + '\n')
            acc.write('\n')
            for i in range(len(classList_sorted)):
                if total[i] == 0:
                    acc.write(classList_sorted[i] + ' accuracy' + ' = ' + 'NAN' + '\n')
                else:
                    acc.write(classList_sorted[i] + ' accuracy ' + ' = ' + str(1.0 * correct[i] / total[i]) +
                              ' (' + str(int(correct[i])) + '/' + str(int(total[i])) + ')' + '\n')

            acc.close()
            wrong.close()

# def main():
#     wrong = open(parameters.wrong_txt, 'w')
#     wrong.close()
#     wrong = open(parameters.wrong_txt, 'a')
#
#     if not os.path.exists(parameters.accuracy_path):
#         os.makedirs(parameters.accuracy_path)
#     pred_files = os.listdir(parameters.predict_txt)
#
#
#     for file in pred_files:
#         if file.split('_')[-1] == 'final.txt':
#             gt = open(parameters.gt_txt, 'r')
#             lines_gt = gt.readlines()
#
#             pred = open(parameters.predict_txt + '/' + file, 'r')
#             lines_pred = pred.readlines()
#
#             acc = open(parameters.accuracy_path + '/' + 'acc_' + file, 'w')
#             acc.close()
#             acc = open(parameters.accuracy_path+ '/' + 'acc_' + file, 'a')
#
#             iou_thresh = 0.5
#             # thresh = 0.001
#             gt_data = {}
#             gt_data_temp = {}
#             pred_data = {}
#             photo_name = []
#             total_img = []
#             jmax_indexes = []
#             wrong_temp = []
#
#
#             correct = np.zeros(len(classList_sorted))
#             proposals = np.zeros(len(classList_sorted))
#             total = np.zeros(len(classList_sorted))
#
#             for i in range(len(lines_gt)):
#                 if '/' in lines_gt[i]:
#                     if i != 0:
#                         gt_data[photo_name] = data
#                         gt_data_temp[photo_name] = data_temp
#                     data = []
#                     data_temp = []
#                     photo_name = lines_gt[i].strip()
#                     total_img.append(lines_gt[i].strip())
#                 else:
#                     for j in range(len(classList_sorted)):
#                         if classList_sorted[j] == lines_gt[i].split(' ')[0]:
#                             total[j] += 1
#                             break
#                     data.append(lines_gt[i].strip())
#                     data_temp.append(lines_gt[i].strip())
#                 if i == len(lines_gt) - 1:
#                     gt_data[photo_name] = data
#                     gt_data_temp[photo_name] = data_temp
#
#
#             for i in range(len(lines_pred)):
#                 if '/' in lines_pred[i]:
#                     if i != 0: pred_data[photo_name] = data
#                     photo_name = lines_pred[i].strip()
#                     data = []
#                 else:
#                     for j in range(len(classList_sorted)):
#                         if classList_sorted[j] == lines_pred[i].split(' ')[0]:
#                             proposals[j] += 1
#                             break
#                     data.append(lines_pred[i].strip())
#                 if i == len(lines_pred) - 1:
#                     pred_data[photo_name.strip()] = data
#             print ("-----{}:\n:gt_data=={}:\n".format( type(gt_data),len(gt_data ) ))
#             print ("\n\n-4--{}-:\n--{}:\n:total_img=={}:\n".format(total_img,type(total_img),len(total_img)))#total_img[:20]) )
#             #total_img ==['/home/gilbert3/Documents/mnt/1000/images/GH012586_2520.jpg','/home/gilbert3/Documents/mnt/1000/images/GH012603_8040.jpg'
#             print ("\n\n--3--{}: \n-{}:\n:pred_data=={}:\n".format( pred_data,type(pred_data),len(pred_data) ))
#             for i in total_img:
#                 gt_box = gt_data[i]
#                 gt_box_temp = gt_data_temp[i]
#                 pred_box = pred_data[i]
#                 ans_temp = []
#                 ovmax_temp = []
#                 for j in range(len(pred_box)):
#                     if pred_box[j] != '\n':
#                         overlaps = []
#                         x1_pred = int(pred_box[j].split(' ')[1])
#                         y1_pred = int(pred_box[j].split(' ')[2])
#                         x2_pred = int(pred_box[j].split(' ')[3])
#                         y2_pred = int(pred_box[j].split(' ')[4])
#                         if len(gt_box) != 0:
#                             for k in range(len(gt_box)):
#                                 x1_gt = int(gt_box[k].split(' ')[1])
#                                 y1_gt = int(gt_box[k].split(' ')[2])
#                                 x2_gt = int(gt_box[k].split(' ')[3])
#                                 y2_gt = int(gt_box[k].split(' ')[4])
#
#                                 inter_x1 = max(x1_gt, x1_pred)
#                                 inter_y1 = max(y1_gt, y1_pred)
#                                 inter_x2 = min(x2_gt, x2_pred)
#                                 inter_y2 = min(y2_gt, y2_pred)
#                                 inter = (max(inter_x2 - inter_x1, 0)) * (max(inter_y2 - inter_y1, 0))
#                                 uni = (x2_gt - x1_gt) * (y2_gt - y1_gt) + \
#                                       (x2_pred - x1_pred) * (y2_pred - y1_pred) - \
#                                       inter
#                                 overlaps.append(float(inter / uni))
#                             if overlaps != [0.0] * len(gt_box):
#                                 ovmax = np.max(np.array(overlaps))
#                                 jmax = np.argmax(np.array(overlaps))
#                             else:
#                                 ovmax = -1 * float("inf")
#                                 jmax = float("inf")
#                             if ovmax >= iou_thresh and gt_box[jmax].split(' ')[0] == pred_box[j].split(' ')[0]:
#                                 for k in range(len(classList_sorted)):
#                                     if classList_sorted[k] == gt_box[jmax].split(' ')[0]:
#                                         correct[k] += 1
#                                         break
#                             else:
#                                 if ovmax != -1 * float("inf"):
#                                     wrong_temp.append(pred_box[j])
#                                     ans_temp.append(gt_box[jmax])
#                                     ovmax_temp.append(ovmax)
#                                 else:
#                                     wrong.write(' ==>Without correspond\n')
#                                     wrong.write(i + '\n')
#                                     wrong.write('predict     : ' + pred_box[j] + '\n')
#                                     wrong.write('\n')
#                         if jmax not in jmax_indexes:
#                             if ovmax != -1 * float("inf"):
#                                 jmax_indexes.append(jmax)
#                 for index in sorted(jmax_indexes, reverse=True):
#                     if ovmax != -1 * float("inf"):
#                         try:
#                             del gt_box_temp[index]
#                         except:
#                             continue
#                     else:
#                         break
#                 if wrong_temp != []:
#                     for l in range(len(wrong_temp)):
#                         if ovmax_temp[l] >= iou_thresh:
#                             wrong.write(' ==>Predict wrong\n')
#                             wrong.write(i + '\n')
#                             try:
#                                 wrong.write('predict     : ' + wrong_temp[l] + '\n')
#                                 wrong.write('ground_truth: ' + ans_temp[l] + '\n')
#                             except:
#                                 continue
#                         else:
#                             wrong.write(' ==>IOU too small\n')
#                             wrong.write(i + '\n')
#                             try:
#                                 wrong.write('predict     : ' + wrong_temp[l] + '\n')
#                                 wrong.write('ground_truth: ' + ans_temp[l] + '\n')
#                             except:
#                                 continue
#                         wrong.write('\n')
#
#                 if gt_box_temp != []:
#                     wrong.write(' ==>Without prediction\n')
#                     wrong.write(i + '\n')
#                     for l in range(len(gt_box_temp)):
#                         wrong.write(gt_box_temp[l] + '\n')
#                     wrong.write('\n')
#                 jmax_indexes = []
#                 wrong_temp = []
#                 ans_temp = []
#
#             precision = 1.0 * np.sum(correct) / np.sum(proposals)
#             recall = 1.0 * np.sum(correct) / np.sum(total)
#             if precision + recall != 0:
#                 fscore = 2.0 * precision * recall / (precision + recall)
#             else:
#                 fscore = 0
#             acc.write('Precision = ' + str(precision) + '\n')
#             acc.write('Recall = ' + str(recall) + '\n')
#             acc.write('Fscore = ' + str(fscore) + '\n')
#             acc.write('\n')
#             for i in range(len(classList_sorted)):
#                 if total[i] == 0:
#                     acc.write(classList_sorted[i] + ' accuracy' + ' = ' + 'NAN' + '\n')
#                 else:
#                     acc.write(classList_sorted[i] + ' accuracy ' + ' = ' + str(1.0 * correct[i] / total[i]) +
#                               ' (' + str(int(correct[i])) + '/' + str(int(total[i])) + ')' + '\n')
#
#             acc.close()
#             wrong.close()


def yolov3_final_2txt():
    try:
        if Pred_generate_txt == True:
            fp0 = open(os.path.join(parameters.predict_txt , "yolov3_final.txt"), 'r')
            lines = fp0.readlines()
            tmp = ""
            for i in range(len(lines)):
                if lines[i][0] == '/':
                    # print(i, lines[i].split("/")[-1].split(".")[0] + ".txt")
                    tmp = lines[i].split("/")[-1].split(".")[0] + ".txt"
                else:
                    # print(lines[i].strip())
                    with open(os.path.join(parameters.predict_txt , tmp), "a") as fd:
                        fd.write(lines[i].strip() + "\n")
    except:
        print ("\nError-----No such file :at ./predict/yolov3_final.txt \n ")
        exit(0)


def print_2Box(GTtxt, cap, classList_sorted, gt_pred_path, id, imgFileName, lines):
    lines_2 = open(GTtxt, 'r').readlines()
    if lines_2 != []:
        [h, w, ch] = cv2.imread(lines[id].split('\n')[0]).shape
        ret, img = cap.read()
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
            pt1 = (x1, y1)
            pt2 = (x2, y2)
            cv2.rectangle(img, pt1, pt2, (0, 255, 0), 2)
            cv2.putText(img, classList_sorted[int(class_id)], (pt1[0] - 10, pt1[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 1,[0, 255, 0], 2)

        Predict_txt = os.path.join(parameters.predict_txt , lines[id].split("/")[-1].strip().replace("jpg", 'txt'))
        # read corresponding predict GH012582_12400.txt
        fp_predict = open(Predict_txt, "r")
        lines_predict = fp_predict.readlines()

        for k in range(len(lines_predict)):
            kk = lines_predict[k].split()
            pt1 = (int(kk[1]), int(kk[2]))
            pt2 = (int(kk[3]), int(kk[4]))
            cv2.rectangle(img, pt1, pt2, (255, 0, 0), 1)
            cv2.putText(img, kk[0], (pt2[0], pt1[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, [255, 0, 0], 2)
    # cv2.imshow(imgFileName, img)
    cv2.imwrite(gt_pred_path + imgFileName, img)

def save_GT_Predict():
    # parser yolov3_final.txt at ./predict/*.jpg -> ./predict/*.txt
    yolov3_final_2txt()

    classes = len(parameters.classList)
    gt_pred_path = os.path.join(parameters.outputPath ,"gt_pred_Dir/")
    if not os.path.exists(gt_pred_path):
        os.makedirs(gt_pred_path)
    folderCharacter = "/"
    ## parameter I1/folder1:img.jpg , I2/folder2:img.txt , O1/folder3.img, O2/folder4.lab
    with open(parameters.cfgFolder + folderCharacter + parameters.cfg_obj_names, 'w') as the_file:
        classList_sorted = {name: i for i, name in parameters.classList.items()}
        # for class_id in range(classes):
        #     print(class_id, ":", classList_sorted[class_id] + "\n")

    lines = open(os.path.join(parameters.cfgFolder ,"test.txt"), 'r').readlines()
    # read test.txt filename mapping 2 gt.txt with corresponding answer("./labels/*.txt")together ==> gt.txt
    for id in range(len(lines)):
        print(".")# print ("\n---------------:{}:\n".format(lines[i]))

        # GH012582_12400.jpg == lines[i].split('/')[-1]
        image_type = lines[id].split('/')[-1].split('.')[-1]
        imgFileName = lines[id].split('/')[-1].strip()
        #read lines getting test.txt and replace .jpg 2 label.txt

        # print ("\n1:----{} :txt :{}:\n{}:-------\n".format(i,parameters.testPathLabel.split("/")[-1],txt))
        # 1: ----0:txt: labels:
        # / home / gilbert3 / Documents / mnt / 1000 / labels / GH012586_2520.txt: -------

        GTtxt = lines[id].split('.')[0] + '.txt'

        # print ("\n2:----{} :txt :{}:\n{}:-------\n".format(i,lines[i].split('.')[0],txt))
        # 2: ----0:txt: / home / gilbert3 / Documents / mnt / 1000 / images / GH012586_2520:
        # / home / gilbert3 / Documents / mnt / 1000 / images / GH012586_2520.txt: -------
        # cp test.txt all *.jpg correspond answer ("labels_xxx/*.txt") to gt.txt
        GTtxt = GTtxt.replace('images', parameters.testPathLabel.split("/")[-1])
        # print ("\n3:----{} :txt :{}:\n{}:-------\n".format(i,lines[i].split('.')[0],Predict_txt))
        cap = cv2.VideoCapture(lines[id].strip())  # ret, img = cap.read()

        print_2Box(GTtxt, cap, classList_sorted, gt_pred_path, id, imgFileName, lines)

        # if cv2.waitKey(0) & 0xFF == ord('q'):
        #     break
    cv2.destroyAllWindows()

if __name__ == '__main__':
    perform()
    # # print GT with Predict img side by side
    # if ShOW_GT_Predict_IMG:
    #     save_GT_Predict()
    # wrong = open(parameters.wrong_txt, 'w')
    # wrong.close()
    # wrong = open(parameters.wrong_txt, 'a')
    #
    # if not os.path.exists(parameters.accuracy_path):
    #     os.makedirs(parameters.accuracy_path)
    # try:
    #     if parameters.dest_weight:
    #         pred_files = [parameters.dest_weight.split('/')[-1].split('.')[0] + '.txt']
    #
    # except:
    #     pred_files = os.listdir(parameters.predict_txt)
    #
    # for file in pred_files:
    #     if file.split('_')[-1] == 'final.txt':
    #         gt = open(parameters.gt_txt, 'r')
    #         lines_gt = gt.readlines()
    #
    #         pred = open(parameters.predict_txt + '/' + file, 'r')
    #         lines_pred = pred.readlines()
    #
    #         acc = open(parameters.accuracy_path + '/' + 'acc_' + file, 'w')
    #         acc.close()
    #         acc = open(parameters.accuracy_path + '/' + 'acc_' + file, 'a')
    #
    #         iou_thresh = 0.5
    #         # thresh = 0.001
    #         gt_data = {}
    #         gt_data_temp = {}
    #         pred_data = {}
    #         photo_name = []
    #         total_img = []
    #         jmax_indexes = []
    #         wrong_temp = []
    #
    #
    #         correct = np.zeros(len(classList_sorted))
    #         proposals = np.zeros(len(classList_sorted))
    #         total = np.zeros(len(classList_sorted))
    #
    #         for i in range(len(lines_gt)):
    #             if '/' in lines_gt[i]:
    #                 if i != 0:
    #                     gt_data[photo_name] = data
    #                     gt_data_temp[photo_name] = data_temp
    #                 data = []
    #                 data_temp = []
    #                 photo_name = lines_gt[i].strip()
    #                 #(lines_gt[i].strip() == /home/gilbert3/Documents/mnt/1000/images/GH012601_9480.jpg:,
    #                 # (photo_name) ==/home/gilbert3/Documents/mnt/1000/images/GH012601_9480.jpg:
    #                 total_img.append(lines_gt[i].strip())
    #             else:
    #                 for j in range(len(classList_sorted)):
    #                     if classList_sorted[j] == lines_gt[i].split(' ')[0]:
    #                         total[j] += 1
    #                         break
    #                 data.append(lines_gt[i].strip())
    #                 data_temp.append(lines_gt[i].strip())
    #             if i == len(lines_gt) - 1:
    #                 gt_data[photo_name] = data
    #                 gt_data_temp[photo_name] = data_temp
    #
    #
    #         for i in range(len(lines_pred)):
    #             if '/' in lines_pred[i]:
    #                 if i != 0: pred_data[photo_name] = data
    #                 photo_name = lines_pred[i].strip()
    #                 data = []
    #             else:
    #                 for j in range(len(classList_sorted)):
    #                     if classList_sorted[j] == lines_pred[i].split(' ')[0]:
    #                         proposals[j] += 1
    #                         break
    #                 data.append(lines_pred[i])#.strip())
    #             if i == len(lines_pred) - 1:
    #                 pred_data[photo_name.strip()] = data
    #         # print ("-----{}:\n:gt_data=={}:\n".format( type(gt_data),gt_data ) )
    #         # print ("-----{}:\n:total_img=={}:\n".format(len(total_img),total_img[:20]) )
    #         # print ("-----{}:\n:pred_data=={}:\n".format( type(pred_data),pred_data) )
    #         # print ("\n---11--{}----11---{}:\n:gt_data=={}:\n".format(correct , len(classList_sorted) ,len(gt_data ) ))
    #         # photo_name.strip() ==/home/gilbert3/Documents/mnt/1000/images/GH012601_9480.jpg
    #         # print ("\n\n-444--{}-:\n--{}:\n:total_img=={}:\n".format(total_img,type(total_img),len(total_img)))#total_img[:20]) )
    #         #-444--['/home/gilbert3/Documents/mnt/1000/images/GH012586_2520.jpg', '/home/gilbert3/Documents/mnt/1000/images/GH012603_8040.jpg', '/home/gilbert3/Documents/mnt/1000/images/GH012601_6840.jpg',
    #         #:total_img==667:
    #         # print ("\n\n--333--{}: \n-{}:\n:pred_data=={}:\n".format( pred_data,type(pred_data),len(pred_data) ))
    #         #--333 - -{'/home/gilbert3/Documents/mnt/1000/images/GH012586_2520.jpg': ['T 32 16 693 416 0.986909\n','T 604 76 934 433 0.566341\n','Ot 5
    #         #-<class 'dict'>:
    #         #:pred_data==667:
    #         folderCharacter = "/"
    #         diff_path = parameters.outputPath + "/" + "diffDir/"
    #         if not os.path.exists(diff_path):
    #                 os.makedirs(diff_path)
    #
    #         fss_all = open(parameters.outputPath + "/diffDir/" + "yolov3_final_diff.txt", "w")
    #         ##--------------------------start
    #         count = 0
    #         for i in total_img:
    #             gt_box = gt_data[i]
    #             gt_box_temp = gt_data_temp[i]
    #             pred_box = pred_data[i]
    #             ans_temp = []
    #             ovmax_temp = []
    #
    #             cap = cv2.VideoCapture(i.strip())
    #             ret, img = cap.read()
    #             imgFileName = i.split('/')[-1].strip()
    #             txtFileName = imgFileName.replace("jpg","txt")
    #             fss = open(parameters.outputPath+"/diffDir/"+txtFileName,"w")
    #             fss_all.write(i+"\n")
    #             correct_internal = 0
    #             for j in range(len(pred_box)):
    #                 # if (j < 2):
    #                 #     print ("---555--\ngt_box:{}\ngt_box_temp:{}\n".format( gt_box ,type(gt_box)))#pred_box[j] , pred_box[j].split(' ') ) )
    #                 # ret, img = cap.read()
    #                 if pred_box[j] != '\n':
    #                     overlaps = []
    #                     x1_pred = int(pred_box[j].split(' ')[1])
    #                     y1_pred = int(pred_box[j].split(' ')[2])
    #                     x2_pred = int(pred_box[j].split(' ')[3])
    #                     y2_pred = int(pred_box[j].split(' ')[4])
    #                     if len(gt_box) != 0:
    #                         for k in range(len(gt_box)):
    #                             x1_gt = int(gt_box[k].split(' ')[1])
    #                             y1_gt = int(gt_box[k].split(' ')[2])
    #                             x2_gt = int(gt_box[k].split(' ')[3])
    #                             y2_gt = int(gt_box[k].split(' ')[4])
    #
    #                             inter_x1 = max(x1_gt, x1_pred)
    #                             inter_y1 = max(y1_gt, y1_pred)
    #                             inter_x2 = min(x2_gt, x2_pred)
    #                             inter_y2 = min(y2_gt, y2_pred)
    #                             inter = (max(inter_x2 - inter_x1, 0)) * (max(inter_y2 - inter_y1, 0))
    #                             uni = (x2_gt - x1_gt) * (y2_gt - y1_gt) + (x2_pred - x1_pred) * (y2_pred - y1_pred) - inter
    #                             overlaps.append(float(inter / uni))
    #                         if overlaps != [0.0] * len(gt_box):
    #                             ovmax = np.max(np.array(overlaps))
    #                             jmax = np.argmax(np.array(overlaps))
    #                         else:
    #                             ovmax = -1 * float("inf")
    #                             jmax = float("inf")
    #                         # gt_box["GH012582_12400.txt"][0] == pred_box["GH012582_12400.txt"][0]
    #                         if ovmax >= iou_thresh and gt_box[jmax].split(' ')[0] == pred_box[j].split(' ')[0]:
    #                             for k in range(len(classList_sorted)):
    #                                 if classList_sorted[k] == gt_box[jmax].split(' ')[0]:
    #                                     correct_internal += 1
    #                                     correct[k] += 1
    #                                     break
    #                             err = "4:Right "
    #                             fss.write(err + gt_box[jmax].strip() +" "+ pred_box[j].strip() +"\n")
    #                             fss_all.write(err + gt_box[jmax].strip() +" "+ pred_box[j].strip() +"\n")
    #                         else:
    #                             if ovmax != -1 * float("inf"):
    #                                 wrong_temp.append(pred_box[j])
    #                                 ans_temp.append(gt_box[jmax])
    #                                 ovmax_temp.append(ovmax)
    #                             else:
    #                                 # ==> Without correspond
    #                                 # /media/shyechih/data/stage3_mixed/mix_20percent-1rotated-10percent-10rotated-increase-thin_20200702/images/GH010227_5580_090.jpg
    #                                 # predict     : Ch 1517 752 1850 1079 0.881586
    #                                 wrong.write(' ==>Without correspond\n')
    #                                 wrong.write(i + '\n')
    #                                 wrong.write('predict     : ' + pred_box[j] + '\n')
    #                                 wrong.write('\n')
    #                                 cl,pct,pt1,pt2 = woCorrespond(pred_box,j)
    #                                 if ShOW_DIFF_IMG:
    #                                     cv2.rectangle(img, pt1, pt2, (0, 0, 255), 2)
    #                                     cv2.putText(img, cl + ":woCOR:"+pct, (pt1[0] - 10, pt1[1] - 5),cv2.FONT_HERSHEY_SIMPLEX, 1, [0, 0, 255], 2)
    #                             err = "3:woCO "
    #                             fss.write(err + pred_box[j].strip()+"\n" )
    #                             fss_all.write(err + pred_box[j].strip()+"\n" )
    #                             # print ("0----pred_box[j]:{} :{}\n".format(type(pred_box[j]),pred_box[j]))
    #                     if jmax not in jmax_indexes:
    #                         if ovmax != -1 * float("inf"):
    #                             jmax_indexes.append(jmax)
    #             for index in sorted(jmax_indexes, reverse=True):
    #                 if ovmax != -1 * float("inf"):
    #                     try:
    #                         del gt_box_temp[index]
    #                     except:
    #                         continue
    #                 else:
    #                     break
    #             if wrong_temp != []:
    #                 for l in range(len(wrong_temp)):
    #                     #  ==>Predict wrong
    #                     # /media/shyechih/data/stage3_mixed/mix_20percent-1rotated-10percent-10rotated-increase-thin_20200702/images/GH010217_5220_300.jpg
    #                     # predict     : P 122 926 644 1080 0.998919
    #                     # ground_truth: C 544 0 906 219
    #                     if ovmax_temp[l] >= iou_thresh:
    #                         wrong.write(' ==>Predict wrong\n')
    #                         wrong.write(i + '\n')
    #                         try:
    #                             wrong.write('predict     : ' + wrong_temp[l] + '\n')
    #                             wrong.write('ground_truth: ' + ans_temp[l] + '\n')
    #                             cl,clg,pct,pt1,pt2,pt3,pt4 = predictWrong(wrong_temp,ans_temp,l)
    #                             if ShOW_DIFF_IMG:
    #                                 cv2.rectangle(img, pt1, pt2, (255, 0, 0), 2)
    #                                 cv2.putText(img, cl + ":pred_F:" + pct, (pt1[0] - 10, pt1[1] - 5),cv2.FONT_HERSHEY_SIMPLEX, 1, [255, 0, 0], 2)
    #                                 cv2.rectangle(img, pt3, pt4, (0, 255, 0), 2)
    #                                 cv2.putText(img, clg + "_pred_F", (pt3[0] - 10, pt3[1] - 5),cv2.FONT_HERSHEY_SIMPLEX, 1, [0, 255, 0], 2)
    #                             # print("1----ans_temp[l]:{} :{}\n".format(type(ans_temp[l]), ans_temp[l]))
    #                             # print("1----wrong_temp[l]:{} :{}\n".format(type(wrong_temp[l]), wrong_temp[l]))
    #                         except:
    #                             continue
    #                         err = "0:prdW "
    #                         fss.write(err + ans_temp[l].strip() +" "+ wrong_temp[l].strip()+"\n")
    #                         fss_all.write(err + ans_temp[l].strip() +" "+ wrong_temp[l].strip()+"\n")
    #                     else:
    #                     # ==>IOU too small
    #                     # /media/shyechih/data/stage3_mixed/mix_20percent-1rotated-10percent-10rotated-increase-thin_20200702/images/GH010227_24540_120.jpg
    #                     # predict     : P 0 482 169 953 0.925573
    #                     # ground_truth: P 0 343 326 760
    #                         wrong.write(' ==>IOU too small\n')
    #                         wrong.write(i + '\n')
    #                         try:
    #                             wrong.write('predict     : ' + wrong_temp[l] + '\n')
    #                             wrong.write('ground_truth: ' + ans_temp[l] + '\n')
    #                             cl,clg,pct,pt1,pt2,pt3,pt4 = iou_2small(wrong_temp,ans_temp,l)
    #                             if ShOW_DIFF_IMG:
    #                                 cv2.rectangle(img, pt1, pt2, (255, 0,255), 2)
    #                                 cv2.putText(img, cl + ":iou_S:" + pct, (pt1[0] - 10, pt1[1] - 5),cv2.FONT_HERSHEY_SIMPLEX, 1, [255, 0, 255], 2)
    #                                 cv2.rectangle(img, pt3, pt4, (0, 255, 0), 2)
    #                                 cv2.putText(img, clg + "_iou", (pt3[0] - 10, pt3[1] - 5),cv2.FONT_HERSHEY_SIMPLEX, 1, [0, 255, 0], 2)
    #                             # cv2.imshow(imgFileName, img)
    #                             # cv2.imwrite(diff_path + imgFileName, img)
    #                             # print("--2--ans_temp[l]:{} :{}\n".format(type(ans_temp[l]), ans_temp[l]))
    #                             # print("--2--wrong_temp[l]:{} :{}\n".format(type(wrong_temp[l]), wrong_temp[l]))
    #                         except:
    #                             continue
    #                         err = "1:iou "
    #                         fss.write(err + ans_temp[l].strip() +" " +wrong_temp[l].strip()+"\n")
    #                         fss_all.write(err + ans_temp[l].strip() +" " +wrong_temp[l].strip()+"\n")
    #                     wrong.write('\n')
    #
    #             if gt_box_temp != []:
    #             #== > Without prediction
    #             #/media/shyechih/data/stage3_mixed/mix_20percent-1rotated-10percent-10rotated-increase-thin_20200702/images/GH010203_3960.jpg
    #             # Ch 789 222 1039 662
    #             # C 1053 893 1467 1069
    #                 wrong.write(' ==>Without prediction\n')
    #                 wrong.write(i + '\n')
    #                 for l in range(len(gt_box_temp)):
    #                     # print ("---woPred:{}\n==:{}\n----:{}\n".format(gt_box_temp[l].split(" "),gt_box_temp[l].split(" ")[0],gt_box_temp[l].split(" ")[1]))
    #                     wrong.write(gt_box_temp[l] + '\n')
    #                     cl,pt1,pt2 = woPredict(gt_box_temp,l)
    #                     cv2.rectangle(img, pt1, pt2, (255, 255, 255), 2)
    #                     cv2.putText(img, cl + ":woPred:", (pt1[0] - 10, pt1[1] - 5),cv2.FONT_HERSHEY_SIMPLEX, 1, [255, 255, 255], 2)
    #                     err = "2:woP "
    #                     fss.write(err + gt_box_temp[l].strip())
    #                     fss_all.write(err + gt_box_temp[l].strip())
    #                     # cv2.imshow(imgFileName, img)
    #                     # cv2.imwrite(diff_path + imgFileName, img)
    #                     print("-3---gt_box_temp[l]:{} :{}\n".format(type(gt_box_temp[l]), gt_box_temp[l])+"\n")
    #                 wrong.write('\n')
    #
    #             # print("\n\n--666--{}: \ngt_box_temp-{}:\n:pred_data=={}:\n".format(i.split("/")[-1],  (gt_box),  (pred_box)))
    #             #i == /home/gilbert3/Documents/mnt/1000/images/GH012601_9480.jpg
    #             print ("=========:{}:{}:gt_all:{} , pred_all:{} , correct_in:{}\t".format(count,"[]" ,len(gt_box), len(pred_box), correct_internal))
    #             if ShOW_DIFF_IMG:
    #                 cv2.imwrite(diff_path + "diff_"+ imgFileName, img)
    #             fss.write("\n* gt_all:{} , pred_all:{} , correct_in:{}".format(len(gt_box), len(pred_box), (correct_internal)))
    #             fss_all.write("\n* gt_all:{} , pred_all:{} , correct_in:{}\n".format(len(gt_box), len(pred_box),(correct_internal)))
    #             fss.close()
    #             count = count + 1
    #             jmax_indexes = []
    #             wrong_temp = []
    #             ans_temp = []
    #         ##--------------------------end
    #         fss_all.close()
    #         if proposals.any() == 0 :
    #             precision = 0.0
    #         else:
    #             precision = 1.0 * np.sum(correct) / np.sum(proposals)
    #         recall = 1.0 * np.sum(correct) / np.sum(total)
    #         if precision + recall != 0:
    #             fscore = 2.0 * precision * recall / (precision + recall)
    #         else:
    #             fscore = 0
    #         acc.write('Precision = ' + str(precision) + '\n')
    #         acc.write('Recall = ' + str(recall) + '\n')
    #         acc.write('Fscore = ' + str(fscore) + '\n')
    #         acc.write('\n')
    #         for i in range(len(classList_sorted)):
    #             if total[i] == 0:
    #                 acc.write(classList_sorted[i] + ' accuracy' + ' = ' + 'NAN' + '\n')
    #             else:
    #                 acc.write(classList_sorted[i] + ' accuracy ' + ' = ' + str(1.0 * correct[i] / total[i]) +
    #                           ' (' + str(int(correct[i])) + '/' + str(int(total[i])) + ')' + '\n')
    #
    #         acc.close()
    #         wrong.close()


