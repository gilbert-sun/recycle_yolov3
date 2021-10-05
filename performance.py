import parameters
import numpy as np
import os

classList_sorted = {name: i for i, name in parameters.classList.items()}

def main():
    wrong = open(parameters.wrong_txt, 'w')
    wrong.close()
    wrong = open(parameters.wrong_txt, 'a')

    if not os.path.exists(parameters.accuracy_path):
        os.makedirs(parameters.accuracy_path)
    pred_files = os.listdir(parameters.predict_txt)


    for file in pred_files:
        if file.split('_')[-1] == 'final.txt':
            gt = open(parameters.gt_txt, 'r')
            lines_gt = gt.readlines()

            pred = open(parameters.predict_txt + '/' + file, 'r')
            lines_pred = pred.readlines()

            acc = open(parameters.accuracy_path + '/' + 'acc_' + file, 'w')
            acc.close()
            acc = open(parameters.accuracy_path+ '/' + 'acc_' + file, 'a')

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
                    data.append(lines_pred[i].strip())
                if i == len(lines_pred) - 1:
                    pred_data[photo_name.strip()] = data

            for i in total_img:
                gt_box = gt_data[i]
                gt_box_temp = gt_data_temp[i]
                pred_box = pred_data[i]
                ans_temp = []
                ovmax_temp = []
                for j in range(len(pred_box)):
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
                                uni = (x2_gt - x1_gt) * (y2_gt - y1_gt) + \
                                      (x2_pred - x1_pred) * (y2_pred - y1_pred) - \
                                      inter
                                overlaps.append(float(inter / uni))
                            if overlaps != [0.0] * len(gt_box):
                                ovmax = np.max(np.array(overlaps))
                                jmax = np.argmax(np.array(overlaps))
                            else:
                                ovmax = -1 * float("inf")
                                jmax = float("inf")
                            if ovmax >= iou_thresh and gt_box[jmax].split(' ')[0] == pred_box[j].split(' ')[0]:
                                for k in range(len(classList_sorted)):
                                    if classList_sorted[k] == gt_box[jmax].split(' ')[0]:
                                        correct[k] += 1
                                        break
                            else:
                                if ovmax != -1 * float("inf"):
                                    wrong_temp.append(pred_box[j])
                                    ans_temp.append(gt_box[jmax])
                                    ovmax_temp.append(ovmax)
                                else:
                                    wrong.write(' ==>Without correspond\n')
                                    wrong.write(i + '\n')
                                    wrong.write('predict     : ' + pred_box[j] + '\n')
                                    wrong.write('\n')
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
                        if ovmax_temp[l] >= iou_thresh:
                            wrong.write(' ==>Predict wrong\n')
                            wrong.write(i + '\n')
                            try:
                                wrong.write('predict     : ' + wrong_temp[l] + '\n')
                                wrong.write('ground_truth: ' + ans_temp[l] + '\n')
                            except:
                                continue
                        else:
                            wrong.write(' ==>IOU too small\n')
                            wrong.write(i + '\n')
                            try:
                                wrong.write('predict     : ' + wrong_temp[l] + '\n')
                                wrong.write('ground_truth: ' + ans_temp[l] + '\n')
                            except:
                                continue
                        wrong.write('\n')

                if gt_box_temp != []:
                    wrong.write(' ==>Without prediction\n')
                    wrong.write(i + '\n')
                    for l in range(len(gt_box_temp)):
                        wrong.write(gt_box_temp[l] + '\n')
                    wrong.write('\n')
                jmax_indexes = []
                wrong_temp = []
                ans_temp = []

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



if __name__ == '__main__':
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

            pred = open(parameters.predict_txt + '/' + file, 'r')
            lines_pred = pred.readlines()

            acc = open(parameters.accuracy_path + '/' + 'acc_' + file, 'w')
            acc.close()
            acc = open(parameters.accuracy_path + '/' + 'acc_' + file, 'a')

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
                    data.append(lines_pred[i].strip())
                if i == len(lines_pred) - 1:
                    pred_data[photo_name.strip()] = data
            # print ("-----{}:\n:gt_data=={}:\n".format( type(gt_data),gt_data ) )
            # print ("-----{}:\n:total_img=={}:\n".format(len(total_img),total_img[:20]) )
            # print ("-----{}:\n:pred_data=={}:\n".format( type(pred_data),pred_data) )
            for i in total_img:
                gt_box = gt_data[i]
                gt_box_temp = gt_data_temp[i]
                pred_box = pred_data[i]
                ans_temp = []
                ovmax_temp = []
                for j in range(len(pred_box)):
                    if (j < 10):
                        print ("-----\n:{}\n:{}\n".format( pred_box[j] , pred_box[j].split(' ') ) )
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
                                uni = (x2_gt - x1_gt) * (y2_gt - y1_gt) + \
                                      (x2_pred - x1_pred) * (y2_pred - y1_pred) - \
                                      inter
                                overlaps.append(float(inter / uni))
                            if overlaps != [0.0] * len(gt_box):
                                ovmax = np.max(np.array(overlaps))
                                jmax = np.argmax(np.array(overlaps))
                            else:
                                ovmax = -1 * float("inf")
                                jmax = float("inf")
                            if ovmax >= iou_thresh and gt_box[jmax].split(' ')[0] == pred_box[j].split(' ')[0]:
                                for k in range(len(classList_sorted)):
                                    if classList_sorted[k] == gt_box[jmax].split(' ')[0]:
                                        correct[k] += 1
                                        break
                            else:
                                if ovmax != -1 * float("inf"):
                                    wrong_temp.append(pred_box[j])
                                    ans_temp.append(gt_box[jmax])
                                    ovmax_temp.append(ovmax)
                                else:
                                    wrong.write(' ==>Without correspond\n')
                                    wrong.write(i + '\n')
                                    wrong.write('predict     : ' + pred_box[j] + '\n')
                                    wrong.write('\n')
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
                        if ovmax_temp[l] >= iou_thresh:
                            wrong.write(' ==>Predict wrong\n')
                            wrong.write(i + '\n')
                            try:
                                wrong.write('predict     : ' + wrong_temp[l] + '\n')
                                wrong.write('ground_truth: ' + ans_temp[l] + '\n')
                            except:
                                continue
                        else:
                            wrong.write(' ==>IOU too small\n')
                            wrong.write(i + '\n')
                            try:
                                wrong.write('predict     : ' + wrong_temp[l] + '\n')
                                wrong.write('ground_truth: ' + ans_temp[l] + '\n')
                            except:
                                continue
                        wrong.write('\n')

                if gt_box_temp != []:
                    wrong.write(' ==>Without prediction\n')
                    wrong.write(i + '\n')
                    for l in range(len(gt_box_temp)):
                        wrong.write(gt_box_temp[l] + '\n')
                    wrong.write('\n')
                jmax_indexes = []
                wrong_temp = []
                ans_temp = []
            print("1).=====----\n:{}\n:{}\n".format(correct,proposals))
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


