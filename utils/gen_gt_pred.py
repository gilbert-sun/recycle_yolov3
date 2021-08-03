# -*- coding: utf-8 -*-
# !/usr/bin/env python3.6

# built-in packages
import os
import re
import csv
import logging
from glob import glob

# installed packages
import cv2

# project packages
import utils


def classid_to_name(classid):
    classid = int(classid)
    mapping = {
        0: 'P',
        1: 'O',
        2: 'S',
        3: 'C',
        4: 'Ot',
        5: 'Ch',
        6: 'T'
    }
    if classid not in mapping.keys():
        raise KeyError('Cant map classid: {}'.format(classid))

    return mapping[classid]


def gen_gt_content(input_img_path, input_label_path):
    # print('gen_gt_content - input_img_path = {}'.format(input_img_path))
    # print('gen_gt_content - os.path.isfile = {}'.format(os.path.islink(input_img_path)))
    # print('gen_gt_content - os.path.islink = {}'.format(os.path.islink(input_img_path)))
    if not os.path.isfile(input_img_path) and not os.path.islink(input_img_path):
    # if os.stat(input_img_path).st_size == 0:
        raise FileNotFoundError(input_img_path)

    if not os.path.isfile(input_label_path) and not os.path.islink(input_label_path):
        raise FileNotFoundError(input_label_path)
    
    output_content = ''
    [h, w, ch] = cv2.imread(input_img_path).shape

    with open(input_label_path, 'r') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=' ')
        for row in spamreader:
            left = round(float(row[1]) * w - float(row[3]) / 2 * w) # x1
            top = round(float(row[2]) * h - float(row[4]) / 2 * h) # y1
            right = round(float(row[1]) * w + float(row[3]) / 2 * w) # x2
            bottom = round(float(row[2]) * h + float(row[4]) / 2 * h) # y2
            
            output_content += classid_to_name(row[0]) + ' ' + \
                                              str(left) + ' ' + \
                                              str(top) + ' ' + \
                                              str(right) + ' ' + \
                                              str(bottom) + '\n'

    return output_content


def gen_batch_pred(input_pred_path, output_pred_dir):
    '''Process the prediction.txt from AlexeyAB's darknet'''
    if not os.path.isfile(input_pred_path):
        raise FileNotFoundError(input_pred_path)

    output_dict = {}
    # T: 97%  (left_x: 1033   top_y:  869   width:  226   height:  207)
    # O: 76%  (left_x: 1033   top_y:  869   width:  226   height:  207)
    # P: 99%  (left_x: 1120   top_y:  114   width:  255   height:  213)
    # C: 100% (left_x: 1187   top_y:  718   width:  397   height:  345)

    with open(input_pred_path, 'r') as f:
        for line in f:
            if '.jpg' in line.lower():
                img_path = line.split(': ')[0]
                output_dict.update({img_path: ''})
                
            if 'left_x:' in line:
                # print('line: {}'.format(line))
                classname = re.search('^.*?(?=:)', line).group(0).strip()
                confidence = int(re.search('(?<=:).*?(?=%)', line).group(0).strip()) * .01
                left = int(re.search('(?<=left_x:).*?(?=top_y)', line).group(0).strip())
                top = int(re.search('(?<=top_y:).*?(?=width)', line).group(0).strip())
                width = int(re.search('(?<=width:).*?(?=height:)', line).group(0).strip())
                height = int(re.search('(?<=height:).*?(?=\))', line).group(0).strip())
                
                left = 0 if left < 0 else left
                top = 0 if top < 0 else top
                width = 0 if width < 0 else width
                height = 0 if height < 0 else height

                right = left + width
                bottom = top + height
                # print('classname: {} / confidence: {} / left: {} / top: {} / right: {} / bottom: {}'.format(classname, confidence, left, top, right, bottom))

                output_dict[img_path] += classname + ' ' + '{:.2f}'.format(confidence) + ' ' + str(left) + ' ' + str(top) + ' ' + str(right) + ' ' + str(bottom) + '\n'

    return output_dict


def gen_gt_files(input_txt_path, output_gt_dir):

    if not os.path.isfile(input_txt_path):
        raise FileNotFoundError(input_txt_path)

    # utils.mkdir_with_check(output_gt_dir)

    with open (input_txt_path, 'r') as f:
        for line in f:
            line = line.strip('\n')
            label_path = re.sub("(?i)jpg", "txt", line).replace('images', 'labels')
            # label_path = line.replace('images', 'labels')
            label_name = label_path.split(os.sep)[-1:][0]
            gt_content = gen_gt_content(line, label_path)
            with open(output_gt_dir + os.sep + label_name, 'w') as f:
                f.write(gt_content)


def gen_pred_files(input_pred_path, output_pred_dir):
    
    if not os.path.isfile(input_pred_path):
        raise FileNotFoundError(input_pred_path)

    # utils.mkdir_with_check(output_gt_dir)

    pred_dict = gen_batch_pred(input_pred_path, output_pred_dir)

    for img_path, pred_content in pred_dict.items():
        pred_name = re.sub("(?i)jpg", "txt", img_path).strip('\n').split(os.sep)[-1:][0]

        with open (os.path.join(output_pred_dir, pred_name), 'w') as f:
            f.write(pred_content)


if __name__ == '__main__':
    input_txt_path = '/home/shyechih/Documents/darknet_visualization/compare_recycle/lists/exam_7cate.txt'
    output_gt_dir = '/home/shyechih/Documents/github_others/Object-Detection-Metrics/groundtruths'
    # gen_gt_files(input_txt_path, output_gt_dir)

    input_pred_path = '/media/shyechih/data/stage4_model/ching_PET1_7cate_base-10rotated-anchor-alexcfg/yolov3_cfg/predictions_0.1thresh.txt'
    # output_pred_dir = '/home/shyechih/Documents/github_others/Object-Detection-Metrics/detections'
    output_pred_dir = '/home/shyechih/Documents/github_others/mAP/input/detection-results'
    # output_pred_dir = '/home/shyechih/Documents/github_others/object-detection-evaluation/detections'
    gen_pred_files(input_pred_path, output_pred_dir)