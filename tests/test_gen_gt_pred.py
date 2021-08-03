import os
import csv
import unittest
from utils.gen_gt_pred import (classid_to_name, gen_gt_content, gen_batch_pred, gen_gt_files)


class TestClassidToName(unittest.TestCase):
    def test_exception(self):
        self.assertRaises(KeyError, classid_to_name, 100)
    def test_classid_to_name(self):
        self.assertEqual(classid_to_name(5), 'Ch')
        self.assertNotEqual(classid_to_name(5), 'Ot')


class TestGenGtContent(unittest.TestCase):
    def test_exception(self):
        # Test wrong img file
        input_img_path = '/media/shyechih/data/stage2_txt/PET_1_exam_7cate/images/GH012603_xxx.jpg'
        input_label_path = '/media/shyechih/data/stage2_txt/PET_1_exam_7cate/labels/GH012603_7560.txt'
        self.assertRaises(FileNotFoundError, gen_gt_content, input_img_path, input_label_path)

        # Test wrong txt(label) file
        input_img_path = '/media/shyechih/data/stage2_txt/PET_1_exam_7cate/images/GH012603_7560.jpg'
        input_label_path = '/media/shyechih/data/stage2_txt/PET_1_exam_7cate/labels/GH012603_what.txt'
        self.assertRaises(FileNotFoundError, gen_gt_content, input_img_path, input_label_path)

    def test_gen_gt_content(self):
    
        input_img_path = '/media/shyechih/data/stage2_txt/PET_1_exam_7cate/images/GH012603_7560.jpg'
        input_label_path = '/media/shyechih/data/stage2_txt/PET_1_exam_7cate/labels/GH012603_7560.txt'

        gt_content = gen_gt_content(input_img_path, input_label_path)
        # output_lines = len(gt_content.split('\n'))
        output_lines = gt_content.count('\n')

        with open(input_label_path) as f:
            reader = csv.reader(f)
            input_lines= len(list(reader))

        self.assertEqual(output_lines, input_lines)



class TestGenBatchPred(unittest.TestCase):
    def test_gen_batch_pred(self):
        input_pred_path = '/media/shyechih/data/stage4_model/ching_PET1_7cate_base-10rotated-anchor-alexcfg/yolov3_cfg/predictions_0.7thresh.txt'
        output_pred_dir = '/media/shyechih/data/stage4_model/ching_PET1_7cate_base-10rotated-anchor-alexcfg/predict'
        pred_content = gen_batch_pred(input_pred_path, output_pred_dir)
        with open(input_pred_path) as f:
            contents = f.read()
            inputs_imgs = contents.lower().count('.jpg')

        self.assertEqual(len(pred_content.keys()), inputs_imgs)
        # print(pred_content)
