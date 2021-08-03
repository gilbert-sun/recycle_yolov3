#!/bin/bash

SOURCE_DIR='/home/shyechih/Documents/train_verify_recycle'
TARGET_DIR='/media/shyechih/data/stage4_model/sc_mix_20percent-1rotated-10percent-10rotated-increase-thin'

cp -r $SOURCE_DIR/accuracy $TARGET_DIR
cp -r $SOURCE_DIR/predict $TARGET_DIR
cp -r $SOURCE_DIR/yolov3_cfg $TARGET_DIR
cp -r $SOURCE_DIR/accuracy $TARGET_DIR
cp -r $SOURCE_DIR/{gt.txt,wrong.txt} $TARGET_DIR
cp -r $SOURCE_DIR/logs $TARGET_DIR
