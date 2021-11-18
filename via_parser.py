#-*- coding: utf-8 -*-
import json
import os
import cv2
import imagesize
from shutil import copy2

def get_colors(number_color):
    pixel_hsv  = np.full((1,1,3), 255, dtype=np.uint8)
    pixels_bgr = np.empty((number_color,3), dtype=int)
    for i in range(number_color):
        pixel_hsv[0,0,0] = i*180/number_color
        pixels_bgr[i,:]  = cv2.cvtColor(pixel_hsv,cv2.COLOR_HSV2BGR)
    return pixels_bgr

def show(image0):
    height_image0, width_image0 = image0.shape[:2]
    SIZE_SHOW = 1000
    ratio = min( float(SIZE_SHOW)/height_image0 , float(SIZE_SHOW)/width_image0 )
    height_image = int(height_image0*ratio)
    width_image  = int( width_image0*ratio)
    image = cv2.resize(image0, (width_image,height_image))
    cv2.imshow('image', image)
    key = cv2.waitKey(0)
    if key == ord('q'):
        raise KeyboardInterrupt

def convert(dirname_via,
            dirname_yolo,
            jsonname):

    dirname_via_image  = os.path.join( dirname_via  , 'images' )
    dirname_yolo_label = os.path.join( dirname_yolo , 'labels' )
    dirname_yolo_image = os.path.join( dirname_yolo , 'images' )

    def _mkdir_with_check(dirname):
        if not os.path.exists(dirname):
            os.makedirs(dirname)
    _mkdir_with_check(dirname_yolo)
    _mkdir_with_check(dirname_yolo_label)
    _mkdir_with_check(dirname_yolo_image)

    with open(jsonname) as f:
        labels = json.load(f)
    for value in labels.values():
        imagename = os.path.join( dirname_via_image , value['filename'] )
        if DEBUG:
            print(imagename)
            image = cv2.imread(imagename)
        width_image, height_image = [ float(s)  for s in imagesize.get(imagename) ]

        copy2( imagename , dirname_yolo_image )

        labelname = os.path.join( dirname_yolo_label , os.path.basename(imagename.replace('jpg','txt')) )
        with open( labelname , 'w' ) as f:
            for region in value.get('regions'):
                names_category   = region.get('region_attributes').get('PET類別')
                shape_attributes = region.get('shape_attributes')
                left        = shape_attributes.get('x')
                top         = shape_attributes.get('y')
                height_bbox = shape_attributes.get('height')
                width_bbox  = shape_attributes.get('width')

                if top==None:
                    #print('%s with bounding box problem'%imagename)
                    #print(json.dumps(region, indent=2))
                    continue

                if names_category not in CATEGORIES:
                    #print('%s with category problem'%imagename)
                    #print(json.dumps(region, indent=2))
                    continue

                center_x = left + width_bbox/2.
                center_y = top + height_bbox/2.
                number_category = CATEGORIES.get(names_category)
                content = '%d %.6f %.6f %.6f %.6f'%(number_category,
                                                    center_x   / width_image,
                                                    center_y   /height_image,
                                                    width_bbox / width_image,
                                                    height_bbox/height_image)
                f.write(content+'\n')

                if DEBUG:
                    print(content)
                    right  = left + width_bbox
                    bottom = top + height_bbox
                    cv2.rectangle(image, (left,top), (right,bottom), COLORS[number_category], 5)

        if DEBUG:
            show(image)

if __name__ == '__main__':

    import numpy as np
    from glob import glob
    import traceback

    # settings
    DIRNAME_ROOT_VIA = '/home/e200/Documents/demo20200313/dataset/samples/raw_data/tag_02/'
    #'/home/e200/Documents/demo20200313/dataset/samples/raw_data/tag_02/20190718/'
    #'/media/e200/新增磁碟區1/data/stage1_enhanced-static/PET_pure-S_純醬油/'
    DIRNAME_ROOT_YOLO = '/media/e200/新增磁碟區1/data/stage1_enhanced-static/out_soy/'
    #'/home/e200/Documents/demo20200313/dataset/samples/for_yolo/tag_02/'
    #'/media/e200/新增磁碟區1/data/stage1_enhanced-static/out_soy/'
    DEBUG = 0
    CATEGORIES = {'透明'   : 0,
                  '油'     : 1,
                  '醬油'   : 2,
                  '非透明' : 3,
                  '其他'   : 4}

    for key,value in CATEGORIES.items():
        del CATEGORIES[key]
        CATEGORIES[key] = value

    COLORS = get_colors(len(CATEGORIES)+1)

    dirnames_via  = sorted(glob( DIRNAME_ROOT_VIA + '/*' ))
    for dirname_via in dirnames_via:
        try:
            dataname = os.path.basename(dirname_via)
            print(dataname)
            jsonnames = glob(os.path.join( dirname_via , 'labels/*.json' ))
            sizes = [ os.path.getsize(name)  for name in jsonnames ]
            jsonname_last = jsonnames[np.argmax(sizes)]
            dirname_yolo = os.path.join( DIRNAME_ROOT_YOLO , dataname )
            convert(dirname_via,
                    dirname_yolo,
                    jsonname_last)
        except:
            print(traceback.print_exc())
