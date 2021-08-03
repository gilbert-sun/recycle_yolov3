# Table of contents
* [General info](#general-info)
* [Setup](#setup)
    * [Enviroments](#enviroments)
        * [Python virtualenv](#python-virtualenv)
        * [Docker](#docker)
    * [Data pre-processing](#data-pre-processing)
    * [Train](#train)
        * [Output files and Logs](#output-files-and-logs)
        * [Test](#test)
        * [Validation](#validation)
    * [Store the Model](#store-the-model)
    * [Step-by-Step instruction](#step-by-tep-instruction)
* [Reference](#reference)

---

# General info
This is where Model Training, Testing and Validation.

---

# Setup
## Enviroments
You can use Python virtualenv or Docker.

## Python Virtualenv

```
python3.6 -m pip install virtualenv
python3.6 -m virtualenv -p /usr/bin/python3.6 py36
source py36/bin/activate
```
after activate the env, you can see `(py36)` in your prompt.

Install requirement packages
```
(py36) python3 -m pip install -r requirements.txt
```

## Docker
We will use **three share volume** in docker:
 - project **yolo_darknet** for compiling darknet(yolov3)
 - project **train_verify_yolo** for train/test/validation
 - `{data_path}`: where raw data located.

Note: git should be run **outside the docker.** (a.k.a. in your local)

Make sure you have these two projects [yolo_darknet](https://gitlab.com/shyechih.ai/shyechich.ai.internal/yolo_darknet) and [train_verify_yolo](https://gitlab.com/shyechih.shyechich.ai.internal/train_verify_yolo) in local.
First in train_verify_yolo directory, copy alphabet image files (under `/modulized/module/labels_futura-normal/`) to `{data_path}/labels/`
```
cp {origin_alphabet_images} {data_path}/labels/
// ex: cp train_verify_yolo/modulized/module/labels_futura-normal/* /home/petserver/Documents/shyechih/data/labels/
```
This `{data_path}/labels` is related to **step2** of [yolo_darknet - Posting Guide](https://gitlab.com/shyechih.ai/shyechich.ai.internal/yolo_darknet#posting-guide)

Build the image in the directory of train_verify_yolo
```
docker build --tag train_env:0.1  .
```
Start a container
```
docker run --net=host --gpus all -v $(pwd):/home/shyechih/train_verify_yolo/ \
-v /home/petserver/Documents/shyechih/yolo_darknet/:/home/shyechih/yolo_darknet/ \
-v /home/petserver/Documents/shyechih/data/:/home/shyechih/data/ \
-it train_env:0.1 /bin/bash
```
Once enter the container, build the darknet
```
cd /home/shyechih/yolo_darknet/
bash build_darknet.sh
```
when back to `train_verify_yolo/`, you will see a symbolic link of an excutable `./darknet`
```
cd /home/shyechih/train_verify_yolo/
ls -la darknet
darknet -> /home/shyechih/yolo_darknet/darknet
```

commit this container to a image
```
// leave container first
exit

// check container id
docker ps -a

// docker commit
docker commit -a "yuching" -m "train_env 0.2" {container_id} train_env:0.2
docker run --net=host --gpus all \
-v $(pwd):/home/shyechih/train_verify_yolo/ \
-v /home/petserver/Documents/shyechih/yolo_darknet/:/home/shyechih/yolo_darknet/ \
-v /home/petserver/Documents/shyechih/data/:/home/shyechih/data/ \
-it train_env:0.2 /bin/bash
```

Re-start then re-enter container
```
docker restart {container_id} && docker exec -it {container_id} bash
```

---

# Data pre-processing
All pre-processing procedures are in [data_rotate_augment](https://gitlab.com/shyechih.ai/shyechich.ai.internal/data_rotate_augment)

# Train 
```
python3 train.py
```
or 
Output to log, **naming should represent its intention**.
```
python3 &> logs/train_PET_1_train_36rotated.log
```
or
distach from current user/terminal
```
nohub python3 &> logs/train_PET_1_train_36rotated.log > /dev/null & disown
```
---

# Output files and logs:
- `yolov3_cfg/test.txt`
- `yolov3_cfg/train.txt`
- `yolov3_cfg/weights/yolov3_final.weights`
- `yolov3_cfg/weights/yolov3_{different_batches}.weights`
- `accurcy/acc_yolov3_final.txt`
- `predict/yolov3_final.txt`
- `predict/{predicted_data_for_each_images}.txt`
- `gt.txt`
- `wrong.txt`

**Note**: `test.py` and `performance.py` are included in `train.py`

---

### Test
```
python3 test.py
```
Result images will be generated to `predict/`


### Validation
```
python3 performance.py
```
Output: `accuracy/acc_yolov3_final.txt`

---

# Store the Model

Change `SOURCE_DIR` and `target_dir`, `target_dir` is **where you store all your models**
```
$ mkdir /media/shyechih/data/stage4_model/sc_PET_1_train_36rotated_20200629

$ vim cp_model.sh

SOURCE_DIR='/home/shyechih/Documents/antec1300/train_verify_yolo-master'
target_dir='/media/shyechih/data/stage4_model/sc_PET_1_train_36rotated_20200629'

cp -r $SOURCE_DIR/accuracy $target_dir
cp -r $SOURCE_DIR/predict $target_dir
cp -r $SOURCE_DIR/yolov3_cfg $target_dir
cp -r $SOURCE_DIR/accuracy $target_dir
cp -r $SOURCE_DIR/{gt.txt,wrong.txt} $target_dir
cp -r $SOURCE_DIR/logs $target_dir
```
Executing
```
bash scripts/cp_model.sh
```


---

## Step-by-Step instruction
#### - step1: At parameters.py , to modify path about image_dataset & darknet
    darknetEcec = "/home/xxx/darknet/darknet"
    imageYoloPath = "/home/xxx/Documents/django-upload-example/mysite/core/dataset/20190903_recycle_origin_tag_01_20190719_v00/images"
    labelYoloPath = "/home/xxx/Documents/django-upload-example/mysite/core/dataset/20190903_recycle_origin_tag_01_20190719_v00/labels"
    
#### - step2: At parameters.py , to modify PET(PE,PP,...) class from 5 to 7
    classList = {"P": 0, "O": 1, "S": 2, "C":3, "Ot":4, "T":5, "Ch":6,}
    
#### - step3: Testing darknet --> using webcam -c = 0 , 1, 2, ...
    ./darknet detector demo cfg/coco.data cfg/yolov3.cfg yolov3.weights -c 0(1,2,..)_

#### - step4: Training your weight file in darknet
    python3 train.py

    ./darknet detector train cfg/obj.data cfg/yolov3.cfg yolov3.conv.75 -gpus 0,1,2,3
    
    /home/petserver/darknet/darknet detector train /home/petserver/train_verify_yolo/yolov3_cfg/obj.data /home/petserver/train_verify_yolo/yolov3_cfg/yolov3.cfg yolov3.conv.75

##### note
    "./darknet detector train {1} {2} {3} {4} {5}-gpus {6}"
                {1}:obj.data,
                {2}:yolov3.cfg,
                {3}:default empty training weight (ex: download yolov3.conv.75 from darknet website),
                {4}: test folder in side file name list,
                {5}: all test folder file list predict result,
                {6}:How many gpu can be used at training (ex: 4 gpu [0,1,2,3] just using 1,3; no value default is 0)


#### - step5: Testing all picture in one folder
    python3 test.py

    ./darknet detector test_v2 cfg/obj.data  cfg/yolov3.cfg yolov3.weights  test.txt  yolov3_final.txt -out predict/ -thresh 0.1
    
    /home/petserver/darknet/./darknet detector test_v2 
    {1}./home/petserver/train_verify_yolo/yolov3_cfg/obj.data 
    {2}./home/petserver/train_verify_yolo/yolov3_cfg/yolov3.cfg 
    {3}./home/petserver/train_verify_yolo/yolov3_cfg/weights/yolov3_final.weights 
    {4}./home/petserver/train_verify_yolo/yolov3_cfg/test.txt 
    {5}./home/petserver/train_verify_yolo/predict/yolov3_final.txt
    

##### note
    "./darknet detector test_v2 {1} {2} {3} {4} {5} -out {6} -thresh {7} "
                {1}:obj.data,
                {2}:yolov3.cfg,
                {3}:your training weight (ex: yolov3.weight),
                {4}:testing all files collection, (ex:test.txt)
                {5}:predict all testing file result at one file, ( yolov3_final.txt)
                {6}:output file folder name
                {7}:threshold :default 0.5 , at pet testing ,set to be 0.1

#### - step6: Verify all predict result with ground truth 
    python3 performance.py
    
##### note
    Origin ./gt.txt(ground truth) and ./predict/yolov3_final.txt(predict result) 
    to do comparasion and final result will be at ./accuracy/acc_yolov3_final.txt

---

## Reference
### [Darknet](https://pjreddie.com/darknet/yolo/ )
Darknet is an open source neural network framework written in C and CUDA. It is fast, easy to install, and supports CPU and GPU computation.