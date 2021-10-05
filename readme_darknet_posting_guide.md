![Darknet Logo](http://pjreddie.com/media/files/darknet-black-small.png)

# Darknet #
Darknet is an open source neural network framework written in C and CUDA. It is fast, easy to install, and supports CPU and GPU computation.

For more information see the [Darknet project website](http://pjreddie.com/darknet).

For questions or issues please use the [Google Group](https://groups.google.com/forum/#!forum/darknet).

## Ref
    https://pjreddie.com/darknet/yolo/ 
     
## Posting Guide
#### [ ]. step1: modify Makefile as following
	GPU=1
    CUDNN=1
    OPENCV=1
    OPENMP=1
    DEBUG=0
    
    ARCH= #-gencode arch=compute_30,code=sm_30 \
          #-gencode arch=compute_35,code=sm_35 \
          #-gencode arch=compute_50,code=[sm_50,compute_50] \
          #-gencode arch=compute_52,code=[sm_52,compute_52] \
          -gencode arch=compute_61,code=[sm_61,compute_61] \
#### [ ]. step2: modify ./src/image.c to change path fitting your ubuntu environment
    image **load_alphabet_modify()
    {
        int i, j;
        const int nsize = 8;
        image **alphabets = calloc(nsize, sizeof(image));
        for(j = 0; j < nsize; ++j){
            alphabets[j] = calloc(128, sizeof(image));
            for(i = 32; i < 127; ++i){
                char buff[256];
                // modify this path to fit your environment, gilbert.start 2020.5
                sprintf(buff, "/home/petserver/darknet/data/labels/%d_%d.png", i, j);
                // gilbert.end 2020.5
                alphabets[j][i] = load_image_color(buff, 0, 0);
            }
        }
        return alphabets;
    }
#### [ ]. step3: modify ./example/detector.c to on/off debug message fitting your debug requirement
    void test_detector_with_images(char *datacfg, char *cfgfile, char *weightfile, char *filename, float thresh, float hier_thresh, char *outfile, int fullscreen, char *outfile_name)
    {
        FILE *in_file, *out_file;
        out_file = fopen(outfile_name, "w");
        fclose(out_file);
#### [ ]. step4: build darknet --> then get ./darknet
    make clean
    make -j8
#### [ ]. step5: Testing darknet --> showing one picture
    ./darknet detect cfg/yolov3.cfg yolov3.weights data/dog.jpg
#### [ ]. step6: Testing darknet --> using webcam -c = 0 , 1, 2, ...
    ./darknet detector demo cfg/coco.data cfg/yolov3.cfg yolov3.weights -c 0(1,2,..)_
    
#### [ ]. step7: Training your weight file in darknet
    ./darknet detector train cfg/obj.data cfg/yolov3.cfg yolov3.conv.75 -gpus 0,1,2,3

##### note
    "./darknet detector train {1} {2} {3} -gpus {4}"
                {1}:obj.data,
                {2}:yolov3.cfg,
                {3}:default empty training weight (ex: download yolov3.conv.75 from darknet website), 
                {4}:How many gpu can be used at training (ex: 4 gpu [0,1,2,3] just using 1,3)
    
    
#### [ ]. step8: Testing all picture in one folder 
    ./darknet detector test_v2 cfg/obj.data  cfg/yolov3.cfg yolov3.weights  test.txt  yolov3_final.txt -out predict/ -thresh 0.1

##### note
    "./darknet detector test_v2 {1} {2} {3} {4} {5} -out {6} -thresh {7} "
                {1}:obj.data,
                {2}:yolov3.cfg,
                {3}:your training weight (ex: yolov3.weight),
                {4}:testing all files collection, (ex:test.txt)
                {5}:predict all testing file result at one file, ( yolov3_final.txt)
                {6}:output file folder name
                {7}:threshold :default 0.5 , at pet testing ,set to be 0.1
            
