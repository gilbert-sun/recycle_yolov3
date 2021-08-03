FROM nvidia/cuda:10.2-cudnn7-devel-ubuntu18.04
MAINTAINER YuChing b0746021.stm06@nctu.edu.tw
# Sets utf-8 encoding for Python et al
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8
ENV TZ=Asia/Taipei
ENV DEBIAN_FRONTEND=noninteractive

RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone
RUN apt-get update && apt-get install python3-pip python3 tzdata git libopencv-dev wget vim -y
RUN dpkg-reconfigure --frontend noninteractive tzdata
ENV LC_TIME=en_US.UTF-8

# Turns off writing .pyc files; superfluous on an ephemeral container.
ENV PYTHONDONTWRITEBYTECODE=1
# Seems to speed things up
ENV PYTHONUNBUFFERED=1

# ################################################
# ###              Install OpenCV              ###
# ################################################
RUN git clone -b 3.4 --single-branch https://github.com/opencv/opencv.git
RUN apt-get install cmake -y
RUN cd opencv && mkdir build && cd build && cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/usr/local ..

# ################################################
# ###              Install OpenCV              ###
# ################################################
RUN python3 -m pip install --upgrade pip && python3 -m pip install numpy scikit-learn imagesize progressbar opencv-python

# # Always set a working directory
WORKDIR /home/shyechih/train_verify_yolo/

RUN echo 'alias python=python3' >> /root/.bashrc &&\
 echo 'PATH=/usr/local/cuda-10.2/bin:$PATH' >> /root/.bashrc &&\ 
 echo 'CUDA_HOME=/usr/local/cuda-10.2' >> /root/.bashrc &&\ 
 echo 'LD_LIBRARY_PATH=/usr/local/cuda-10.2/lib64:$LD_LIBRARY_PATH' >> /root/.bashrc

