FROM pytorch/pytorch:2.3.0-cuda11.8-cudnn8-devel
ARG DEBIAN_FRONTEND=noninteractive

#RUN rm /etc/apt/sources.list.d/cuda.list
#RUN rm /etc/apt/sources.list.d/nvidia-ml.list

RUN apt update && apt install -y software-properties-common
RUN apt update && add-apt-repository ppa:deadsnakes/ppa
 
# base tools
RUN apt update && apt install -y \
    build-essential vim curl git wget zip \
    imagemagick 
 
# python
RUN apt update && apt install -y python3.8 python3.8-venv python3.8-dev python3.8-distutils
# pip
RUN curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
RUN python3.8 get-pip.py
 
# python packages
RUN apt-get update && apt-get install -y python3-opencv
RUN python3.8 -m pip install opencv-python

