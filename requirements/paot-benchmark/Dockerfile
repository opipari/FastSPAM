FROM nvidia/cuda:11.7.1-cudnn8-devel-ubuntu20.04
ARG DEBIAN_FRONTEND=noninteractive


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

# install pytorch
#RUN python3.8 -m pip install \
#    torch==2.0.1+cu117 \
#    torchvision==0.15.2+cu117 --index-url https://download.pytorch.org/whl/cu117

# install pytorch3d
#RUN python3.8 -m pip install \
#    https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py38_cu117_pyt201/pytorch3d-0.7.4-cp38-cp38-linux_x86_64.whl
