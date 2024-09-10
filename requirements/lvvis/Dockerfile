FROM pytorch/pytorch:2.3.0-cuda11.8-cudnn8-devel
ARG DEBIAN_FRONTEND=noninteractive

RUN rm /etc/apt/sources.list.d/cuda.list
RUN rm /etc/apt/sources.list.d/nvidia-ml.list

RUN apt update && apt install -y software-properties-common
RUN apt update && add-apt-repository ppa:deadsnakes/ppa
 
# base tools
RUN apt update && apt install -y \
    build-essential vim curl git wget zip \
    imagemagick 

RUN apt update && apt install -y python3-opencv

RUN pip install --upgrade pip

