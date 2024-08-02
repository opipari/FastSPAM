FROM pytorch/pytorch:2.3.0-cuda11.8-cudnn8-devel
ARG DEBIAN_FRONTEND=noninteractive


RUN apt update && apt install -y software-properties-common
RUN apt update && add-apt-repository ppa:deadsnakes/ppa
 
# base tools
RUN apt update && apt install -y \
    build-essential vim curl git wget zip \
    imagemagick 

RUN apt-get update && apt-get install -y python3-opencv

RUN pip install --upgrade pip
RUN python -m pip install https://github.com/open-mmlab/mmengine/archive/refs/tags/v0.8.5.zip
RUN TORCH_CUDA_ARCH_LIST="8.0 8.9" TORCH_NVCC_FLAGS="-Xfatbin -compress-all" CUDA_HOME=$(dirname $(dirname $(which nvcc))) LD_LIBRARY_PATH=$(dirname $(dirname $(which nvcc)))/lib MMCV_WITH_OPS=1 FORCE_CUDA=1 python -m pip install git+https://github.com/open-mmlab/mmcv.git@4f65f91db6502d990ce2ee5de0337441fb69dd10


