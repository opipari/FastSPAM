
https://github.com/open-mmlab/mmcv/archive/refs/tags/v1.3.16.zip
https://github.com/open-mmlab/mmdetection/archive/refs/tags/v2.18.0.zip

mim install mmcv-full==1.3.16

echo "Setting up virtualenvironment"
# Setup virtualenvironment
python3.8 -m venv ./envs/video-k-net && \
source ./envs/video-k-net/bin/activate && \
pip install -r ./requirements/video-k-net/base.txt && \
pip install -r ./requirements/video-k-net/deps.txt


mkdir ./video_panoptic_segmentation/models/Video-K-Net/Video-K-Net/data/
ln -s $PWD/video_panoptic_segmentation/datasets/MVPd/MVPd ./video_panoptic_segmentation/models/Video-K-Net/Video-K-Net/data/

cd video_panoptic_segmentation/models/Video-K-Net/Video-K-Net

wget https://github.com/open-mmlab/mmcv/archive/refs/tags/v1.3.16.zip
unzip v1.3.16.zip
cd mmcv-1.3.16/


CONFIG="configs/det/video_knet_mvpd/video_knet_s3_r50_rpn_mvpd_mask_embed_link_ffn_joint_train.py"
WORK_DIR="./results"
LOAD_PATH="./pretrain_models/knet_coco_pan_r50.pth"
GPUS="1"
PORT=${PORT:-$((29500 + $RANDOM % 29))}
PYTHONPATH=$PYTHONPATH:./ python -m torch.distributed.launch  --nproc_per_node=$GPUS --master_port=$PORT ./tools/train.py $CONFIG --launcher pytorch --work-dir=${WORK_DIR} --load-from ${LOAD_PATH}