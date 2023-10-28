


pip install Cython==3.0.2
pip install -r ./requirements/video-k-net/video-k-net.txt


cd video_panoptic_segmentation/datasets/MVPd
./data/copy.sh -s train -m -d imagesRGB -d panomasksRGB
./data/copy.sh -s val -m -d imagesRGB -d panomasksRGB


cd ../../..

mkdir ./video_panoptic_segmentation/models/Video-K-Net/Video-K-Net/data/
ln -s $PWD/video_panoptic_segmentation/datasets/MVPd/MVPd ./video_panoptic_segmentation/models/Video-K-Net/Video-K-Net/data/

cd video_panoptic_segmentation/models/Video-K-Net/Video-K-Net




CONFIG="configs/det/video_knet_mvpd/video_knet_s3_r50_rpn_mvpd_mask_embed_link_ffn_joint_train.py"
WORK_DIR="./results"
LOAD_PATH="./pretrained/panoptic_models/knet_coco_pan_r50.pth"
GPUS="1"
PORT=${PORT:-$((29500 + $RANDOM % 29))}
PYTHONPATH=$PYTHONPATH:./ python -m torch.distributed.launch  --nproc_per_node=$GPUS --master_port=$PORT ./tools/train.py $CONFIG --launcher pytorch --work-dir=${WORK_DIR} --load-from ${LOAD_PATH}