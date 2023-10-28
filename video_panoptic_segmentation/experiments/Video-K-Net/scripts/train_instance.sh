


pip install Cython==3.0.2
pip install -r ./requirements/video-k-net/video-k-net.txt


cd video_panoptic_segmentation/datasets/MVPd
./data/copy.sh -s train -m -d imagesRGB -d panomasksRGB
./data/copy.sh -s val -m -d imagesRGB -d panomasksRGB


cd ../../..

mkdir ./video_panoptic_segmentation/models/Video-K-Net/Video-K-Net/data/
ln -s $PWD/video_panoptic_segmentation/datasets/MVPd/MVPd ./video_panoptic_segmentation/models/Video-K-Net/Video-K-Net/data/

cd video_panoptic_segmentation/models/Video-K-Net/Video-K-Net


# ln -s $PWD/video_panoptic_segmentation/datasets/youtube_vis_2019 ./video_panoptic_segmentation/models/Video-K-Net/Video-K-Net/data/
# CONFIG="configs/video_knet_vis/video_knet_vis/knet_track_r50_1x_youtubevis.py"


CONFIG="configs/video_knet_vis/video_knet_vis/knet_track_r50_1x_mvpdvis.py"
WORK_DIR="./results"
LOAD_PATH="./pretrained/instance_models/knet_r50_instance_coco_3x_100p.pth"
GPUS="1"
PORT=${PORT:-$((29500 + $RANDOM % 29))}
PYTHONPATH=$PYTHONPATH:./ python -m torch.distributed.launch  --nproc_per_node=$GPUS --master_port=$PORT ./tools/train.py $CONFIG --launcher pytorch --work-dir=${WORK_DIR} --load-from ${LOAD_PATH}