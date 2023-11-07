EXPERIMENT_NAME="train_instance_swinb_test"
OUTPUT_DIR="video_panoptic_segmentation/models/Video-K-Net/Video-K-Net/results"

# Save status of repository for reference
git log -1 --oneline > ${OUTPUT_DIR}/${EXPERIMENT_NAME}/repo_state.txt


pip install --upgrade pip
pip install Cython==3.0.2
pip install -r ./requirements/video-k-net/video-k-net.txt


cd video_panoptic_segmentation/datasets/MVPd
./data/download.sh -s train -p -m -d imagesRGB -d panomasksRGB
# ./data/download.sh -s val -m -d imagesRGB -d panomasksRGB


cd ../../..

mkdir ./video_panoptic_segmentation/models/Video-K-Net/Video-K-Net/data/
ln -s $PWD/video_panoptic_segmentation/datasets/MVPd/MVPd ./video_panoptic_segmentation/models/Video-K-Net/Video-K-Net/data/

cd video_panoptic_segmentation/models/Video-K-Net/Video-K-Net


# ln -s $PWD/video_panoptic_segmentation/datasets/youtube_vis_2019 ./video_panoptic_segmentation/models/Video-K-Net/Video-K-Net/data/
# CONFIG="configs/video_knet_vis/video_knet_vis/knet_track_r50_1x_youtubevis.py"

mkdir pretrained
mkdir pretrained/instance_models
aws s3 cp s3://vesta-intern-anthony/video_panoptic_segmentation/models/Video-K-Net/pretrained/instance_models/knet_deformable_fpn_swin_b_coco.pth ./pretrained/instance_models/ > /dev/null


CONFIG="configs/video_knet_vis/video_knet_vis/knet_track_swinb_deformable_1x_mvpdvis.py"
WORK_DIR="./results"
LOAD_PATH="./pretrained/instance_models/knet_deformable_fpn_swin_b_coco.pth"
GPUS="8"
PORT=${PORT:-$((29500 + $RANDOM % 29))}
PYTHONPATH=$PYTHONPATH:./ python -m torch.distributed.launch  --nproc_per_node=$GPUS --master_port=$PORT ./tools/train.py $CONFIG --launcher pytorch --work-dir=${WORK_DIR} --load-from ${LOAD_PATH} --no-validate


echo "Compressing results"
tar -cf ${EXPERIMENT_NAME}.tar.gz ${WORK_DIR}/
echo "Uploading results"
uploader=/opt/amazon/compute_grid_utils/output_uploader
$uploader ${EXPERIMENT_NAME}.tar.gz

echo "Finished"