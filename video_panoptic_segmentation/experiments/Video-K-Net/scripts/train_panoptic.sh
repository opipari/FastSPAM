EXPERIMENT_NAME="train_panoptic"
OUTPUT_DIR="video_panoptic_segmentation/models/Video-K-Net/Video-K-Net/results"

# Save status of repository for reference
git log -1 --oneline > ${OUTPUT_DIR}/${EXPERIMENT_NAME}/repo_state.txt



pip install Cython==3.0.2
pip install -r ./requirements/video-k-net/video-k-net.txt


cd video_panoptic_segmentation/datasets/MVPd
./data/download.sh -s train -m -d imagesRGB -d panomasksRGB
./data/download.sh -s val -m -d imagesRGB -d panomasksRGB


cd ../../..

mkdir ./video_panoptic_segmentation/models/Video-K-Net/Video-K-Net/data/
ln -s $PWD/video_panoptic_segmentation/datasets/MVPd/MVPd ./video_panoptic_segmentation/models/Video-K-Net/Video-K-Net/data/

cd video_panoptic_segmentation/models/Video-K-Net/Video-K-Net





mkdir pretrained
mkdir pretrained/panoptic_models
aws s3 cp s3://vesta-intern-anthony/video_panoptic_segmentation/models/Video-K-Net/pretrained/panoptic_models/knet_coco_pan_r50.pth ./pretrained/panoptic_models/ > /dev/null


CONFIG="configs/det/video_knet_mvpd/video_knet_s3_r50_rpn_mvpd_mask_embed_link_ffn_joint_train.py"
WORK_DIR="./results"
LOAD_PATH="./pretrained/panoptic_models/knet_coco_pan_r50.pth"
GPUS="8"
PORT=${PORT:-$((29500 + $RANDOM % 29))}
PYTHONPATH=$PYTHONPATH:./ python -m torch.distributed.launch  --nproc_per_node=$GPUS --master_port=$PORT ./tools/train.py $CONFIG --launcher pytorch --work-dir=${WORK_DIR} --load-from ${LOAD_PATH}



echo "Compressing results"
tar -cf ${EXPERIMENT_NAME}.tar.gz ${WORK_DIR}/
echo "Uploading results"
uploader=/opt/amazon/compute_grid_utils/output_uploader
$uploader ${EXPERIMENT_NAME}.tar.gz

echo "Finished"