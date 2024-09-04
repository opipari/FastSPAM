EXPERIMENT_NAME="test_instance_swinb"
OUTPUT_DIR="video_segmentation/models/Video-K-Net/Video-K-Net/results"

# Save status of repository for reference
git log -1 --oneline > ${OUTPUT_DIR}/${EXPERIMENT_NAME}/repo_state.txt

nvidia-smi > ${OUTPUT_DIR}/${EXPERIMENT_NAME}/nvidia_smi.txt

nvidia-smi

pip install --upgrade pip
pip install Cython==3.0.2
pip install natsort
pip install -r ./requirements/video-k-net/video-k-net.txt


cd video_segmentation/datasets/MVPd
# ./data/download.sh -s train -m -d imagesRGB.0000000000 -d panomasksRGB
# ./data/download.sh -s val -m -d imagesRGB.0000000000 -d panomasksRGB
./data/download.sh -s test -m -d imagesRGB.0000000000 -d panomasksRGB

cd ../../..

mkdir ./video_segmentation/models/Video-K-Net/Video-K-Net/data/
ln -s $PWD/video_segmentation/datasets/MVPd/MVPd ./video_segmentation/models/Video-K-Net/Video-K-Net/data/

cd video_segmentation/models/Video-K-Net/Video-K-Net


# ln -s $PWD/video_panoptic_segmentation/datasets/youtube_vis_2019 ./video_panoptic_segmentation/models/Video-K-Net/Video-K-Net/data/
# CONFIG="configs/video_knet_vis/video_knet_vis/knet_track_r50_1x_youtubevis.py"

mkdir results
aws s3 cp s3://vesta-intern-anthony/video_panoptic_segmentation/models/Video-K-Net/trained/instance_models/swin/iter_200000.pth ./results/ > /dev/null


CONFIG="configs/video_knet_vis/video_knet_vis/knet_track_swinb_deformable_1x_mvpdvis.py"
WORK_DIR="./results"
CHECKPOINT="./results/iter_200000.pth"
GPUS="8"
PORT=${PORT:-$((29500 + $RANDOM % 29))}
PYTHONPATH=$PYTHONPATH:./ python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT ./tools_vis/test_whole_video.py $CONFIG $CHECKPOINT --launcher pytorch --work-dir=${WORK_DIR} --format-only

echo "Compressing results"
tar -cf ${EXPERIMENT_NAME}.tar.gz ${WORK_DIR}/
echo "Uploading results"
uploader=/opt/amazon/compute_grid_utils/output_uploader
$uploader ${EXPERIMENT_NAME}.tar.gz

echo "Finished"
