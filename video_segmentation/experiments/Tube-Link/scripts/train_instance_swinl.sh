EXPERIMENT_NAME="train_instance_swinl"
OUTPUT_DIR="video_segmentation/models/Tube-Link/Tube-Link/results"

mkdir -p ${OUTPUT_DIR}/${EXPERIMENT_NAME}
# Save status of repository for reference
git log -1 --oneline > ${OUTPUT_DIR}/${EXPERIMENT_NAME}/repo_state.txt


pip install --upgrade pip
pip install Cython==3.0.2
pip install -r ./requirements/tube-link/tube-link.txt


cd video_segmentation/datasets/MVPd
./data/download.sh -s train -m -d imagesRGB.0000000000 -d panomasksRGB
# ./data/download.sh -s val -m -d imagesRGB.0000000000 -d panomasksRGB


cd ../../..

mkdir ./video_segmentation/models/Tube-Link/Tube-Link/data/
ln -s $PWD/video_segmentation/datasets/MVPd/MVPd ./video_segmentation/models/Tube-Link/Tube-Link/data/

cd video_segmentation/models/Tube-Link/Tube-Link

mkdir pretrained
aws s3 cp s3://vesta-intern-anthony/video_panoptic_segmentation/models/Tube-Link/swin/pretrained/iter_100000.pth ./pretrained/ > /dev/null



CONFIG="configs/video/mvpd/mvpd_swin_l_tb_link_2_5k_5k_10k.py"
WORK_DIR="./results"
LOAD_PATH="./pretrained/iter_100000.pth"
GPUS="8"
PORT=${PORT:-$((29500 + $RANDOM % 29))}
PYTHONPATH=$PYTHONPATH:./ python -m torch.distributed.launch  --nproc_per_node=$GPUS --master_port=$PORT ./tools/train.py $CONFIG --launcher pytorch --work-dir=${WORK_DIR} --resume-from=${LOAD_PATH} --no-validate


echo "Compressing results"
tar -cf ${EXPERIMENT_NAME}.tar.gz ${WORK_DIR}/
echo "Uploading results"
uploader=/opt/amazon/compute_grid_utils/output_uploader
$uploader ${EXPERIMENT_NAME}.tar.gz

echo "Finished"
