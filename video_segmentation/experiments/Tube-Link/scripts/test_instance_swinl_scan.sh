EXPERIMENT_NAME="test_instance_swinl_scan"
OUTPUT_DIR="video_segmentation/models/Tube-Link/Tube-Link/results"

mkdir -p ${OUTPUT_DIR}/${EXPERIMENT_NAME}
# Save status of repository for reference
git log -1 --oneline > ${OUTPUT_DIR}/${EXPERIMENT_NAME}/repo_state.txt


pip install --upgrade pip
pip install Cython==3.0.2
pip install -r ./requirements/tube-link/tube-link.txt


# Move data onto device at video_segmentation/datasets/ScanNet/ScanNet


mkdir ./video_segmentation/models/Tube-Link/Tube-Link/data/
ln -s $PWD/video_segmentation/datasets/ScanNet/ScanNet ./video_segmentation/models/Tube-Link/Tube-Link/data/

cd video_segmentation/models/Tube-Link/Tube-Link


# Move trained model onto device at pretrained/iter_200000.pth


CONFIG="configs/video/scannet/scannet_swin_l_tb_link_2_5k_5k_10k_test.py"
WORK_DIR="./results"
CHECKPOINT="./pretrained/iter_200000.pth"
GPUS="1"
PORT=${PORT:-$((29500 + $RANDOM % 29))}
PYTHONPATH=$PYTHONPATH:./ python -m torch.distributed.launch  --nproc_per_node=$GPUS --master_port=$PORT ./tools/test_video.py $CONFIG $CHECKPOINT --launcher pytorch --work-dir=${WORK_DIR} 


echo "Finished"
