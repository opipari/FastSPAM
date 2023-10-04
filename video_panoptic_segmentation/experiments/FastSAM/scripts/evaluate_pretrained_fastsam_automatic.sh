EXPERIMENT_NAME="evaluate_pretrained_fastsam_automatic"
OUTPUT_DIR="video_panoptic_segmentation/models/FastSAM/results"


YOLO_VERBOSE="false"

# Save status of repository for reference
# git log -1 --oneline > ${OUTPUT_DIR}/${EXPERIMENT_NAME}/repo_state.txt

# echo "Setting up virtualenvironment"
# # Setup virtualenvironment
# python3.8 -m venv ./envs/FastSAM
# source ./envs/FastSAM/bin/activate
# pip install -r ./requirements/FastSAM.txt

# echo "Downloading data"
# cd video_panoptic_segmentation/datasets/MVPd
# ./data/download.sh -s val
# cd ../../..

# # Download pre-trained model
# echo "Downloading pretrained model"
# aws s3 cp s3://vesta-intern-anthony/video_panoptic_segmentation/models/segment-anything/pretrained/sam_vit_b_01ec64.pth ./video_panoptic_segmentation/models/segment-anything/segment-anything/checkpoints/ > /dev/null

echo "Starting evaluation"
python video_panoptic_segmentation/models/FastSAM/evaluate_automatic.py \
		--config-path ./video_panoptic_segmentation/experiments/FastSAM/configs/${EXPERIMENT_NAME}.json \
		--output-path ${OUTPUT_DIR}


# echo "Compressing results"
# tar -C ${OUTPUT_DIR} -cf ${EXPERIMENT_NAME}.tar.gz ${EXPERIMENT_NAME}/
# echo "Uploading results"
# uploader=/opt/amazon/compute_grid_utils/output_uploader
# $uploader ${EXPERIMENT_NAME}.tar.gz

# echo "Finished"