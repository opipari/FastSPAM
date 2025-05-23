EXPERIMENT_NAME="evaluate_pretrained_sam_reprojection_delta_proportional"
OUTPUT_DIR="video_panoptic_segmentation/models/segment-anything/results"

# echo "Setting up virtualenvironment"
# # Setup virtualenvironment
# python3.8 -m venv ./envs/segment-anything
# source ./envs/segment-anything/bin/activate
# pip install -r ./requirements/segment-anything.txt

# echo "Downloading data"
# cd video_panoptic_segmentation/datasets/MVPd
# ./data/download.sh -s val
# cd ../../..

# # Download pre-trained model
# echo "Downloading pretrained model"
# aws s3 cp s3://vesta-intern-anthony/video_panoptic_segmentation/models/segment-anything/pretrained/sam_vit_h_4b8939.pth ./video_panoptic_segmentation/models/segment-anything/segment-anything/checkpoints/ > /dev/null
# # wget -P ./video_panoptic_segmentation/models/segment-anything/segment-anything/checkpoints/ https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth

echo "Starting evaluation"
python video_panoptic_segmentation/models/segment-anything/evaluate_reprojection.py \
		--config-path ./video_panoptic_segmentation/experiments/segment-anything/configs/${EXPERIMENT_NAME}.json \
		--output-path ${OUTPUT_DIR}

# # Save status of repository for reference
# git log -1 --oneline > ${OUTPUT_DIR}/${EXPERIMENT_NAME}/repo_state.txt

# echo "Compressing results"
# tar -C ${OUTPUT_DIR} -cf ${EXPERIMENT_NAME}.tar.gz ${EXPERIMENT_NAME}/
# echo "Uploading results"
# aws s3 cp ${EXPERIMENT_NAME}.tar.gz s3://vesta-intern-anthony/${OUTPUT_DIR}/ > /dev/null

echo "Finished"