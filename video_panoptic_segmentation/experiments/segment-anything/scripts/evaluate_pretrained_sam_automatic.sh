EXPERIMENT_NAME="evaluate_pretrained_sam_automatic"
OUTPUT_DIR="./video_panoptic_segmentation/results/segment-anything/"

# Download pre-trained model
/opt/amazon/bin/aws s3 cp s3://prism-intern-anthony/models/segment-anything/pretrained/sam_vit_h_4b8939.pth ./video_panoptic_segmentation/models/segment-anything/segment-anything/checkpoints/

echo "Downloading data"
cd video_panoptic_segmentation/datasets/MVPd
./data/download.sh -s val
cd ../../..

echo "Setting up virtualenvironment"
# Setup virtualenvironment
python3.8 -m venv ./envs/segment-anything
source ./envs/segment-anything/bin/activate
pip install -r ./requirements/segment-anything.txt




# wget -P ./video_panoptic_segmentation/models/segment-anything/segment-anything/checkpoints/ https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth

echo "Starting evaluation"
python video_panoptic_segmentation/models/segment-anything/evaluate.py --config-path ./video_panoptic_segmentation/experiments/segment-anything/configs/evaluate_pretrained_sam_automatic.json --output-path ${OUTPUT_DIR}


echo "Compressing results"
tar -C ${OUTPUT_DIR} -cf ${EXPERIMENT_NAME}.tar.gz ${EXPERIMENT_NAME}/
aws s3 cp ${EXPERIMENT_NAME}.tar.gz s3://prism-intern-anthony/results/segment-anything/

echo "Finished"