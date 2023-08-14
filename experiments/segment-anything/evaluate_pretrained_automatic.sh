
DATA_DIR="./video_panoptic_segmentation/datasets/MVPd/MVPd"
OUT_DIR="./video_panoptic_segmentation/m"

# Setup virtualenvironment
python3.8 -m venv ./envs/segment-anything
source ./envs/segment-anything/bin/activate
pip install -r ./requirements/segment-anything.txt


cd video_panoptic_segmentation/datasets/MVPd && \
./download.sh
cd ../../..


# Download pre-trained model
wget -P ./zero_shot_scene_segmentation/models/segment-anything/segment-anything/checkpoints/ https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth

# python video_panoptic_segmentation/segment-anything/evaluate_automatic.py -out 
# tar -cf 

# aws s3 cp 

echo "Ready to Evaluate"