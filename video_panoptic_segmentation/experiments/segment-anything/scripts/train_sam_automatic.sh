CONFIG_FILE="video_panoptic_segmentation/models/segment-anything/configs/train_automatic_sam_vit_b.py"


EXPERIMENT_NAME="$(python ${CONFIG_FILE} experiment_name)"
OUTPUT_DIR="$(python ${CONFIG_FILE} output_dir)"


echo "Setting up virtualenvironment"
# Setup virtualenvironment
python3.8 -m venv ./envs/segment-anything
source ./envs/segment-anything/bin/activate
pip install -r ./requirements/segment-anything.txt

echo "Downloading data"
cd video_panoptic_segmentation/datasets/MVPd
./data/download.sh -s train -p -d panomasksRGB -d imagesRGB.0000000000
./data/download.sh -s val -p -d panomasksRGB -d imagesRGB.0000000000
cd ../../..

# Download pre-trained model
echo "Downloading pretrained model"
aws s3 cp s3://vesta-intern-anthony/video_panoptic_segmentation/models/segment-anything/pretrained/sam_vit_b_01ec64.pth ./video_panoptic_segmentation/models/segment-anything/segment-anything/checkpoints/ > /dev/null

echo "Starting evaluation"
python video_panoptic_segmentation/models/segment-anything/train_automatic.py \
		--config-path ${CONFIG_FILE}

# Save status of repository for reference
git log -1 --oneline > ${OUTPUT_DIR}/${EXPERIMENT_NAME}/repo_state.txt

echo "The experiment script executed is:" > ${OUTPUT_DIR}/${EXPERIMENT_NAME}/script_state.txt
echo "basename: [$(basename "$0")]" >> ${OUTPUT_DIR}/${EXPERIMENT_NAME}/script_state.txt
echo "dirname : [$(dirname "$0")]" >> ${OUTPUT_DIR}/${EXPERIMENT_NAME}/script_state.txt
echo "pwd     : [$(pwd)]" >> ${OUTPUT_DIR}/${EXPERIMENT_NAME}/script_state.txt


echo "Compressing results"
tar -C ${OUTPUT_DIR} -cf ${EXPERIMENT_NAME}.tar.gz ${EXPERIMENT_NAME}/
echo "Uploading results"
uploader=/opt/amazon/compute_grid_utils/output_uploader
$uploader ${EXPERIMENT_NAME}.tar.gz

echo "Finished"