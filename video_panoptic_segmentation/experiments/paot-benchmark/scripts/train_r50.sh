EXPERIMENT_NAME="train_r50"
OUTPUT_DIR="results"

echo "Setting up virtualenvironment"
# Setup virtualenvironment
python3.8 -m venv ./envs/paot-benchmark
source ./envs/paot-benchmark/bin/activate
pip install -r ./requirements/paot-benchmark/base.txt
pip install -r ./requirements/paot-benchmark/deps.txt

# Download pre-trained model
echo "Downloading pretrained model"
pip install --upgrade gdown
gdown 1IIkeBV6t4Iei3r4tJNKqu---gAHOCurn -O ./video_panoptic_segmentation/models/aot-benchmark/paot-benchmark/pretrain_models/

echo "Downloading data"
cd video_panoptic_segmentation/datasets/MVPd
./data/download.sh -s train -d panomasksRGB -d imagesRGB.0000000000
./data/download.sh -s val -d panomasksRGB -d imagesRGB.0000000000
cd ../../..

ln -s $PWD/video_panoptic_segmentation/datasets/MVPd/MVPd ./video_panoptic_segmentation/models/aot-benchmark/paot-benchmark/datasets/

nvidia-smi > ${OUTPUT_DIR}/${EXPERIMENT_NAME}/nvidia_smi.txt

echo "Starting training"
cd video_panoptic_segmentation/models/aot-benchmark/paot-benchmark

# Main training stage
exp="pano_r50"
pretrain="pretrain_models/R50_AOTv3_PRE.pth"
stage="PRE_YTB_DAV_MVPd"
python tools/train.py --amp \
        --exp_name ${exp} \
        --config ${exp} \
        --stage ${stage} \
        --pretrained_path ${pretrain} \
        --gpu_num 8

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