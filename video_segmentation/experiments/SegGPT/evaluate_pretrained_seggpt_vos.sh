EXPERIMENT_NAME="evaluate_pretrained_seggpt_vos"
OUTPUT_DIR="video_segmentation/models/Painter/results"

# Save status of repository for reference
git log -1 --oneline > ${OUTPUT_DIR}/${EXPERIMENT_NAME}/repo_state.txt

echo "Setting up virtualenvironment"
# Setup virtualenvironment
python3.8 -m venv ./envs/seggpt
source ./envs/seggpt/bin/activate
xargs -L 1 pip install < requirements/seggpt.txt


# Download pre-trained model
echo "Downloading pretrained model"
cd video_segmentation/models/Painter/SegGPT/SegGPT_inference/
wget https://huggingface.co/BAAI/SegGPT/resolve/main/seggpt_vit_large.pth

echo "Starting evaluation"
python video_segmentation/models/Painter/SegGPT/SegGPT_inference/seggpt_inference_mvpd.py \
--ckpt_path video_segmentation/models/Painter/SegGPT/SegGPT_inference/seggpt_vit_large.pth \
--input_directory ./video_segmentation/datasets/MVPd/MVPd/test/ \
--output_dir ${OUTPUT_DIR}

echo "Finished"