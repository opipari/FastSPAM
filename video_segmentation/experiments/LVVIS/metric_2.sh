EXPERIMENT_NAME="evaluate"
OUTPUT_DIR="results"

mkdir -p ${OUTPUT_DIR}/${EXPERIMENT_NAME}
# Save status of repository for reference
git log -1 --oneline > ${OUTPUT_DIR}/${EXPERIMENT_NAME}/repo_state.txt


pip install --upgrade pip
pip install -r ./requirements/temp_base.txt


cd video_segmentation/datasets/MVPd
bash ./data/download_2.sh -s test -m -d imagesRGB.0000000000 -d panomasksRGB
# ./data/download.sh -s val -m -d imagesRGB.0000000000 -d panomasksRGB


cd ../../..


cd results
aws s3 cp s3://vesta-intern-anthony/video_panoptic_segmentation/models/LLVIS/results/ov2seg_resnet50.tar.gz ./ > /dev/null
tar -xvf ov2seg_resnet50.tar.gz
rm ov2seg_resnet50.tar.gz
cd ..
echo "Downloaded pretrained models"

python video_segmentation/metrics/evaluate_iVPQ.py --rle_path ./results/ov2seg_resnet50 --ref_path ./video_segmentation/datasets/MVPd/MVPd --ref_split test --compute --vkn


WORK_DIR="./results"

echo "Compressing results"
tar -cf ${EXPERIMENT_NAME}.tar.gz ${WORK_DIR}/
echo "Uploading results"
uploader=/opt/amazon/compute_grid_utils/output_uploader
$uploader ${EXPERIMENT_NAME}.tar.gz

echo "Finished"
