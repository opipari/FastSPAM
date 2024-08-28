EXPERIMENT_NAME="evaluate"
OUTPUT_DIR="results"

mkdir -p ${OUTPUT_DIR}/${EXPERIMENT_NAME}
# Save status of repository for reference
git log -1 --oneline > ${OUTPUT_DIR}/${EXPERIMENT_NAME}/repo_state.txt


curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
python get-pip.py

python -m pip install --upgrade pip

xargs -L 1 python -m pip install < ./requirements/temp_base.txt
python -m pip install tqdm


cd video_segmentation/datasets/MVPd
bash ./data/download.sh -s test -m -d imagesRGB.0000000000 -d panomasksRGB
# ./data/download.sh -s val -m -d imagesRGB.0000000000 -d panomasksRGB


cd ../../..


cd results
aws s3 cp s3://vesta-intern-anthony/video_panoptic_segmentation/models/Tube-Link/results/evaluate_trained_tube_link_r50.tar.gz ./ > /dev/null
tar -xf evaluate_trained_tube_link_r50.tar.gz
rm evaluate_trained_tube_link_r50.tar.gz
cd ..
echo "Downloaded pretrained models"

python video_segmentation/metrics/evaluate_STQ.py --rle_path ./results/results --ref_path ./video_segmentation/datasets/MVPd/MVPd --ref_split test --compute

rm -rf ./results/results/panomasksRLE

WORK_DIR="./results"

echo "Compressing results"
tar -cf ${EXPERIMENT_NAME}.tar.gz ${WORK_DIR}/
echo "Uploading results"
uploader=/opt/amazon/compute_grid_utils/output_uploader
$uploader ${EXPERIMENT_NAME}.tar.gz

echo "Finished"
