EXPERIMENT_NAME="train"
OUTPUT_DIR="output"

mkdir -p ${OUTPUT_DIR}/${EXPERIMENT_NAME}
# Save status of repository for reference
git log -1 --oneline > ${OUTPUT_DIR}/${EXPERIMENT_NAME}/repo_state.txt

curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
python get-pip.py

python -m pip install --upgrade pip

cd video_segmentation/datasets/MVPd
xargs -L 1 python -m pip install < ./requirements/mvpd.txt
python -m pip install tqdm

bash ./data/download_1.sh -s test -m -d imagesRGB.0000000000 -d panomasksRGB


python convert2coco.py --root_path MVPd --split test

mv MVPd_test_coco.json output

WORK_DIR="./output"

echo "Compressing results"
tar -cf ${EXPERIMENT_NAME}.tar.gz ${WORK_DIR}/
echo "Uploading results"
uploader=/opt/amazon/compute_grid_utils/output_uploader
$uploader ${EXPERIMENT_NAME}.tar.gz

echo "Finished"
