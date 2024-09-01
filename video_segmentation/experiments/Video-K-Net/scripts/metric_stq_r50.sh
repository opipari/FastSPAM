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
aws s3 cp s3://vesta-intern-anthony/video_panoptic_segmentation/models/Video-K-Net/results/vkn_r50_metrics.tar.gz ./ > /dev/null
tar -xf vkn_r50_metrics.tar.gz
rm vkn_r50_metrics.tar.gz
cd ..
echo "Downloaded pretrained models"

python video_segmentation/metrics/evaluate_STQ.py --rle_path ./results/train_instance_r50 --ref_path ./video_segmentation/datasets/MVPd/MVPd --ref_split test --compute

rm -rf ./results/train_instance_r50/panomasksRLE
mv res.json results/

WORK_DIR="./results"

echo "Compressing results"
tar -cf ${EXPERIMENT_NAME}.tar.gz ${WORK_DIR}/
echo "Uploading results"
uploader=/opt/amazon/compute_grid_utils/output_uploader
$uploader ${EXPERIMENT_NAME}.tar.gz

echo "Finished"
