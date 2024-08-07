EXPERIMENT_NAME="train"
OUTPUT_DIR="./work_dirs"


cd video_segmentation/datasets/MVPd
mkdir ./work_dirs


python3.8 -m pip install --upgrade pip
xargs -L 1 pip install < ./requirements/mvpd.txt

./data/download_4.sh -s train -m -d imagesRGB.0000000000 -d panomasksRGB
# ./data/download.sh -s val -m -d imagesRGB.0000000000 -d panomasksRGB


python3.8 convert2detectron.py --root_path MVPd --split train

mv MVPd_train_detectron2.json ./work_dirs/


echo "Compressing results"
tar -cf ${EXPERIMENT_NAME}.tar.gz ${OUTPUT_DIR}/
echo "Uploading results"
uploader=/opt/amazon/compute_grid_utils/output_uploader
$uploader ${EXPERIMENT_NAME}.tar.gz

echo "Finished"
