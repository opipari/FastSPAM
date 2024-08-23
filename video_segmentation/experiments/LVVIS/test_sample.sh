EXPERIMENT_NAME="train"
OUTPUT_DIR="video_segmentation/models/LVVIS/LVVIS/output"

mkdir -p ${OUTPUT_DIR}/${EXPERIMENT_NAME}
# Save status of repository for reference
git log -1 --oneline > ${OUTPUT_DIR}/${EXPERIMENT_NAME}/repo_state.txt


pip install --upgrade pip
pip install -r ./requirements/lvvis/lvvis.txt
# TORCH_CUDA_ARCH_LIST="8.0" TORCH_NVCC_FLAGS="-Xfatbin -compress-all" CUDA_HOME=$(dirname $(dirname $(which nvcc))) LD_LIBRARY_PATH=$(dirname $(dirname $(which nvcc)))/lib MMCV_WITH_OPS=1 FORCE_CUDA=1 python -m pip install git+https://github.com/open-mmlab/mmcv.git@4f65f91db6502d990ce2ee5de0337441fb69dd10

cd video_segmentation/models/LVVIS/LVVIS/ov2seg/modeling/pixel_decoder/ops
sh make.sh
cd /root/PanopticMemoryClouds


cd video_segmentation/datasets/MVPd
aws s3 cp s3://vesta-intern-anthony/video_panoptic_segmentation/datasets/MVPd/MVPd/MVPd_test_coco_sample.json ./MVPd/ > /dev/null
mv ./MVPd/MVPd_test_coco_sample.json ./MVPd/MVPd_test_coco.json
echo "Downloaded test annotations json"
bash ./data/download.sh -s test -m -d imagesRGB.0000000000 -d panomasksRGB
# ./data/download.sh -s val -m -d imagesRGB.0000000000 -d panomasksRGB


cd ../../..

ln -s $PWD/video_segmentation/datasets/MVPd/MVPd ./video_segmentation/models/LVVIS/LVVIS/

cd video_segmentation/models/LVVIS/LVVIS

mkdir models
cd models
aws s3 cp s3://vesta-intern-anthony/video_panoptic_segmentation/models/LLVIS/models/resnet50_miil_21k.pkl ./ > /dev/null
aws s3 cp s3://vesta-intern-anthony/video_panoptic_segmentation/models/LLVIS/models/swin_base_patch4_window7_224.pkl ./ > /dev/null
aws s3 cp s3://vesta-intern-anthony/video_panoptic_segmentation/models/LLVIS/models/model_final_4.pth ./ > /dev/null
cd ..
echo "Downloaded pretrained models"

./scripts/eval_video_mvpd.sh

WORK_DIR="./output"

echo "Compressing results"
tar -cf ${EXPERIMENT_NAME}.tar.gz ${WORK_DIR}/
echo "Uploading results"
uploader=/opt/amazon/compute_grid_utils/output_uploader
$uploader ${EXPERIMENT_NAME}.tar.gz

echo "Finished"
