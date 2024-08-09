EXPERIMENT_NAME="train"
OUTPUT_DIR="video_segmentation/models/OMG-Seg/OMG-Seg/work_dirs"

mkdir -p ${OUTPUT_DIR}/${EXPERIMENT_NAME}
# Save status of repository for reference
git log -1 --oneline > ${OUTPUT_DIR}/${EXPERIMENT_NAME}/repo_state.txt

nvidia-smi > ${OUTPUT_DIR}/${EXPERIMENT_NAME}/nvidia.txt

pip install --upgrade pip
pip install -r ./requirements/omg-seg/omg-seg.txt
# TORCH_CUDA_ARCH_LIST="8.0" TORCH_NVCC_FLAGS="-Xfatbin -compress-all" CUDA_HOME=$(dirname $(dirname $(which nvcc))) LD_LIBRARY_PATH=$(dirname $(dirname $(which nvcc)))/lib MMCV_WITH_OPS=1 FORCE_CUDA=1 python -m pip install git+https://github.com/open-mmlab/mmcv.git@4f65f91db6502d990ce2ee5de0337441fb69dd10
pip install yapf==0.32

cd video_segmentation/datasets/MVPd
bash ./data/download_1.sh -s train -m -d imagesRGB.0000000000 -d panomasksRGB
# ./data/download.sh -s val -m -d imagesRGB.0000000000 -d panomasksRGB


cd ../../..

mkdir ./video_segmentation/models/OMG-Seg/OMG-Seg/data/
ln -s $PWD/video_segmentation/datasets/MVPd/MVPd ./video_segmentation/models/OMG-Seg/OMG-Seg/data/

cd video_segmentation/models/OMG-Seg/OMG-Seg


# ./tools/dist.sh gen_cls seg/configs/m2ov_val/eval_m2_convl_300q_ov_mvpd.py 1
# ./tools/dist.sh train seg/configs/m2ov_train/omg_convl_vlm_fix_12e_ov_mvpd.py 8
./tools/dist.sh train seg/configs/m2_train_close_set/omg_convl_mvpd_short.py 1

mv data/MVPd/train_annotations.json ./work_dirs/

WORK_DIR="./work_dirs"

echo "Compressing results"
tar -cf ${EXPERIMENT_NAME}.tar.gz ${WORK_DIR}/
echo "Uploading results"
uploader=/opt/amazon/compute_grid_utils/output_uploader
$uploader ${EXPERIMENT_NAME}.tar.gz

echo "Finished"
