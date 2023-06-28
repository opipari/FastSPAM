# ZeroShotSceneSegmentation

## Repository Structure

This repository is setup to contain a centralized python package, `ZeroShotSceneSegmentation`, that stores all code for data simulation, data processing, model training, and model evaluation. Correspondingly, the code for these core functions is spread across sub-directories.

#### Directory Structure

```
.                                   # Root project directory
├── README.md                       # This readme file
├── setup.py                        # Setup script using setuptools to install the ZeroShotSceneSegmentation package
├── envs                            # Directory containing virtual environment source files and package dependency requirements
│   ├── requirements                # Directory containing the specification of package dependency requirements
│   │   └── segment-anything.txt    # Package requirements for segment anything model package
│   └── segment-anything            # Directory for segment anything's virtual environment
└── zero_shot_scene_segmentation    # The central python package for ZeroShotSceneSegmentation package
    ├── models                      # Directory containing source code for model development, training and evaluation
    │   └── segment-anything        # Directory containing all source for training and evaluating segment anything model
    ├── datasets                    # Directory containing source code for dataset processing as well as PyTorch Dataset classes
    └── simulators                  # Directory containing source code for data simulationa and rendering
        └── blender                 # Directory containing source code for blender-based rendering
```


## Setup

1. Clone repository and submodules

    ```
    git clone --recurse-submodules git@github.com:opipari/ZeroShotSceneSegmentation.git && \
    cd ZeroShotSceneSegmentation
    ```

2. Install Python v3.8

    ```
    sudo apt install python3.8 python3.8-venv python3.8-dev
    ```

3. Install Python v3.9

    ```
    sudo apt install python3.9 python3.9-venv python3.9-dev
    ```


<hr>

### Datasets

#### Generic Data Processing Environment

This environment is to be used for dataset preprocessing since Blender's version of python3.10 is not compatible with needed cocotools api.

```
python3.8 -m venv ./envs/data-processing && \
  source ./envs/data-processing/bin/activate && \
    pip install -r ./envs/requirements/data-processing.txt && \
      deactivate
```

#### [Habitat Matterport 3D Semantic Dataset](https://aihabitat.org/datasets/hm3d-semantics/)

1. Create an account on [Matterport website](https://buy.matterport.com/free-account-register?_ga=2.183460966.1764739312.1687379653-577208820.1687379653)
2. Request access to the dataset by accepting [the terms and conditions](https://matterport.com/matterport-end-user-license-agreement-academic-use-model-data)
3. Download the [v0.2 dataset files](https://github.com/matterport/habitat-matterport-3dresearch#-downloading-hm3d-v02)
4. Extract dataset files into the following directory structure
   - Recommended location for extraction: `./zero_shot_scene_segmentation/datasets/HM3D/`
    ```
    .                                                             # Root directory of dataset (Recommended to be ./zero_shot_scene_segmentation/datasets/HM3D/)
    ├── hm3d_annotated_basis.scene_dataset_config.json            # JSON files describing scene information per-subset of the overall dataset
    ├── hm3d_annotated_example_basis.scene_dataset_config.json    # The dataset is broken down into a collection of scans (i.e. 'scenes'), each at building scale
    ├── hm3d_annotated_minival_basis.scene_dataset_config.json
    ├── hm3d_annotated_train_basis.scene_dataset_config.json
    ├── hm3d_annotated_val_basis.scene_dataset_config.json
    ├── example                                                   # Directory containing a single scene directory (00337, 00770, and 00861)
    ├── minival                                                   # Directory containing 10 scene directories subsampled from overall val set
    ├── train                                                     # Directory containing most scenes in the dataset
    └── val                                                       # Directory containing scenes to be used for model validation (chosen by dataset authors)
    ```

#### Static Images

  - [MSRA10K](https://mmcheng.net/msra10k/)
      ```
      aws s3 cp s3://prism-intern-anthony/raw_data/pretraining/static/MSRA10K/MSRA10K_Imgs_GT.zip ./zero_shot_scene_segmentation/datasets/raw_data/pretraining/static/MSRA10K/ && \
        unzip ./zero_shot_scene_segmentation/datasets/raw_data/pretraining/static/MSRA10K/MSRA10K_Imgs_GT.zip -d ./zero_shot_scene_segmentation/datasets/raw_data/pretraining/static/MSRA10K && \
          rm ./zero_shot_scene_segmentation/datasets/raw_data/pretraining/static/MSRA10K/MSRA10K_Imgs_GT.zip
      ```
  - [ECSSD](http://www.cse.cuhk.edu.hk/leojia/projects/hsaliency/dataset.html)
      ```
      aws s3 sync s3://prism-intern-anthony/raw_data/pretraining/static/ECSSD/ ./zero_shot_scene_segmentation/datasets/raw_data/pretraining/static/ECSSD/ && \
        unzip ./zero_shot_scene_segmentation/datasets/raw_data/pretraining/static/ECSSD/images.zip -d ./zero_shot_scene_segmentation/datasets/raw_data/pretraining/static/ECSSD/ && \
          rm ./zero_shot_scene_segmentation/datasets/raw_data/pretraining/static/ECSSD/images.zip
        unzip ./zero_shot_scene_segmentation/datasets/raw_data/pretraining/static/ECSSD/ground_truth_mask.zip -d ./zero_shot_scene_segmentation/datasets/raw_data/pretraining/static/ECSSD/ && \
          rm ./zero_shot_scene_segmentation/datasets/raw_data/pretraining/static/ECSSD/ground_truth_mask.zip
      ```
  - [PASCAL-S](http://cbs.ic.gatech.edu/salobj/download/salObj.zip)
      ```

      ```
  - [PASCAL VOC2012](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/)
      ```

      ```
  - [COCO 2017](https://cocodataset.org/#download)
    See here for confirmation of 2017: https://github.com/xmlyqing00/AFB-URR/issues/15


#### DAVIS

<details open>
<summary>Download From AWS Cloud</summary>
    
```
aws s3 cp s3://prism-intern-anthony/raw_data/pretraining/DAVIS-2017-trainval-480p.zip ./zero_shot_scene_segmentation/datasets/raw_data/ && \
  unzip ./zero_shot_scene_segmentation/datasets/raw_data/DAVIS-2017-trainval-480p.zip -d ./zero_shot_scene_segmentation/datasets/raw_data/ && \
    rm ./zero_shot_scene_segmentation/datasets/raw_data/DAVIS-2017-trainval-480p.zip
```

</details>


<details>
<summary> Download From Open Internet</summary>
    
```
wget https://data.vision.ee.ethz.ch/csergi/share/davis/DAVIS-2017-trainval-480p.zip -P ./zero_shot_scene_segmentation/datasets/raw_data/ && \
  unzip ./zero_shot_scene_segmentation/datasets/raw_data/DAVIS-2017-trainval-480p.zip -d ./zero_shot_scene_segmentation/datasets/raw_data/ && \
    rm ./zero_shot_scene_segmentation/datasets/raw_data/DAVIS-2017-trainval-480p.zip
```
</details>


#### YouTube-VOS

<details open>
<summary>Download From AWS Cloud</summary>
    
```
aws s3 cp s3://prism-intern-anthony/raw_data/DAVIS-2017-trainval-480p.zip ./zero_shot_scene_segmentation/datasets/raw_data/ && \
  unzip ./zero_shot_scene_segmentation/datasets/raw_data/DAVIS-2017-trainval-480p.zip -d ./zero_shot_scene_segmentation/datasets/raw_data/ && \
    rm ./zero_shot_scene_segmentation/datasets/raw_data/DAVIS-2017-trainval-480p.zip
```

</details>


<details>
<summary> Download From Open Internet</summary>

Follow download instructions from [aot-benchmark](https://github.com/yoxu515/aot-benchmark#getting-started)

Link for 2018: https://drive.google.com/drive/folders/1bI5J1H3mxsIGo7Kp-pPZU8i6rnykOw7f

Link for 2019: https://drive.google.com/drive/folders/1BWzrCWyPEmBEKm0lOHe5KLuBuQxUSwqz

</details>




#### [VIPOSeg-Benchmark](https://aihabitat.org/datasets/hm3d-semantics/)


<details open>
<summary>Download From AWS Cloud</summary>
    
```
export UNZIP_DISABLE_ZIPBOMB_DETECTION=TRUE && \
  aws s3 sync s3://prism-intern-anthony/raw_data/VIPOSeg/ ./zero_shot_scene_segmentation/datasets/raw_data/VIPOSeg/ && \
    unzip ./zero_shot_scene_segmentation/datasets/raw_data/VIPOSeg/VIPOSeg_train.zip -d ./zero_shot_scene_segmentation/datasets/raw_data/VIPOSeg/ && \
      rm ./zero_shot_scene_segmentation/datasets/raw_data/VIPOSeg/VIPOSeg_train.zip && \
    unzip ./zero_shot_scene_segmentation/datasets/raw_data/VIPOSeg/VIPOSeg_valid.zip -d ./zero_shot_scene_segmentation/datasets/raw_data/VIPOSeg/ && \
      rm ./zero_shot_scene_segmentation/datasets/raw_data/VIPOSeg/VIPOSeg_valid.zip && \
unset UNZIP_DISABLE_ZIPBOMB_DETECTION
```

</details>


<details>
<summary> Download From Open Internet</summary>
    
```
export UNZIP_DISABLE_ZIPBOMB_DETECTION=TRUE && \
  source ./envs/data-processing/bin/activate && \
    gdown 1GdhTyV8s6wJi8HnlncBWoI2gb_CmrbS1 -O ./zero_shot_scene_segmentation/datasets/raw_data/VIPOSeg/ && \
      unzip ./zero_shot_scene_segmentation/datasets/raw_data/VIPOSeg/VIPOSeg_train.zip -d ./zero_shot_scene_segmentation/datasets/raw_data/VIPOSeg/ && \
        rm ./zero_shot_scene_segmentation/datasets/raw_data/VIPOSeg/VIPOSeg_train.zip && \
    gdown 1E6cB6FqXhLKT6N5_NEXO7QckwH45IWU2 -O zero_shot_scene_segmentation/datasets/raw_data/VIPOSeg/ && \
      unzip ./zero_shot_scene_segmentation/datasets/raw_data/VIPOSeg/VIPOSeg_valid.zip -d ./zero_shot_scene_segmentation/datasets/raw_data/VIPOSeg/ && \
        rm ./zero_shot_scene_segmentation/datasets/raw_data/VIPOSeg/VIPOSeg_valid.zip && \
  deactivate && \
unset UNZIP_DISABLE_ZIPBOMB_DETECTION
```

</details>




<hr>

### Models

#### Segment Anything Model (SAM)

```
python3.8 -m venv ./envs/segment-anything && \
  source ./envs/segment-anything/bin/activate && \
    pip install -r ./envs/requirements/segment-anything.txt && \
      deactivate && \
        wget -P ./zero_shot_scene_segmentation/models/segment-anything/segment-anything/checkpoints/ https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
```

#### Pyramid / Panoptic Associating Objects with Transformers (PAOT)

```
python3.8 -m venv ./envs/paot-benchmark && \
  source ./envs/paot-benchmark/bin/activate && \
    pip install -r ./envs/requirements/paot-benchmark/base.txt &&
    pip install -r ./envs/requirements/paot-benchmark/deps.txt && \
      deactivate
```

<hr>

### Simulators

#### Blender

```
wget -P ./zero_shot_scene_segmentation/simulators/blender/ https://mirrors.ocf.berkeley.edu/blender/release/Blender3.3/blender-3.3.7-linux-x64.tar.xz && \
  tar -xf ./zero_shot_scene_segmentation/simulators/blender/blender-3.3.7-linux-x64.tar.xz -C ./zero_shot_scene_segmentation/simulators/blender/ && \
    rm ./zero_shot_scene_segmentation/simulators/blender/blender-3.3.7-linux-x64.tar.xz && \
      ./zero_shot_scene_segmentation/simulators/blender/blender-3.3.7-linux-x64/3.3/python/bin/python3.10 -m ensurepip && \
        ./zero_shot_scene_segmentation/simulators/blender/blender-3.3.7-linux-x64/3.3/python/bin/python3.10 -m pip install -r ./envs/requirements/blender.txt
```

#### Habitat-Sim


<!-- ```
python3.9 -m venv ./envs/habitat-sim && \
  source ./envs/habitat-sim/bin/activate && \
    python -m pip install -r ./envs/requirements/habitat-sim.txt && \
      deactivate
``` -->

```
# Update the dependency solver for base conda for faster install
conda update -n base conda
conda install -n base conda-libmamba-solver
conda config --set solver libmamba

# Create environment for habitat-sim and habitat-lab
conda create -n habitat python=3.9 cmake=3.14.0
conda activate habitat
conda install habitat-sim withbullet -c conda-forge -c aihabitat
pip install -r ./envs/requirements/habitat.txt
```
