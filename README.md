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



## Models

### Segment Anything Model (SAM)

```
python3.8 -m venv ./envs/segment-anything && \
  source ./envs/segment-anything/bin/activate && \
    pip install -r ./envs/requirements/segment-anything.txt && \
      deactivate && \
        wget -P ./zero_shot_scene_segmentation/models/segment-anything/segment-anything/checkpoints/ https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
```

## Datasets

### Generic Data Processing Environment

This environment is to be used for dataset preprocessing since Blender's version of python3.10 is not compatible with needed cocotools api.

```
python3.8 -m venv ./envs/data-processing && \
  source ./envs/data-processing/bin/activate && \
    pip install -r ./envs/requirements/data-processing.txt && \
      deactivate
```

### [Habitat Matterport 3D Semantic Dataset](https://aihabitat.org/datasets/hm3d-semantics/)

#### Setup

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

## Simulators

### Blender

#### Setup

```
wget -P ./zero_shot_scene_segmentation/simulators/blender/ https://mirrors.ocf.berkeley.edu/blender/release/Blender3.3/blender-3.3.7-linux-x64.tar.xz && \
  tar -xf ./zero_shot_scene_segmentation/simulators/blender/blender-3.3.7-linux-x64.tar.xz -C ./zero_shot_scene_segmentation/simulators/blender/ && \
    rm ./zero_shot_scene_segmentation/simulators/blender/blender-3.3.7-linux-x64.tar.xz && \
      ./zero_shot_scene_segmentation/simulators/blender/blender-3.3.7-linux-x64/3.3/python/bin/python3.10 -m ensurepip && \
        ./zero_shot_scene_segmentation/simulators/blender/blender-3.3.7-linux-x64/3.3/python/bin/python3.10 -m pip install -r ./envs/requirements/blender.txt
```

### Habitat-Sim


<!-- ```
python3.9 -m venv ./envs/habitat-sim && \
  source ./envs/habitat-sim/bin/activate && \
    python -m pip install -r ./envs/requirements/habitat-sim.txt && \
      deactivate
``` -->

#### Setup

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

## ZeroShotSceneSegmentation Package

#### Setup

```
pip install -e .
```
