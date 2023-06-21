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
    ├── datasets                    # Directory containing source code for dataset processing as well as PyTorch Dataset classes
    ├── models                      # Directory containing source code for model development, training and evaluation
    │   └── segment-anything        # Directory containing all source for training and evaluating segment anything model
    └── simulators                  # Directory containing source code for data simulationa and rendering
        └── blender                 # Directory containing source code for blender-based rendering
```


## Setup

1. Clone repository and submodules
    ```
    git clone --recurse-submodules git@github.com:opipari/ZeroShotSceneSegmentation.git
    ```



## Models

### Segment Anything Model (SAM)

```
python3.8 -m venv ./envs/segment-anything && \
  source ./envs/segment-anything/bin/activate && \
    pip install -r ./envs/requirements/segment-anything.txt && \
      deactivate
```


## Simulators

### Blender
```
wget -P ./zero_shot_scene_segmentation/simulators/blender/ https://mirrors.ocf.berkeley.edu/blender/release/Blender3.3/blender-3.3.7-linux-x64.tar.xz && \
  tar -xf ./zero_shot_scene_segmentation/simulators/blender/blender-3.3.7-linux-x64.tar.xz -C ./zero_shot_scene_segmentation/simulators/blender/ && \
    rm ./zero_shot_scene_segmentation/simulators/blender/blender-3.3.7-linux-x64.tar.xz && \
      ./zero_shot_scene_segmentation/simulators/blender/blender-3.3.7-linux-x64/3.3/python/bin/python3.10 -m ensurepip && \
        ./zero_shot_scene_segmentation/simulators/blender/blender-3.3.7-linux-x64/3.3/python/bin/python3.10 -m pip install -r ./envs/requirements/blender.txt
```

### Habitat-Sim

1. Setup habitat-sim environment
    1. `conda create -n habitat python=3.9 and cmake=3.14.0`
    2. `conda activate habitat`
    3. `conda install habitat-sim withbullet -c conda-forge -c aihabitat`
    4. `cd zeroshot_rgbd/simulators/habitat-lab`
    5. `pip install -e habitat-lab`
    6. `pip install -e habitat-baselines`
    7. `pip install pygame==2.0.1 pybullet==3.0.4`
    8. `cd ../../..`
3. Install ZeroShotRGBD package `pip install -e .`
