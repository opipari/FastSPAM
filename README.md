# Panoptic Memory Clouds
## Experiment Branch

## Repository Structure

This repository is setup to contain a centralized python package, `ZeroShotSceneSegmentation`, that stores all code for data simulation, data processing, model training, and model evaluation. Correspondingly, the code for these core functions is spread across sub-directories.

#### Directory Structure

```
.                                   # Root project directory
├── README.md                       # This readme file
├── setup.py                        # Setup script using setuptools to install the ZeroShotSceneSegmentation package
├── requirements                    # Directory containing the specification of package dependency requirements
│   └── segment-anything.txt        # Package requirements for segment anything model package
├── envs                            # Directory containing virtual environment source files and package dependency requirements
│   └── segment-anything            # Directory for segment anything's virtual environment
└── zero_shot_scene_segmentation    # The central python package for ZeroShotSceneSegmentation package
    ├── models                      # Directory containing source code for model development, training and evaluation
    │   └── segment-anything        # Directory containing all source for training and evaluating segment anything model
    ├── datasets                    # Directory containing raw datasets (DO NOT ADD TO GIT TRACKING)
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

#### Massive Panoptic Video Dataset (MVPd)

This dataset is hosted in [external repository](https://github.com/opipari/MVPd), and managed in these experiments as a submodule within the [`video_panoptic_segmentation/datasets/` folder](video_panoptic_segmentation/datasets/).

To download this dataset, simply execute the following commands:

```
cd video_panoptic_segmentation/datasets/MVPd && \
  ./download.sh && \
    cd ../../..
```


<hr>

### Models

#### Segment Anything Model (SAM)

```
python3.8 -m venv ./envs/segment-anything && \
  source ./envs/segment-anything/bin/activate && \
    pip install -r ./requirements/segment-anything.txt && \
      deactivate && \
        wget -P ./zero_shot_scene_segmentation/models/segment-anything/segment-anything/checkpoints/ https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
```

#### Pyramid / Panoptic Associating Objects with Transformers (PAOT)

Setup environments:

```
python3.8 -m venv ./envs/paot-benchmark && \
  source ./envs/paot-benchmark/bin/activate && \
    pip install -r ./requirements/paot-benchmark/base.txt &&
    pip install -r ./requirements/paot-benchmark/deps.txt && \
      deactivate
```

For reproducing published results on the VIPOSeg dataset, refer to the [paot-benchmark.md](https://github.com/opipari/ZeroShotSceneSegmentation/blob/main/paot-benchmark.md) file.


#### Geometry-Pyramid / Panoptic Associating Objects with Transformers (GEO-PAOT)

Setup environments:

```
python3.8 -m venv ./envs/geo-paot-benchmark && \
  source ./envs/geo-paot-benchmark/bin/activate && \
    pip install -r ./requirements/geo-paot-benchmark/base.txt &&
    pip install -r ./requirements/geo-paot-benchmark/deps.txt && \
      deactivate
```

#### Gemoetry Aware Panoptic Segmentation (GAPS)

Setup environments:

```
python3.8 -m venv ./envs/gaps && \
  source ./envs/gaps/bin/activate && \
    pip install -r ./requirements/gaps.txt && \
      deactivate
```






### Add New model submodule

```
git submodule add -b <branch name> <remote.git> ./zero_shot_scene_segmentation/models/<model_name>/<model_name>/
```
