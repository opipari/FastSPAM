# Panoptic Memory Clouds <br/> Experiment Branch

## Branch Structure

This branch of the repository is setup to track all code for the experiments including data processing, model training, and model evaluation. Correspondingly, the code for these core functions is spread across sub-directories.

#### Directory Structure

```
.                                   # Root project directory
├── README.md                       # This readme file
├── setup.py                        # Setup script using setuptools to install the VideoPanopticSegmentation experiment package
├── requirements                    # Directory containing the specification of package dependency requirements
│   └── segment-anything.txt        # Package requirements for segment anything model package
├── envs                            # Directory containing virtual environment source files and package dependency requirements
│   └── segment-anything            # Directory for segment anything's virtual environment
└── video_panoptic_segmentation     # The central python package for VideoPanopticSegmentation package experiments
    ├── models                      # Directory containing source code for model development, training and evaluation
    │   └── segment-anything        # Directory containing all source for training and evaluating segment anything model
    ├── datasets                    # Directory containing raw datasets (DO NOT ADD DATA TO GIT TRACKING)
    └── metrics                     # Directory containing code for evaluation
```


## Setup

1. Clone repository and submodules

    ```
    git clone --recurse-submodules -b experiments git@github.com:opipari/PanopticMemoryClouds.git && \
    cd PanopticMemoryClouds
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

#### Panoptic Memory Clouds (GAPS)

Setup environments:

```
python3.8 -m venv ./envs/panoptic-memory-clouds && \
  source ./envs/panoptic-memory-clouds/bin/activate && \
    pip install -r ./requirements/panoptic-memory-clouds.txt && \
      deactivate
```

#### Video K-Net

Setup environments:

```
python3.8 -m venv ./envs/video-k-net && \
  source ./envs/video-k-net/bin/activate && \
    pip install -r ./requirements/video-k-net.txt && \
      deactivate
```






### How to use submodules

```
git submodule add -b <branch name> <remote.git> ./video_panoptic_segmentation/models/<model_name>/<model_name>/
```

### Updating a submodule to point to latest commit
```
cd <path_to_submodule>
git pull
cd ..
git commit -m "Updating <submodule>" <path_to_submodule>
git push
```
