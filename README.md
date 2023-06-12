# ZeroShot_RGB_D

## Repository Structure

This repository is setup to contain a centralized python package, `ZeroShotRGBD`, that stores all code for data simulation, data processing, model training, and model evaluation. Correspondingly, the code for these core functions is spread across sub-directories.

## Directory Structure

1. `zeroshot_rgbd`: The central python package
2. `zeroshot_rgbd/envs`: Directory containing virtualenvironment requirement and source files.
3. `zeroshot_rgbd/simulators`: Directory containing source code for data simulationa and rendering.
4. `zeroshot_rgbd/datasets`: Directory containing source code for dataset pre and post-processing as well as PyTorch Dataset Classes.
5. `zeroshot_rgbd/models`: Directory containing source code for model development, training and evaluation.

## Setup

1. Clone repository and submodules `git clone --recurse-submodules git@github.com:opipari/ZeroShot_RGB_D.git`
2. Setup habitat-sim environment
    1. `conda create -n habitat python=3.9 and cmake=3.14.0`
    2. `conda activate habitat`
    3. `conda install habitat-sim withbullet -c conda-forge -c aihabitat`
    4. `cd zeroshot_rgbd/simulators/habitat-lab`
    5. `pip install -e habitat-lab`
    6. `pip install -e habitat-baselines`
    7. `pip install pygame==2.0.1 pybullet==3.0.4`
    8. `cd ../../..`
3. Install ZeroShotRGBD package `pip install -e .`
