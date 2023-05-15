# ZeroShot_RGB_D


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
