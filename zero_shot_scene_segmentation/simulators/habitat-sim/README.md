

## Setup

Follow the conda environment installation process detailed on the main repository [README.md#habitat-sim](https://github.com/opipari/ZeroShotSceneSegmentation/tree/main#habitat-sim).




## Workflow

1. Initialize the `habitat` conda environment that you installed during [setup](#Setup) 

    ```
    conda activate habitat
    ```
2. Configure Sampling Parameters
    - Set any specific sampling parameters you'd like to use in a `.ini` configuration file located in `./configs/`
3. Sample Trajectories
	- **Shortcut**
        ```
        python zero_shot_scene_segmentation/simulators/habitat-sim/sample_trajectories.py -- \
	      -config zero_shot_scene_segmentation/simulators/habitat-sim/configs/trajectory_config.ini \
	      -data zero_shot_scene_segmentation/datasets/raw_data/HM3D/example/ \
	      -out zero_shot_scene_segmentation/datasets/raw_data/trajectory_renders/example/
        ```
    - If you'd like to render RGB images using the habitat-sim render, simply add the `-render` flag to the above script.
    - **Expected Output**
        The above command should create a new directory for each scene in the example set of matterport that includes a semantic label mesh. For each of these scenes, there will be a sub-directory each containing a `<scene name>.habitat_trajectory_poses.csv` file describing the camera pose for each sampled trajectory.
4. Modify Sampling Parameters
    - Edit the parameters specified in the `./configs/trajectory_config.ini` configuration file.
    - Useful, for example, if you want to simulate data from multiple agent heights.
    - In our experiments, we are simulating agent trajectories with `sensor_height = 100`(mm) and `sensor_height = 1000`mm, which is equivalent to `0.1m` and `1m` respectively.
5. Re-sample trajectories and append to existing output `csv` files.
    ```

    ```
