# Blender Simulation Pipeline


This folder contains all blender python scripts for rendering visual imagery and corresponding instance segmentation labels from the [Matterport 3D Semantic dataset](https://aihabitat.org/datasets/hm3d-semantics/).

## Prerequisites

1. Install [Blender LTS](https://www.blender.org/download/releases/3-3/). This code was developed and tested with Blenderv3.3.7 on Ubuntu 20.04.
    - Details on main repository [README.md()]()
2. Download and extract all scenes in the [Matterport 3D Semantic dataset](https://aihabitat.org/datasets/hm3d-semantics/)
    - Details on main repository [README.md]()


## Simulation Workflow

1. **Calculate valid camera views**
    - Run a blender command line script (i.e. ./blender.exe --python script.py -- command line args) to sample valid views for each scene in the matterport dataset
    - Input: desired sampling resolution in position and rotation and camera parameters (focal length, image size)
    - Output: one csv file per scene listing the valid camera poses
        - Valid camera pose defined as one with all corners and center ray viewing inner mesh and at least 0.25 meters from camera origin
        - Changing this definition will require re-rendering
    - **Shortcut**
        ```
        ./zero_shot_scene_segmentation/simulators/blender/blender-3.3.7-linux-x64/blender \
          --python ./zero_shot_scene_segmentation/simulators/blender/sample_views.py \
            -- \
            -config ./zero_shot_scene_segmentation/simulators/blender/configs/render_config.ini \
            -data ./zero_shot_scene_segmentation/datasets/matterport/HM3D/example/ \
            -out ./zero_shot_scene_segmentation/datasets/renders/example/
      ```
    - **Visualize Sampled Views**
      ```
      ./zero_shot_scene_segmentation/simulators/blender/blender-3.3.7-linux-x64/blender \
        --python ./zero_shot_scene_segmentation/simulators/blender/visualize_views.py \
          -- \
          -scene 00861-GLAQ4DNUx5U \
          -config ./zero_shot_scene_segmentation/simulators/blender/configs/render_config.ini \
          -data ./zero_shot_scene_segmentation/datasets/matterport/HM3D/example/ \
          -out ./zero_shot_scene_segmentation/datasets/renders/example/
      ```
2. **Render semantic label images**
    - Run a blender command line script to render semantic images given valid view metadata
    - Input: meta data generated from step 1
    - Output: one directory per scene containing semantic images in png format which are mappable by name to corresponding pose
    - **Shortcut**
        ```
        ./blender-3.3.7-linux-x64/3.3/python/bin/python3.10 ./zero_shot_scene_segmentation/simulators/blender/post_process_semantics.py \
            -- \
            -data ./zero_shot_scene_segmentation/datasets/matterport/HM3D/example/ \
            -out ./zero_shot_scene_segmentation/datasets/renders/example/ 
        ```
    - **Post-Process semantic labels**
        - Run a post-processing python script to map Scene ID to Object ID to Image ID
        - Output: one csv file per scene which contains Scene ID, Object ID, Image ID triplets
        - A triplet will be included if at least one pixel of a given object is visible within a corresponding image based on the color value in semantic images from step 2
        - **Shortcut**
            ```
            ./zero_shot_scene_segmentation/simulators/blender/blender-3.3.7-linux-x64/blender \
            --background \
            --python ./zero_shot_scene_segmentation/simulators/blender/render_semantics.py \
              -- \
              -config ./zero_shot_scene_segmentation/simulators/blender/configs/render_config.ini \
              -data ./zero_shot_scene_segmentation/datasets/matterport/HM3D/example/ \
              -out ./zero_shot_scene_segmentation/datasets/renders/example/ 
            ```
4. Render visual images
    - Run a blender command line script to render visual RGB images given valid view metadata
    - Input: meta data generated from step 1 and lighting parameters (i.e. spot light strength)
    - Output: one directory per scene containing semantic images in png format which are mappable by name to corresponding pose and light conditions
    - **Shortcut**
        ````
        ./blender-3.3.7-linux-x64/blender \
        --background \
        --python ./zero_shot_scene_segmentation/simulators/blender/render_color.py \
        -- \
        -config ./zero_shot_scene_segmentation/simulators/blender/configs/render_config.ini \
        -light-config ./zero_shot_scene_segmentation/simulators/blender/configs/illumination_0000000000_config.ini \
        -data ./zero_shot_scene_segmentation/datasets/matterport/HM3D/example/ \
        -out ./zero_shot_scene_segmentation/datasets/renders/example/
        ````


- Copy view files between machines:
    ```
    ./blender-3.3.7-linux-x64/3.3/python/bin/python3.10 ./zero_shot_scene_segmentation/simulators/blender/copy_views.py \
    -- \
    -src /media/topipari/0CD418EB76995EEF/ZeroShotSceneSegmentation/datasets/renders/example/ \
    -dst ./zero_shot_scene_segmentation/datasets/renders/example/
    ```

## Simulation Debugging

The blender python script in `./workbench.py` is intended for use in the interactive Blender GUI. This script allows visualization of the rejection sampling process used for view sampling and can be used for interactive script development.

