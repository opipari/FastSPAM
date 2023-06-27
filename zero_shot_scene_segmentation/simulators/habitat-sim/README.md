


```
conda activate habitat
```


```
python zero_shot_scene_segmentation/simulators/habitat-sim/sample_trajectories.py -- \
	-config zero_shot_scene_segmentation/simulators/habitat-sim/configs/render_config.ini \
	-data zero_shot_scene_segmentation/datasets/raw_data/HM3D/example/ \
	-out zero_shot_scene_segmentation/datasets/raw_data/samples/example/
```