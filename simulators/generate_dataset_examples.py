import math
import os
import random
import sys

import imageio
import magnum as mn
import numpy as np

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

# function to display the topdown map
from PIL import Image

import habitat_sim
from habitat_sim.utils import common as utils
from habitat_sim.utils import viz_utils as vut
from habitat.utils.visualizations import maps


def save_sample(index, dest_dir, rgb_obs, semantic_obs=np.array([]), depth_obs=np.array([])):
    from habitat_sim.utils.common import d3_40_colors_rgb

    rgb_img = Image.fromarray(rgb_obs, mode="RGBA")
    rgb_img.save(os.path.join(dest_dir, str(index)+"_rgb.png"))

    if semantic_obs.size != 0:
        semantic_img = Image.new("P", (semantic_obs.shape[1], semantic_obs.shape[0]))
        semantic_img.putpalette(d3_40_colors_rgb.flatten())
        semantic_img.putdata((semantic_obs.flatten() % 40).astype(np.uint8))
        semantic_img = semantic_img.convert("RGBA")
        semantic_img.save(os.path.join(dest_dir, str(index)+"_semantic.png"))


    if depth_obs.size != 0:
        depth_img = Image.fromarray((depth_obs / 10 * 255).astype(np.uint8), mode="L")
        depth_img.save(os.path.join(dest_dir, str(index)+"_depth.png"))



def make_cfg(settings):
    sim_cfg = habitat_sim.SimulatorConfiguration()
    sim_cfg.gpu_device_id = 0
    sim_cfg.scene_id = settings["scene"]
    sim_cfg.scene_dataset_config_file = settings["scene_dataset"]
    sim_cfg.enable_physics = settings["enable_physics"]

    # Note: all sensors must have the same resolution
    sensor_specs = []

    color_sensor_spec = habitat_sim.CameraSensorSpec()
    color_sensor_spec.uuid = "color_sensor"
    color_sensor_spec.sensor_type = habitat_sim.SensorType.COLOR
    color_sensor_spec.resolution = [settings["height"], settings["width"]]
    color_sensor_spec.position = [0.0, settings["sensor_height"], 0.0]
    color_sensor_spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE
    sensor_specs.append(color_sensor_spec)

    depth_sensor_spec = habitat_sim.CameraSensorSpec()
    depth_sensor_spec.uuid = "depth_sensor"
    depth_sensor_spec.sensor_type = habitat_sim.SensorType.DEPTH
    depth_sensor_spec.resolution = [settings["height"], settings["width"]]
    depth_sensor_spec.position = [0.0, settings["sensor_height"], 0.0]
    depth_sensor_spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE
    sensor_specs.append(depth_sensor_spec)

    semantic_sensor_spec = habitat_sim.CameraSensorSpec()
    semantic_sensor_spec.uuid = "semantic_sensor"
    semantic_sensor_spec.sensor_type = habitat_sim.SensorType.SEMANTIC
    semantic_sensor_spec.resolution = [settings["height"], settings["width"]]
    semantic_sensor_spec.position = [0.0, settings["sensor_height"], 0.0]
    semantic_sensor_spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE
    sensor_specs.append(semantic_sensor_spec)

    # Here you can specify the amount of displacement in a forward action and the turn angle
    agent_cfg = habitat_sim.agent.AgentConfiguration()
    agent_cfg.sensor_specifications = sensor_specs
    agent_cfg.action_space = {
        "move_forward": habitat_sim.agent.ActionSpec(
            "move_forward", habitat_sim.agent.ActuationSpec(amount=0.25)
        ),
        "turn_left": habitat_sim.agent.ActionSpec(
            "turn_left", habitat_sim.agent.ActuationSpec(amount=30.0)
        ),
        "turn_right": habitat_sim.agent.ActionSpec(
            "turn_right", habitat_sim.agent.ActuationSpec(amount=30.0)
        ),
    }

    return habitat_sim.Configuration(sim_cfg, [agent_cfg])



# convert 3d points to 2d topdown coordinates
def convert_points_to_topdown(pathfinder, points, meters_per_pixel):
    points_topdown = []
    bounds = pathfinder.get_bounds()
    for point in points:
        # convert 3D x,z to topdown x,y
        px = (point[0] - bounds[0][0]) / meters_per_pixel
        py = (point[2] - bounds[0][2]) / meters_per_pixel
        points_topdown.append(np.array([px, py]))
    return points_topdown


# display a topdown map with matplotlib
def save_map(dest_dir, topdown_map, key_points=None):
    plt.figure(figsize=(12, 8))
    ax = plt.subplot(1, 1, 1)
    ax.axis("off")
    plt.imshow(topdown_map)
    # plot points on map
    if key_points is not None:
        for point in key_points:
            plt.plot(point[0], point[1], marker="o", markersize=10, alpha=0.8)
    plt.savefig(os.path.join(dest_dir,"map.png"))




if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--test-scene", dest="test_scene")
    parser.add_argument("--test-scene-config", dest="test_scene_dataset")
    parser.add_argument("--dest-dir", dest="dest_dir")
    parser.set_defaults(test_scene="/media/mytre/0CD418EB76995EEF/SegmentationProject/datasets/matterport/HM3D/val/00800-TEEsavR23oF/TEEsavR23oF.basis.glb",    
                    test_scene_dataset="/media/mytre/0CD418EB76995EEF/SegmentationProject/datasets/matterport/HM3D/hm3d_annotated_val_basis.scene_dataset_config.json",
                    dest_dir="/media/mytre/0CD418EB76995EEF/SegmentationProject/simulators/test/")
    args, _ = parser.parse_known_args()
    test_scene = args.test_scene
    test_scene_dataset = args.test_scene_dataset
    dest_dir = args.dest_dir

    if not os.path.isdir(dest_dir):
        os.makedirs(dest_dir)

    sim_settings = {
        "width": 800,  # Spatial resolution of the observations
        "height": 534,
        "scene": test_scene,  # Scene path
        "scene_dataset": test_scene_dataset,  # the scene dataset configuration files
        "default_agent": 0,
        "sensor_height": 0.25,  # Height of sensors in meters
        "color_sensor": True,  # RGB sensor
        "depth_sensor": True,  # Depth sensor
        "semantic_sensor": True,  # Semantic sensor
        "seed": 1,  # used in the random navigation
        "enable_physics": False,  # kinematics only
    }

    cfg = make_cfg(sim_settings)
    # Needed to handle out of order cell run in Colab
    try:  # Got to make initialization idiot proof
        sim.close()
    except NameError:
        pass
    sim = habitat_sim.Simulator(cfg)



    # the randomness is needed when choosing the actions
    random.seed(sim_settings["seed"])
    sim.seed(sim_settings["seed"])

    # Set agent state
    agent = sim.initialize_agent(sim_settings["default_agent"])
    agent_state = habitat_sim.AgentState()
    agent_state.position = np.array([-0.6, 0.0, 0.0])  # world space
    agent.set_state(agent_state)

    # Get agent state
    agent_state = agent.get_state()
    print("agent_state: position", agent_state.position, "rotation", agent_state.rotation)



    # @markdown With a valid PathFinder instance:
    if not sim.pathfinder.is_loaded:
        print("Pathfinder not initialized, aborting.")
    else:
        seed = 4  # @param {type:"integer"}
        sim.pathfinder.seed(seed)

        # fmt off
        # @markdown 1. Sample valid points on the NavMesh for agent spawn location and pathfinding goal.
        # fmt on
        sample1 = sim.pathfinder.get_random_navigable_point()
        sample2 = sim.pathfinder.get_random_navigable_point()

        # @markdown 2. Use ShortestPath module to compute path between samples.
        path = habitat_sim.ShortestPath()
        path.requested_start = sample1
        path.requested_end = sample2
        found_path = sim.pathfinder.find_path(path)
        geodesic_distance = path.geodesic_distance
        path_points = path.points
        # @markdown - Success, geodesic path length, and 3D points can be queried.
        print("found_path : " + str(found_path))
        print("geodesic_distance : " + str(geodesic_distance))
        print("path_points : " + str(path_points))

        # @markdown 3. Display trajectory (if found) on a topdown map of ground floor
        if found_path:
            meters_per_pixel = 0.025
            scene_bb = sim.get_active_scene_graph().get_root_node().cumulative_bb
            height = scene_bb.y().min

            top_down_map = maps.get_topdown_map(
                sim.pathfinder, height, meters_per_pixel=meters_per_pixel
            )
            recolor_map = np.array(
                [[255, 255, 255], [128, 128, 128], [0, 0, 0]], dtype=np.uint8
            )
            top_down_map = recolor_map[top_down_map]
            grid_dimensions = (top_down_map.shape[0], top_down_map.shape[1])
            # convert world trajectory points to maps module grid points
            trajectory = [
                maps.to_grid(
                    path_point[2],
                    path_point[0],
                    grid_dimensions,
                    pathfinder=sim.pathfinder,
                )
                for path_point in path_points
            ]
            grid_tangent = mn.Vector2(
                trajectory[1][1] - trajectory[0][1], trajectory[1][0] - trajectory[0][0]
            )
            path_initial_tangent = grid_tangent / grid_tangent.length()
            initial_angle = math.atan2(path_initial_tangent[0], path_initial_tangent[1])
            # draw the agent and trajectory on the map
            maps.draw_path(top_down_map, trajectory)
            maps.draw_agent(
                top_down_map, trajectory[0], initial_angle, agent_radius_px=8
            )
            print("saving map...")
            save_map(dest_dir, top_down_map)

            # @markdown 4. (optional) Place agent and render images at trajectory points (if found).
            print("Rendering observations at path points:")
            tangent = path_points[1] - path_points[0]
            agent_state = habitat_sim.AgentState()
            for ix, point in enumerate(path_points):
                if ix < len(path_points) - 1:
                    tangent = path_points[ix + 1] - point
                    agent_state.position = point
                    tangent_orientation_matrix = mn.Matrix4.look_at(
                        point, point + tangent, np.array([0, 1.0, 0])
                    )
                    tangent_orientation_q = mn.Quaternion.from_matrix(
                        tangent_orientation_matrix.rotation()
                    )
                    agent_state.rotation = utils.quat_from_magnum(tangent_orientation_q)
                    agent.set_state(agent_state)

                    observations = sim.get_sensor_observations()
                    rgb = observations["color_sensor"]
                    semantic = observations["semantic_sensor"]
                    depth = observations["depth_sensor"]

                    save_sample(ix, dest_dir, rgb, semantic, depth)