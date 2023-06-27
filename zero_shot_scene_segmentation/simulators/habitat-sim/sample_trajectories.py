import os
import sys
import argparse
import configparser

import math
import numpy as np
import magnum as mn
import itertools, functools, operator

import habitat_sim
from habitat_sim.utils import common as utils
from habitat_sim.utils import viz_utils as vut

import cv2
from PIL import Image 
from scipy.spatial.transform import Rotation as R



def make_cfg(scene, scene_config, CONFIG):
    sim_cfg = habitat_sim.SimulatorConfiguration()
    sim_cfg.gpu_device_id = 0
    sim_cfg.scene_id = scene
    sim_cfg.scene_dataset_config_file = scene_config
    sim_cfg.enable_physics = False

    # Note: all sensors must have the same resolution
    sensor_specs = []

    color_sensor_spec = habitat_sim.CameraSensorSpec()
    color_sensor_spec.uuid = "color_sensor"
    color_sensor_spec.sensor_type = habitat_sim.SensorType.COLOR
    color_sensor_spec.resolution = [CONFIG['sensor_spec'].getfloat('image_height'), CONFIG['sensor_spec'].getfloat('image_width')]
    color_sensor_spec.position = [0.0, CONFIG['sensor_spec'].getfloat('sensor_height'), 0.0]
    color_sensor_spec.hfov = mn.Deg(CONFIG['sensor_spec'].getfloat('hfov'))
    color_sensor_spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE
    sensor_specs.append(color_sensor_spec)

    depth_sensor_spec = habitat_sim.CameraSensorSpec()
    depth_sensor_spec.uuid = "depth_sensor"
    depth_sensor_spec.sensor_type = habitat_sim.SensorType.DEPTH
    depth_sensor_spec.resolution = [CONFIG['sensor_spec'].getfloat('image_height'), CONFIG['sensor_spec'].getfloat('image_width')]
    depth_sensor_spec.position = [0.0, CONFIG['sensor_spec'].getfloat('sensor_height'), 0.0]
    depth_sensor_spec.hfov = mn.Deg(CONFIG['sensor_spec'].getfloat('hfov'))
    depth_sensor_spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE
    sensor_specs.append(depth_sensor_spec)

    semantic_sensor_spec = habitat_sim.CameraSensorSpec()
    semantic_sensor_spec.uuid = "semantic_sensor"
    semantic_sensor_spec.sensor_type = habitat_sim.SensorType.SEMANTIC
    semantic_sensor_spec.resolution = [CONFIG['sensor_spec'].getfloat('image_height'), CONFIG['sensor_spec'].getfloat('image_width')]
    semantic_sensor_spec.position = [0.0, CONFIG['sensor_spec'].getfloat('sensor_height'), 0.0]
    semantic_sensor_spec.hfov = mn.Deg(CONFIG['sensor_spec'].getfloat('hfov'))
    semantic_sensor_spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE
    sensor_specs.append(semantic_sensor_spec)

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



def euclidean_distance(arr_a, arr_b):
    print(np.sqrt(np.sum((arr_a-arr_b)**2)))
    print(np.linalg.norm(arr_a-arr_b))

def smooth_path(path_points, forward=0.1, turn=15):
    expanded_targets = []
    for ix, point in enumerate(path_points):
        if ix < len(path_points) - 1:
            tangent = path_points[ix + 1] - point
            
            euclidean_distance(path_points[ix + 1], point)
            print(type(tangent), point, path_points[ix+1])

            raise
#python zero_shot_scene_segmentation/simulators/habitat-sim/render_trajectories.py -- -config zero_shot_scene_segmentation/simulators/habitat-sim/configs/render_config.ini -data zero_shot_scene_segmentation/datasets/raw_data/HM3D/train -out zero_shot_scene_segmentation/datasets/raw_data/samples/train/
def render_scene_trajectories(SCENE_DIR, SCENE_OUT_DIR, CONFIG, verbose=True):
    
    SCENE_NAME = SCENE_DIR.split('/')[-1].split('-')[-1]
    SCENE_FILE = SCENE_NAME+'.glb'
    SEMANTIC_SCENE_FILE = SCENE_NAME+'.semantic.glb'
    SCENE_CONFIG_FILE = f"hm3d_annotated_{SCENE_DIR.split('/')[-2]}_basis.scene_dataset_config.json"
    SCENE_CONFIG_PATH = os.path.join('/'.join(SCENE_DIR.split('/')[:-2]), SCENE_CONFIG_FILE)


    if verbose:
        print()
        print("********************")
        print(f"SAMPLING TRAJECTORIES FOR SCENE: {SCENE_NAME}")
        print("********************")
        print()

    os.makedirs(SCENE_OUT_DIR, exist_ok=True)




    if verbose:
        print()
        print("***********************")
        print("INITIALIZING SCENE")
    
    cfg = make_cfg(os.path.join(SCENE_DIR, SCENE_FILE), SCENE_CONFIG_PATH, CONFIG)
    sim = habitat_sim.Simulator(cfg)

    if not sim.pathfinder.is_loaded:
        navmesh_settings = habitat_sim.NavMeshSettings()
        navmesh_settings.set_defaults()
        # navmesh_settings.cell_size = CONFIG['navmesh_settings'].getfloat('cell_size')
        # navmesh_settings.cell_height = CONFIG['navmesh_settings'].getfloat('cell_height')
        # navmesh_settings.agent_height = CONFIG['navmesh_settings'].getfloat('agent_height')
        # navmesh_settings.agent_radius = CONFIG['navmesh_settings'].getfloat('agent_radius')
        # navmesh_settings.agent_max_climb = CONFIG['navmesh_settings'].getfloat('agent_max_climb')
        navmesh_success = sim.recompute_navmesh(sim.pathfinder, navmesh_settings)

        # assert navmesh_success
    

    agent = sim.initialize_agent(0)


    if verbose:
        print("DONE INITIALIZING SCENE")
        print("***********************")
        print()

    
    
        
    accepted_view_file = open(os.path.join(SCENE_OUT_DIR, f"{SCENE_NAME}.habitat_trajectory_poses.csv"), "w")
    accepted_view_file.write('Scene-ID,Trajectory-ID,Valid-View-ID,X-Position,Y-Position,Z-Position,W-Quaternion,X-Quaternion,Y-Quaternion,Z-Quaternion\n')
    accepted_view_file.flush()
    


    if verbose:
        print()
        print("***********************")
        print(f"INITIATING RENDERING")
    


    valid_view_count = 0
    valid_path_count = 0 
    seed = 0
    sim.pathfinder.seed(seed)
    while valid_view_count < CONFIG['trajectory_settings'].getint('number_frames_per_scene'):

        # Sample valid points on the NavMesh for agent spawn location and pathfinding goal.
        sample1 = sim.pathfinder.get_random_navigable_point()
        sample2 = sim.pathfinder.get_random_navigable_point()
        # Use ShortestPath module to compute path between samples.
        path = habitat_sim.ShortestPath()
        path.requested_start = sample1
        path.requested_end = sample2
        found_path = sim.pathfinder.find_path(path)
        geodesic_distance = path.geodesic_distance
        path_points = path.points

        print(len(path_points), sample1, sample2)
        # Display trajectory (if found) on a topdown map of ground floor
        if found_path and len(path_points) > CONFIG['trajectory_settings'].getint('minimum_frames_per_trajectory'):
            smooth_path(path_points)
            trajectory_view_count = 0
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
                    rgb, depth, semantic = observations["color_sensor"], observations["depth_sensor"], observations["semantic_sensor"]

                    rgb_img = Image.fromarray(rgb, mode="RGBA")
                    rgb_img.save(os.path.join(SCENE_OUT_DIR, f"{SCENE_NAME}.{valid_path_count:010}.{trajectory_view_count:010}.RGB.png"))


                    # x, y, z = agent_state.position
                    # quat_w, quat_x, quat_y, quat_z = [agent_state.rotation.real]+list(agent_state.rotation.imag)

                    sensor_pose = agent.get_state().sensor_states['color_sensor']
                    x, y, z = sensor_pose.position
                    quat_w, quat_x, quat_y, quat_z = [sensor_pose.rotation.real]+list(sensor_pose.rotation.imag)

                    accepted_view_file.write(f'{SCENE_NAME},{valid_path_count:010},{trajectory_view_count:010},{x},{y},{z},{quat_w},{quat_x},{quat_y},{quat_z}\n')
                    accepted_view_file.flush()


                    trajectory_view_count += 1
                    valid_view_count += 1
            valid_path_count += 1

    accepted_view_file.close()

    if verbose:
        print("***********************")
        print(f"DONE SIMULATING {valid_path_count} TRAJECTORY SAMPLES")
        print(f"ACCEPTED {valid_view_count} VALID VIEWS")
        print("***********************")
        print()



if __name__ == "__main__":

    # Read command line arguments
    argv = sys.argv

    # Ignore Blender executable and Blender-specific command line arguments
    if "--" in argv:
        argv = argv[argv.index("--") + 1:]


    parser = argparse.ArgumentParser(
                    prog='render_trajectories',
                    usage='python <path to render_trajectories.py> -- [options]',
                    description='Python script for rendering trajectories from the Matterport 3D semantic dataset using habitat sim',
                    epilog='For more information, see: https://github.com/opipari/ZeroShot_RGB_D/tree/main/zeroshot_rgbd/simulators/habitat-sim')

    parser.add_argument('-config', '--config-file', help='path to ini file containing rendering and sampling configuration', type=str)
    parser.add_argument('-data', '--dataset-dir', help='path to directory of Matterport semantic dataset directory formatted as one sub-directory per scene', type=str)
    parser.add_argument('-out', '--output-dir', help='path to directory where output dataset should be stored', type=str)
    parser.add_argument('-v', '--verbose', help='whether verbose output printed to stdout', type=int, default=1)

    args = parser.parse_args(argv)

    config = configparser.ConfigParser()
    config.read(args.config_file)

    if args.verbose:
        print()
        print(args)
        print()
        print(config)
        print()


    scene_directories = sorted([path for path in os.listdir(args.dataset_dir) if os.path.isdir(os.path.join(args.dataset_dir, path))])
    for scene_dir in scene_directories:
        scene_dir_path = os.path.join(args.dataset_dir, scene_dir)

        scene_files = os.listdir(scene_dir_path)

        scene_out_path = os.path.join(args.output_dir, scene_dir)

        scene_has_semantic_mesh = any([fl.endswith('.semantic.glb') for fl in scene_files])
        scene_has_semantic_txt = any([fl.endswith('.semantic.txt') for fl in scene_files])

        if scene_has_semantic_mesh and scene_has_semantic_txt:
            render_scene_trajectories(scene_dir_path, scene_out_path, config, verbose=args.verbose)

    if args.verbose:
        print()
        print("***********************")
        print(f"DONE RENDERING ALL SCENES")
        print("***********************")
        print()