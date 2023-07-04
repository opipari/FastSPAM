import os
import sys
import argparse
import configparser
import csv

import math
import numpy as np
import itertools, functools, operator



def extract_scene_sequences(SCENE_DIR, SCENE_VIEWS_FILE, OUT_DIR, verbose=True):
    
    rgb2hex = lambda r,g,b: '%02X%02X%02X' % (r,g,b)
    hex2rgb = lambda hex: tuple(int(h[i:i+2], 16) for i in (0, 2, 4))

    scene_semantic_objects = {}
    with open(SEMANTIC_SCENE_FILE_PATH, "r") as sem_file:
        for line in sem_file.readlines():
            if line.startswith("HM3D Semantic Annotations"):
                continue
            
            object_id, object_hex_color, object_name, unknown = line.split(',')

            assert object_hex_color not in scene_semantic_objects.keys()
            scene_semantic_objects[object_hex_color] = {"object_id": object_id, 
                                                        "color_id": object_hex_color, 
                                                        "object_name": object_name.strip("\""),
                                                        "visible_views": []
                                                        }

    
    


    extracting_sequence = None 

    render_image_count = 0

    with open(SCENE_VIEWS_FILE, 'r') as csvfile:

        pose_reader = csv.reader(csvfile, delimiter=',')

        for pose_meta in pose_reader:
            info_ID = pose_meta[:4]
            info_position = pose_meta[4:7]
            info_rotation = pose_meta[7:11]

            # Skip information line if it is first
            if info_ID[0]=='Scene-ID':
                continue

            scene_id, trajectory_id, sensor_height_id, view_id = info_ID
            x_pos, y_pos, z_pos = info_position
            quat_w, quat_x, quat_y, quat_z = info_rotation

            # Parse pose infomration out of string type
            x_pos, y_pos, z_pos = float(x_pos), float(y_pos), float(z_pos)
            quat_w, quat_x, quat_y, quat_z = float(quat_w), float(quat_x), float(quat_y), float(quat_z)


            if (scene_id, trajectory_id, sensor_height_id) != extracting_sequence:
                extracting_sequence = (scene_id, trajectory_id, sensor_height_id)
                
                OUT_RGB_DIR = os.path.join(OUT_DIR, "JPEGImages")
                OUT_SEM_DIR = os.path.join(OUT_DIR, "Annotations")
                OUT_DEPTH_DIR = os.path.join(OUT_DIR, "PNGDepthImages")

                os.makedirs(OUT_RGB_DIR, exist_ok=True)
                os.makedirs(OUT_SEM_DIR, exist_ok=True)
                os.makedirs(OUT_DEPTH_DIR, exist_ok=True)



            rgb_file = f"{'.'.join(info_ID)}.RGB.{0:010}.png"
            rgb_image = Image.open(os.path.join(SCENE_DIR, rgb_file)).convert("RGB")
            rgb_image.save(os.path.join(OUT_RGB_DIR, view_id+".jpg"))

            sem_file = f"{'.'.join(info_ID)}.SEM.png"
            shutil.copyfile(os.path.join(SCENE_DIR, sem_file), os.path.join(OUT_SEM_DIR, view_id+".png"))

            depth_file = f"{'.'.join(info_ID)}.DEPTH.png"
            shutil.copyfile(os.path.join(SCENE_DIR, depth_file), os.path.join(OUT_DEPTH_DIR, view_id+".png"))
            



if __name__ == "__main__":

    # Read command line arguments
    argv = sys.argv

    # Ignore Blender executable and Blender-specific command line arguments
    if "--" in argv:
        argv = argv[argv.index("--") + 1:]


    parser = argparse.ArgumentParser(
                    prog='postprocess_dataset',
                    usage='python <path to postprocess_dataset.py> -- [options]',
                    description='Python script for rendering color images under active spot light illumination from the Matterport 3D semantic dataset assuming valid views have already been sampled',
                    epilog='For more information, see: https://github.com/opipari/ZeroShot_RGB_D/tree/main/zeroshot_rgbd/simulators/blender')

    parser.add_argument('-data', '--dataset-dir', help='path to directory of Matterport semantic dataset directory formatted as one sub-directory per scene', type=str)
    parser.add_argument('-out', '--output-dir', help='path to directory where output dataset should be stored', type=str)
    parser.add_argument('-v', '--verbose', help='whether verbose output printed to stdout', type=int, default=1)

    args = parser.parse_args(argv)


    if args.verbose:
        print()
        print(args)
        print()


    scene_semantic_objects = {}
    scene_directories = sorted([path for path in os.listdir(args.dataset_dir) if os.path.isdir(os.path.join(args.dataset_dir, path))])
    for scene_dir in scene_directories:
        scene_dir_path = os.path.join(args.dataset_dir, scene_dir)

        scene_view_poses_path = os.path.join(scene_dir_path, scene_dir+'.render_view_poses.csv')
        scene_has_sampled_views = os.path.isfile(scene_view_poses_path)

        SEMANTIC_SCENE_FILE_PATH = os.path.join(scene_dir_path, scene_dir+'.semantic.txt')

        
        with open(SEMANTIC_SCENE_FILE_PATH, "r") as sem_file:
            for line in sem_file.readlines():
                if line.startswith("HM3D Semantic Annotations"):
                    continue
                
                object_id, object_hex_color, object_name, unknown = line.split(',')

                if object_name not in scene_semantic_objects:
                    scene_semantic_objects[object_name] = {}
                if scene_dir not in scene_semantic_objects[object_name]:
                    scene_semantic_objects[object_name][scene_dir] = []

                scene_semantic_objects[object_name][scene_dir].append(object_hex_color)

    import json
    with open("res.json", 'w') as f:
        json.dump(scene_semantic_objects, f, ensure_ascii=False, indent=4)
        # if scene_has_sampled_views:
        #     extract_scene_sequences(scene_dir_path, scene_view_poses_path, args.output_dir, verbose=args.verbose)
    print("number categories:", len(scene_semantic_objects.keys()))
    print("max number of instances:", max([len(scene_semantic_objects[obj_name][scene_dir]) for obj_name in scene_semantic_objects for scene_dir in scene_semantic_objects[obj_name]]))
    
    if args.verbose:
        print()
        print("***********************")
        print(f"DONE RENDERING ALL SCENES")
        print("***********************")
        print()