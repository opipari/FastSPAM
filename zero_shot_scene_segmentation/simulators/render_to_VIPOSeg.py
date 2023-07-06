import os
import sys
import argparse
import configparser
import shutil
import csv
import json

import math
import numpy as np
from PIL import Image
import itertools, functools, operator

import torch
import torchvision


def get_raw_category_to_mpcat40_map():

    # RAW_CATEGORY_TO_MPCAT40_MAPPING represents a many to one mapping to identify the keys to use for mapping labeled categories to matterport categories
    category_to_mpcat40_mapping = {}

    # RAW_CATEGORY_TO_CATEGORY_MAPPING represents a many to one mapping to identify the keys to use for mapping raw categories to labeled cattegories
    raw_category_to_category_mapping ={}
    
    CATEGORY_TO_MPCAT40_MAPPING_FILE = os.path.join(os.getcwd(), './zero_shot_scene_segmentation/simulators/matterport_category_mappings.tsv')
    with open(CATEGORY_TO_MPCAT40_MAPPING_FILE, 'r') as csvfile:

        map_reader = csv.reader(csvfile, delimiter='\t')

        for object_meta in map_reader:
            if object_meta[0]=='index':
                continue
            # print(object_meta)
            
            object_raw_category = object_meta[1]
            object_category = object_meta[2]
            # print(object_meta[16])
            object_mpcat40index = int(object_meta[16])
            object_mpcat40 = object_meta[17]

            if object_raw_category not in raw_category_to_category_mapping:
                raw_category_to_category_mapping[object_raw_category] = object_category
            else:
                assert raw_category_to_category_mapping[object_raw_category]==object_category

            if object_category in category_to_mpcat40_mapping:
                assert category_to_mpcat40_mapping[object_category] == (object_mpcat40index, object_mpcat40)
            else:
                category_to_mpcat40_mapping[object_category] = (object_mpcat40index, object_mpcat40)
 

    return raw_category_to_category_mapping, category_to_mpcat40_mapping


def hex_color_to_category_map(SEMANTIC_SCENE_FILE_PATH):
    hex_color_to_category_map = {'000000': 'void'}
    with open(SEMANTIC_SCENE_FILE_PATH, "r") as sem_file:
        for line in sem_file.readlines():
            if line.startswith("HM3D Semantic Annotations"):
                continue
            
            object_id, object_hex_color, object_name, unknown = line.split(',')
            object_name = object_name.strip('"')

            # assert object_hex_color not in hex_color_to_category_map.keys()
            hex_color_to_category_map[object_hex_color] = object_name.strip('"')

    return hex_color_to_category_map


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

    

    rgb2hex = lambda r,g,b: '%02X%02X%02X' % (r,g,b)
    hex2rgb = lambda hex: tuple(int(h[i:i+2], 16) for i in (0, 2, 4))

    OUT_DIR = args.output_dir

    
    RAW_CATEGORY_TO_CATEGORY_MAPPING, CATEGORY_TO_MPCAT40_MAPPING = get_raw_category_to_mpcat40_map()
    if 'void' in CATEGORY_TO_MPCAT40_MAPPING:
        assert CATEGORY_TO_MPCAT40_MAPPING['void']==(0, 'void')
    else:
        CATEGORY_TO_MPCAT40_MAPPING['void'] = (0, 'void')

    obj_class_map = {}
    scene_directories = sorted([path for path in os.listdir(args.dataset_dir) if os.path.isdir(os.path.join(args.dataset_dir, path))])
    # scene_directories = ['00006-HkseAnWCgqk']
    tot_scenes = 0
    for scene_dir in scene_directories:
        scene_dir_path = os.path.join(args.dataset_dir, scene_dir)
        SCENE_DIR = scene_dir_path

        scene_view_poses_path = os.path.join(scene_dir_path, scene_dir+'.render_view_poses.csv')
        scene_has_sampled_views = os.path.isfile(scene_view_poses_path)

        SEMANTIC_SCENE_FILE_PATH = os.path.join(scene_dir_path, scene_dir+'.semantic.txt')
        if not os.path.isfile(SEMANTIC_SCENE_FILE_PATH):
            continue

        SCENE_VIEWS_FILE = os.path.join(scene_dir_path, scene_dir+'.render_view_poses.csv')

        tot_scenes+=1


        
        scene_hex_color_to_category_map = hex_color_to_category_map(SEMANTIC_SCENE_FILE_PATH)


        # found_in_cat_in_raw = 0
        # found_in_cat_not_raw = 0
        # found_not_cat_in_raw = 0
        # found_not_cat_not_raw = 0
        # for object_hex_color, object_name in hex_color_to_raw_category_map.items():
        #     if object_name in CATEGORY_TO_MPCAT40_MAPPING:
        #         if object_name in RAW_CATEGORY_TO_CATEGORY_MAPPING:
        #             found_in_cat_in_raw += 1
        #         else:
        #             found_in_cat_not_raw += 1
        #     else:
        #         if object_name in RAW_CATEGORY_TO_CATEGORY_MAPPING:
        #             found_not_cat_in_raw += 1
        #         else:
        #             found_not_cat_not_raw += 1
        # print("Found in both:", found_in_cat_in_raw)
        # print("Found in cat:", found_in_cat_not_raw)
        # print("Found in raw:", found_not_cat_in_raw)
        # print("Found neither:", found_not_cat_not_raw)
        # print()
            # if object_name not in RAW_CATEGORY_TO_CATEGORY_MAPPING:
            #     if object_name not in CATEGORY_TO_MPCAT40_MAPPING:
            #         not_found += 1
            #     else:
                    
            # if object_name not in CATEGORY_TO_MPCAT40_MAPPING and object_name not in RAW_CATEGORY_TO_CATEGORY_MAPPING:
            #     CATEGORY_TO_MPCAT40_MAPPING[object_name] = (0, 'void')

        
        extracting_sequence = None 

        with open(SCENE_VIEWS_FILE, 'r') as csvfile:

            pose_reader = csv.reader(csvfile, delimiter=',')

            for pose_idx, pose_meta in enumerate(pose_reader):
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
                    SEQ_NAME = '.'.join(extracting_sequence)
                    
                    OUT_RGB_DIR = os.path.join(OUT_DIR, "JPEGImages", SEQ_NAME)
                    OUT_SEM_DIR = os.path.join(OUT_DIR, "Annotations", SEQ_NAME)

                    os.makedirs(OUT_RGB_DIR, exist_ok=True)
                    os.makedirs(OUT_SEM_DIR, exist_ok=True)

                    object_ids_in_seq = set()

                    sequence_object_hex_color_to_id = {}
                    obj_class_map[SEQ_NAME] = {}


                rgb_file = f"{'.'.join(info_ID)}.RGB.{0:010}.png"
                rgb_image = Image.open(os.path.join(SCENE_DIR, rgb_file)).convert("RGB")
                rgb_image.save(os.path.join(OUT_RGB_DIR, view_id+".jpg"))


                sem_file = f"{'.'.join(info_ID)}.SEM.png"
                semantic_label_image = Image.open(os.path.join(SCENE_DIR, sem_file)).convert('RGB')
                semantic_label_image =  torchvision.transforms.functional.pil_to_tensor(semantic_label_image)
                if torch.cuda.is_available():
                    semantic_label_image = semantic_label_image.cuda()

                semantic_label_image_rgb_colors = torch.unique(semantic_label_image.flatten(1,2).T, dim=0, sorted=True)
                semantic_label_image_hex_colors = [rgb2hex(*(rgb_color.cpu())) for rgb_color in semantic_label_image_rgb_colors]

                for rgb_color, hex_color in zip(semantic_label_image_rgb_colors, semantic_label_image_hex_colors):
                    if hex_color not in scene_hex_color_to_category_map.keys() and hex_color!='000000':
                        mask = torch.all(semantic_label_image == rgb_color.reshape(3,1,1), dim=0)
                        semantic_label_image = torch.where(mask, 0, semantic_label_image)

                semantic_label_image_rgb_colors = torch.unique(semantic_label_image.flatten(1,2).T, dim=0, sorted=True)
                semantic_label_image_hex_colors = [rgb2hex(*(rgb_color.cpu())) for rgb_color in semantic_label_image_rgb_colors]
                semantic_label_image_object_names = [scene_hex_color_to_category_map[object_hex_color] for object_hex_color in semantic_label_image_hex_colors]
                semantic_label_mask = torch.all(semantic_label_image.unsqueeze(0) == semantic_label_image_rgb_colors.reshape(-1,3,1,1), dim=1).bool()
                

                
                category_ids = []
                instance_ids = []
                for object_hex_color, object_name in zip(semantic_label_image_hex_colors, semantic_label_image_object_names):
                    if object_name in CATEGORY_TO_MPCAT40_MAPPING:
                        class_id = CATEGORY_TO_MPCAT40_MAPPING[object_name][0]
                        category_ids.append(class_id)
                        
                        if class_id!=0:
                            if object_hex_color not in sequence_object_hex_color_to_id:
                                sequence_object_hex_color_to_id[object_hex_color] = len(sequence_object_hex_color_to_id.keys())+1
                            instance_id = sequence_object_hex_color_to_id[object_hex_color]

                            obj_class_map[SEQ_NAME][str(instance_id)] = str(class_id)
                        else:
                            instance_id = 0
                        instance_ids.append(instance_id)

                    elif object_name in RAW_CATEGORY_TO_CATEGORY_MAPPING:
                        class_id = CATEGORY_TO_MPCAT40_MAPPING[RAW_CATEGORY_TO_CATEGORY_MAPPING[object_name]][0]
                        category_ids.append(class_id)
                        
                        if class_id!=0:
                            if object_hex_color not in sequence_object_hex_color_to_id:
                                sequence_object_hex_color_to_id[object_hex_color] = len(sequence_object_hex_color_to_id.keys())+1
                            instance_id = sequence_object_hex_color_to_id[object_hex_color]

                            obj_class_map[SEQ_NAME][str(instance_id)] = str(class_id)
                        else:
                            instance_id = 0
                        instance_ids.append(instance_id)
                    else:
                        class_id = 0
                        category_ids.append(class_id)

                        instance_id = 0
                        instance_ids.append(instance_id)

                if max(instance_ids)>255:
                    raise

                panomask = torch.sum(torch.tensor(instance_ids).reshape(-1,1,1).to(semantic_label_mask.device) * semantic_label_mask, dim=0).to(torch.uint8).cpu()
                panomask = Image.fromarray(np.array(panomask).astype(np.uint8))
                panomask.save(os.path.join(OUT_SEM_DIR, view_id+".png"))

    with open(os.path.join(OUT_DIR, "obj_class.json"), "w") as outfile:
        json.dump(obj_class_map, outfile)
                

    if args.verbose:
        print()
        print("***********************")
        print(f"DONE RENDERING ALL SCENES")
        print("***********************")
        print()