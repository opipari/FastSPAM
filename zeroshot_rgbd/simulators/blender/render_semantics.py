import os
import sys
import argparse
import configparser
import csv

import math
import numpy as np
import itertools, functools, operator


import bpy
import bmesh
from mathutils import Vector, Euler




##############################################################################
#                             BLENDER UTILITIES                              #
##############################################################################


def get_camera(pos, rot, name="Camera_Sample", rot_mode='ZXY', lens=15, clip_start=1e-2, scale=(1,1,1)):
    camera = bpy.data.cameras.get(name)
    if camera is None:
        camera = bpy.data.cameras.new(name)
        camera.lens = lens
        camera.clip_start = clip_start
    
    camera_sample_obj = bpy.data.objects.new("Camera", camera)
    camera_sample_obj.location = Vector(pos)
    camera_sample_obj.rotation_mode = rot_mode
    camera_sample_obj.rotation_euler = Euler(rot)
    camera_sample_obj.scale = scale

    return camera_sample_obj


def delete_object(obj):
    if obj is not None:
        bpy.ops.object.delete({"selected_objects": [obj]})


def delete_collection(collection):
    if collection is not None:
        for obj in collection.objects:
            bpy.data.objects.remove(obj, do_unlink=True)


def add_object_to_collection(object, collection):
    for coll in object.users_collection:
        coll.objects.unlink(object)
    collection.objects.link(object)


##############################################################################
#                         END OF BLENDER UTILITIES                           #
##############################################################################


def render_scene_semantics(SCENE_DIR, SCENE_VIEWS_FILE, SCENE_OUT_DIR, ARGS):
    
    SCENE_NAME = SCENE_DIR.split('/')[-1].split('-')[-1]
    SCENE_FILE = SCENE_NAME+'.glb'
    SEMANTIC_SCENE_FILE = SCENE_NAME+'.semantic.glb'
    
    if ARGS.verbose:
        print()
        print("********************")
        print(f"SAMPLING VIEWS FOR SCENE: {SCENE_NAME}")
        print("********************")
        print()

    os.makedirs(SCENE_OUT_DIR, exist_ok=True)

    if ARGS.verbose:
        print()
        print("********************")
        print("RESETTING SCENE")

    
    general_collection = bpy.context.scene.collection

    delete_object(bpy.data.objects.get("Camera"))
    camera_obj = get_camera(pos=(0,0,0), rot=(0,0,math.pi), name="Camera", rot_mode='ZXY', lens=ARGS.camera_lens, clip_start=ARGS.camera_clip)
    add_object_to_collection(camera_obj, general_collection)
    bpy.context.scene.camera = camera_obj

    semantic_building_collection = bpy.data.collections.get("Semantic_Building")
    delete_collection(semantic_building_collection)


    bpy.context.scene.render.film_transparent = True
    bpy.context.scene.render.image_settings.color_mode = 'RGBA'
    bpy.context.scene.render.resolution_x = ARGS.camera_width
    bpy.context.scene.render.resolution_y = ARGS.camera_height

    if ARGS.verbose:
        print("DONE RESETTING SCENE")
        print("********************")
        print()




    if ARGS.verbose:
        print()
        print("***********************")
        print("INITIALIZING SCENE")
    
    bpy.ops.import_scene.gltf(filepath=os.path.join(SCENE_DIR,SEMANTIC_SCENE_FILE))

    semantic_building_collection = bpy.data.collections.get("Semantic_Building")
    if semantic_building_collection is None:
        semantic_building_collection = bpy.data.collections.new("Semantic_Building")
        bpy.context.scene.collection.children.link(semantic_building_collection)
    
    bpy.ops.object.select_all(action='DESELECT')
    
    for obj in bpy.data.objects:
        if obj.type=="MESH":
            add_object_to_collection(obj, semantic_building_collection)
            for mat in obj.data.materials:
                mat.use_backface_culling = False
                if mat.node_tree:
                    for node in mat.node_tree.nodes:
                        if node.type == 'TEX_IMAGE':
                            node.interpolation = 'Closest'


    semantic_building_collection.hide_render=False
    bpy.context.scene.render.engine = 'BLENDER_WORKBENCH'
    bpy.context.scene.display.shading.light = 'FLAT'
    bpy.context.scene.display.shading.color_type = 'TEXTURE'
    bpy.context.scene.render.dither_intensity = 0.0
    bpy.context.scene.display.render_aa = 'OFF'
    bpy.context.scene.view_settings.view_transform = 'Standard'


    if ARGS.verbose:
        print("DONE INITIALIZING SCENE")
        print("***********************")
        print()

        
    

    



    if ARGS.verbose:
        print()
        print("***********************")
        print(f"INITIATING RENDERING")


    render_image_count = 0

    with open(SCENE_VIEWS_FILE, 'r') as csvfile:

        pose_reader = csv.reader(csvfile, delimiter=',')

        for pose_meta in pose_reader:
            scene_name, view_idx, pos_idx, rot_idx, x_pos, y_pos, z_pos, roll, pitch, yaw = pose_meta
            
            # Skip information line if it is first
            if scene_name=='Scene-ID':
                continue

            # Parse pose infomration out of string type
            view_idx, pos_idx, rot_idx = int(view_idx), int(pos_idx), int(rot_idx)
            x_pos, y_pos, z_pos = float(x_pos), float(y_pos), float(z_pos)
            roll, pitch, yaw = float(roll), float(pitch), float(yaw)

            # Set camera position
            camera_obj.location = Vector((x_pos, y_pos, z_pos))

            # Set camera rotation
            camera_obj.rotation_euler = Euler((pitch,yaw,roll))

            # Update scene view layer to recalculate camera extrensic matrix
            bpy.context.view_layer.update()

            bpy.context.scene.render.filepath = os.path.join(SCENE_OUT_DIR, f'{SCENE_NAME}.{view_idx:010}.{pos_idx:010}.{rot_idx:010}.SEM.png')
            bpy.ops.render.render(write_still = True)

            render_image_count += 1
    
    if ARGS.verbose:
        print(f"DONE RENDERING {render_image_count} VIEWS")
        print("***********************")
        print()


if __name__ == "__main__":

    # Read command line arguments
    argv = sys.argv

    # Ignore Blender executable and Blender-specific command line arguments
    if "--" in argv:
        argv = argv[argv.index("--") + 1:]


    parser = argparse.ArgumentParser(
                    prog='sample_views',
                    usage='blender --background --python <path to sample_views.py> -- [options]',
                    description='Blender python script for using rejection sampling to uniformly sample valid views from the Matterport 3D semantic dataset',
                    epilog='For more information, see: https://github.com/opipari/ZeroShot_RGB_D/tree/main/zeroshot_rgbd/simulators/blender')


    parser.add_argument('-data', '--dataset-dir', help='path to directory of Matterport semantic dataset directory formatted as one sub-directory per scene', type=str)
    parser.add_argument('-out', '--output-dir', help='path to directory where output dataset should be stored', type=str)
    parser.add_argument('-v', '--verbose', help='whether verbose output printed to stdout', type=int, default=1)

    parser.add_argument('-lens', '--camera-lens', help='float controlling focal length of blender camera in (mm) units. default 15', type=float, default=15.0)
    parser.add_argument('-clip', '--camera-clip', help='float controlling distance of clip start of blender camera in (m) units. default 1e-2', type=float, default=1e-2)
    parser.add_argument('-width', '--camera-width', help='int controlling rendered image width in pixels. default 960', type=int, default=960)
    parser.add_argument('-height', '--camera-height', help='int controlling rendered image height in pixels. default 15', type=int, default=720)

    parser.add_argument('-pos-density', '--position-samples-per-meter', help='float controlling density of camera samples in 3D position. default 1 per meter', type=float, default=1)
    
    parser.add_argument('-roll-num', '--roll-samples-count', help='int controlling number of rotation samples for roll rotation (about forward -Z direction). default 1', type=int, default=1)
    parser.add_argument('-roll-min', '--roll-samples-minimum', help='float controlling minimum extent of roll samples for rotation. default math.radians(180)', type=float, default=math.radians(180))
    parser.add_argument('-roll-max', '--roll-samples-maximum', help='float controlling maximum extent of roll samples for rotation. default math.radians(180)', type=float, default=math.radians(180))

    parser.add_argument('-pitch-num', '--pitch-samples-count', help='int controlling number of rotation samples for pitch rotation (about sideways X direction). default 3', type=int, default=3)
    parser.add_argument('-pitch-min', '--pitch-samples-minimum', help='float controlling minimum extent of pitch samples for rotation. default math.radians(-20)', type=float, default=math.radians(-20))
    parser.add_argument('-pitch-max', '--pitch-samples-maximum', help='float controlling maximum extent of pitch samples for rotation. default math.radians(20)', type=float, default=math.radians(20))
    
    parser.add_argument('-yaw-num', '--yaw-samples-count', help='int controlling number of rotation samples for yaw rotation (about vertical Y direction). default 8', type=int, default=8)
    parser.add_argument('-yaw-min', '--yaw-samples-minimum', help='float controlling minimum extent of yaw samples for rotation. default math.radians(0)', type=float, default=0)
    parser.add_argument('-yaw-max', '--yaw-samples-maximum', help='float controlling maximum extent of yaw samples for rotation. default math.radians(360)-(math.radians(360)/8)', type=float, default=math.radians(360)-(math.radians(360)/8))

    args = parser.parse_args(argv)


    scene_directories = [path for path in os.listdir(args.dataset_dir) if os.path.isdir(os.path.join(args.dataset_dir, path))]
    for scene_dir in scene_directories:
        scene_dir_path = os.path.join(args.dataset_dir, scene_dir)

        scene_files = os.listdir(scene_dir_path)
        scene_name = scene_files[0].split('.')[0]
        assert scene_dir.endswith(scene_name)

        scene_out_path = os.path.join(args.output_dir, scene_name)
        scene_view_poses_path = os.path.join(args.output_dir, scene_name+'_accepted_view_poses.csv')

        scene_has_semantic_mesh = any([fl.endswith('.semantic.glb') for fl in scene_files])
        scene_has_semantic_txt = any([fl.endswith('.semantic.txt') for fl in scene_files])
        scene_has_sampled_views = os.path.isfile(scene_view_poses_path)

        if scene_has_semantic_mesh and scene_has_semantic_txt and scene_has_sampled_views:
            render_scene_semantics(scene_dir_path, scene_view_poses_path, scene_out_path, args)