import os
import math
import numpy as np
import itertools, functools, operator


import bpy
import bmesh
from mathutils import Vector, Euler



def render_scene_images(SCENE_DIR, OUTPUT_DIR, INITIALIZE_SCENE=True, VISUALIZE_GRID=False, RENDER_IMAGES=True):
    
    SCENE_NAME = SCENE_DIR.split('/')[-1].split('-')[-1]
    SCENE_FILE = SCENE_NAME+'.glb'
    SEMANTIC_SCENE_FILE = SCENE_NAME+'.semantic.glb'
    
    if ARGS.verbose:
        print()
        print("********************")
        print(f"SAMPLING VIEWS FOR SCENE: {SCENE_NAME}")
        print("********************")
        print()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

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

    if ARGS.verbose:
        print("DONE INITIALIZING SCENE")
        print("***********************")
        print()

        
    

        semantic_building_collection.hide_render=False
        building_collection.hide_render=True
        bpy.context.scene.render.engine = 'BLENDER_WORKBENCH'
        bpy.context.scene.display.shading.light = 'FLAT'
        bpy.context.scene.display.shading.color_type = 'TEXTURE'
        bpy.context.scene.render.dither_intensity = 0.0
        bpy.context.scene.display.render_aa = 'OFF'
        bpy.context.scene.view_settings.view_transform = 'Standard'
        img_i = 0
        for pos_i in sorted(valid_poses.keys()):
            x,y,z = grid_pos_idx[pos_i]
            camera_obj.location = Vector((x,y,z))
            for rot_i in valid_poses[pos_i]:
                roll,pitch,yaw = grid_rot_idx[rot_i]
                
                camera_obj.rotation_euler = Euler((pitch,yaw,roll))
                bpy.context.view_layer.update()
                
                bpy.context.scene.render.filepath = os.path.join(OUTPUT_DIR, f'{SCENE_NAME.split(".")[0]}.{img_i:010}.{pos_i:05}.{rot_i:05}.SEM.png')
                bpy.ops.render.render(write_still = True)
                img_i += 1
        
        



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


    SCENE_DIR = '/media/topipari/0CD418EB76995EEF/SegmentationProject/zeroshot_rgbd/datasets/matterport/HM3D/example/00861-GLAQ4DNUx5U'

    print(args)



    scene_directories = [path for path in os.listdir(args.dataset_dir) if os.path.isdir(os.path.join(args.dataset_dir, path))]
    for scene_dir in scene_directories:
        scene_dir_path = os.path.join(args.dataset_dir, scene_dir)

        scene_files = os.listdir(scene_dir_path)
        scene_name = scene_files[0].split('.')[0]
        assert scene_dir.endswith(scene_name)
        scene_has_semantic_mesh = any([fl.endswith('.semantic.glb') for fl in scene_files])
        scene_has_semantic_txt = any([fl.endswith('.semantic.txt') for fl in scene_files])
        
        if scene_has_semantic_mesh and scene_has_semantic_txt:
            sample_scene_views(scene_dir_path, args.output_dir, args)