import os
import math
import numpy as np
import itertools, functools, operator


import bpy
import bmesh
from mathutils import Vector, Euler







def bounding_box(ob_name, coords, edges=[], faces=[]):
    """Create mesh object representing object bounding boxes.

    Keyword arguments:
    ob_name -- new object name
    coords -- float triplets eg: [(-1.0, 1.0, 0.0), (-1.0, -1.0, 0.0)]
    edges -- int pairs eg: [(0,1), (0,2)]
    """

    # Create new mesh and a new object
    me = bpy.data.meshes.new(ob_name + "Mesh")
    ob = bpy.data.objects.new(ob_name, me)

    # Make a mesh from a list of vertices/edges/faces
    me.from_pydata(coords, edges, faces)

    # Display name and update the mesh
    ob.show_name = True
    me.update()
    return ob


def get_mesh_aabb(mesh_obj):
    """Calculate axis-aligned bounding box around mesh object in world frame.

    Keyword arguments:
    mesh_obj -- mesh object in blender scene
    """
    
    # Transform local corners to world coordinate frame
    bbox_corners = np.array([mesh_obj.matrix_world @ Vector(corner) for corner in mesh_obj.bound_box])
    
    # Calculate bounding box in axis-aligned format
    bbox_min = np.min(bbox_corners,axis=0)
    bbox_max = np.max(bbox_corners,axis=0)
    
    bound_x = [bbox_min[0],bbox_max[0]]
    bound_y = [bbox_min[1],bbox_max[1]]
    bound_z = [bbox_min[2],bbox_max[2]]
    
    # Return corners of boundning box
    aabb_corners = [Vector((x,y,z)) for x,y,z in itertools.product(bound_x, bound_y, bound_z)]
    
    return aabb_corners



def aabb_3d_inter_areas(aabb1, aabb2):
    """Calculate 3D IOU from two axis-aligned bounding boxes.

    Keyword arguments:
    aabb1 -- list of float triplets representing first box corners
    aabb2 -- list of float triplets representing second box corners
    """
    aabb1, aabb2 = np.array(aabb1), np.array(aabb2)
    
    aabb1_min, aabb1_max = np.min(aabb1,axis=0), np.max(aabb1,axis=0)
    aabb2_min, aabb2_max = np.min(aabb2,axis=0), np.max(aabb2,axis=0)
    
    aabb1_area = np.prod(aabb1_max - aabb1_min)
    aabb2_area = np.prod(aabb2_max - aabb2_min)
    
    if np.any(aabb1_min>aabb2_max) or np.any(aabb1_max<aabb2_min):
        return 0.0, aabb1_area, aabb2_area
        
    inter_min = np.maximum(aabb1_min, aabb2_min)
    inter_max = np.minimum(aabb1_max, aabb2_max)
    
    inter_area = np.prod(inter_max - inter_min)

    iou = inter_area / aabb2_area#(aabb1_area + aabb2_area - inter_area)
    
    assert iou>=0.0 and iou<=1.0
    return inter_area, aabb1_area, aabb2_area

def merge_aabb(aabb1, aabb2):
    """Combine two axis-algined bounding boxes into one.

    Keyword arguments:
    aabb1 -- list of float triplets representing first box corners
    aabb2 -- list of float triplets representing second box corners
    """
    aabb1, aabb2 = np.array(aabb1), np.array(aabb2)
    
    aabb1_min, aabb1_max = np.min(aabb1,axis=0), np.max(aabb1,axis=0)
    aabb2_min, aabb2_max = np.min(aabb2,axis=0), np.max(aabb2,axis=0)
    
    aabb_min = np.minimum(aabb1_min, aabb2_min)
    aabb_max = np.maximum(aabb1_max, aabb2_max)
    
    bound_x = [aabb_min[0],aabb_max[0]]
    bound_y = [aabb_min[1],aabb_max[1]]
    bound_z = [aabb_min[2],aabb_max[2]]
    
    aabb_corners = [Vector((x,y,z)) for x,y,z in itertools.product(bound_x, bound_y, bound_z)]
    
    return aabb_corners

        
def get_collection_aabb(collection):
    """Calculate axis-aligned bounding box around collection of meshes.

    Keyword arguments:
    collection -- blender collection object
    """
    assert len(collection.all_objects)>0
    
    aabb = get_mesh_aabb(collection.all_objects[0])
    for i, obj_i in enumerate(collection.all_objects):
        aabb_i = get_mesh_aabb(obj_i)
        aabb = merge_aabb(aabb, aabb_i)
    return aabb



def get_grid_points(aabb_bounds, samples_per_meter=1, margin_pcnt=0.025):
    """Calculate uniform grid of 3D location samples.

    Keyword arguments:
    aabb_bounds -- list of float tripliets representing corners of 3D boundary for samples
    samples_per_meter -- float for density of samples in each dimension
    margin_pcnt -- float representing inner margin on sampling bounds as percent of smallest boundary dimension
    """
    
    aabb_bounds = np.array(aabb_bounds)
    
    bounds_min = np.min(aabb_bounds, axis=0)
    bounds_max = np.max(aabb_bounds, axis=0)
    bounds_range = bounds_max-bounds_min
    
    margin = margin_pcnt * np.min(bounds_range)
    
    bounds_min += margin
    bounds_max -= margin
    bounds_range -= 2 * margin
    
    num_samples = np.floor(bounds_range * samples_per_meter).astype(dtype=np.int32)
    
    return np.linspace(bounds_min[0], bounds_max[0], num=num_samples[0], endpoint=True), \
            np.linspace(bounds_min[1], bounds_max[1], num=num_samples[1], endpoint=True), \
            np.linspace(bounds_min[2], bounds_max[2], num=num_samples[2], endpoint=True)

def get_grid_euler(roll_bounds=(math.radians(145),math.radians(225)), roll_samples=3, 
                   pitch_bounds=(math.radians(-20),math.radians(20)), pitch_samples=3, 
                   yaw_bounds=(0,2*math.pi-(2*math.pi/8)), yaw_samples=8):
    """Calculate uniform grid of samples in euler angle rotations.

    Keyword arguments:
    roll_bounds -- float pairs of min and max roll angles as radians
    roll_samples -- samples per degree in roll (Z)
    
    pitch_bounds -- float pairs of min and max roll angles as radians
    pitch_samples -- samples per degree in pitch (X)
    
    yaw_bounds -- float pairs of min and max roll angles as radians
    yaw_samples -- samples per degree in yaw (Y)
    """
    
    return np.linspace(roll_bounds[0], roll_bounds[1], num=roll_samples, endpoint=True), \
            np.linspace(pitch_bounds[0], pitch_bounds[1], num=pitch_samples, endpoint=True), \
            np.linspace(yaw_bounds[0], yaw_bounds[1], num=yaw_samples, endpoint=True)
            
def euclidean_distance(v1, v2):
    """Calculate euclidean distance between vectors.

    Keyword arguments:
    v1 -- 3D Vector
    v2 -- 3D Vector
    """
    diff = v1 - v2
    return math.sqrt(diff.x**2 + diff.y**2 + diff.z**2)



def get_sphere(pos, rot, name='Basic_Sphere', mat=None):
    sphere_mesh = bpy.data.meshes.get(name)
    if sphere_mesh is None:
        sphere_mesh = bpy.data.meshes.new(name)
        
        bm = bmesh.new()
        bmesh.ops.create_uvsphere(bm, u_segments=32, v_segments=16, radius=0.1)
        bm.to_mesh(sphere_mesh)
        bm.free()
        
        
    sphere_obj = bpy.data.objects.new(name, sphere_mesh)
    sphere_obj.location = Vector(pos)
    sphere_obj.rotation_mode = 'ZXY'
    sphere_obj.rotation_euler = Euler(rot)
    sphere_obj.scale = Vector((1,1,1))
    if mat is not None:
        sphere_obj.data.materials.append(mat)
    
    bpy.context.collection.objects.link(sphere_obj)
    
#    bpy.ops.object.select_all(action='DESELECT')
#    bpy.context.view_layer.objects.active = sphere_obj
#    sphere_obj.select_set(True)
#    bpy.ops.object.modifier_add(type='SUBSURF')
#    bpy.ops.object.shade_smooth()
    
    return sphere_obj

def get_camera(pos, rot, name="Camera_Sample", lens=15, clip_start=1e-2, scale=(1,1,1)):
    camera = bpy.data.cameras.get(name)
    if camera is None:
        camera = bpy.data.cameras.new(name)
        camera.lens = lens
        camera.clip_start = clip_start
    camera_sample_obj = bpy.data.objects.new("Camera", camera)
    camera_sample_obj.location = Vector(pos)
    camera_sample_obj.rotation_mode = 'ZXY'
    camera_sample_obj.rotation_euler = Euler(rot)
    camera_sample_obj.scale = scale
    bpy.context.collection.objects.link(camera_sample_obj)
#    print("added camera")
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


def camera_viewing_valid_surface(camera_obj, dist_threshold=0.25):
    camera_center_and_corner_dirs = [((camera_obj.matrix_world @ corner) - camera_obj.location).normalized() for corner in [Vector((0,0,-1))]+list(camera_obj.data.view_frame(scene=bpy.context.scene))]
    
    camera_center_and_corner_ray_casts = [bpy.context.scene.ray_cast(bpy.context.view_layer.depsgraph, camera_obj.location, camera_dir) for camera_dir in camera_center_and_corner_dirs]
    camera_center_and_corner_normal_dots = [corner_dir.dot(ray_cast[2].normalized()) for (corner_dir, ray_cast) in zip(camera_center_and_corner_dirs, camera_center_and_corner_ray_casts)]
    camera_center_and_corner_dists = [euclidean_distance(ray_cast[1], camera_obj.location) for ray_cast in camera_center_and_corner_ray_casts]
    
    # Valid view if all corners and center ray hit inner mesh of scene
    all_rays_hit_surface = all([ray_cast[0] for ray_cast in camera_center_and_corner_ray_casts])
    all_rays_hit_inner_mesh = all([dot<0 for dot in camera_center_and_corner_normal_dots])
    all_rays_hit_within_dist_threshold = all([dist>=dist_threshold for dist in camera_center_and_corner_dists])
    
    return all_rays_hit_surface and all_rays_hit_inner_mesh and all_rays_hit_within_dist_threshold



def clear_scene():
    
    for c in bpy.context.scene.collection.children:
        bpy.context.scene.collection.children.unlink(c)
    
    for block in bpy.data.meshes:
        if block.users == 0:
            bpy.data.meshes.remove(block)

    for block in bpy.data.materials:
        if block.users == 0:
            bpy.data.materials.remove(block)

    for block in bpy.data.textures:
        if block.users == 0:
            bpy.data.textures.remove(block)

    for block in bpy.data.images:
        if block.users == 0:
            bpy.data.images.remove(block)


            
def render_scene_images(SCENE_DIR, OUTPUT_DIR):
    
    SCENE_NAME = SCENE_DIR.split('/')[-1].split('-')[-1]
    SCENE_FILE = SCENE_NAME+'.glb'
    SEMANTIC_SCENE_FILE = SCENE_NAME+'.semantic.glb'
    
    print()
    print("********************")
    print(f"SAMPLING VIEWS FOR SCENE: {SCENE_NAME}")
    print("********************")
    print()

    
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print()
    print("********************")
    print("RESETTING SCENE")

    clear_scene()
    
    general_collection = bpy.data.collections.get("Collection")
    if general_collection is None:
        general_collection = bpy.data.collections.new("Collection")
        bpy.context.scene.collection.children.link(general_collection)

    delete_object(bpy.data.objects.get("Camera"))
    camera = bpy.data.cameras.new("Camera")
    camera.lens = 15
    camera.clip_start = 1e-2
    camera_obj = bpy.data.objects.new("Camera", camera)
    camera_obj.location = Vector((0,0,0))
    camera_obj.rotation_mode = 'ZXY'
    camera_obj.rotation_euler = Euler((0,0,math.pi))
    add_object_to_collection(camera_obj, general_collection)
    bpy.context.scene.camera = camera_obj


    building_collection = bpy.data.collections.get("Building")
    delete_collection(building_collection)
    
    semantic_building_collection = bpy.data.collections.get("Semantic_Building")
    delete_collection(semantic_building_collection)

    delete_object(bpy.data.objects.get("Building_Box"))

    initial_sample_collection = bpy.data.collections.get("Initial_Sample_Grid")
    delete_collection(initial_sample_collection)

    grid_post_coll = bpy.data.collections.get("Accepted_Sample_Grid")
    delete_collection(grid_post_coll)

    print("DONE RESETTING SCENE")
    print("********************")
    print()




    print()
    print("***********************")
    print("INITIALIZING SCENE")
    
    bpy.ops.import_scene.gltf(filepath=os.path.join(SCENE_DIR,SCENE_FILE))

    building_collection = bpy.data.collections.get("Building")
    if building_collection is None:
        building_collection = bpy.data.collections.new("Building")
        bpy.context.scene.collection.children.link(building_collection)
    
    bpy.ops.object.select_all(action='DESELECT')
    
    for obj in bpy.data.objects:
        if obj.type=="MESH":
            add_object_to_collection(obj, building_collection)
            for mat in obj.data.materials:
                mat.use_backface_culling = False
    

    bpy.ops.import_scene.gltf(filepath=os.path.join(SCENE_DIR,SEMANTIC_SCENE_FILE))
    
    semantic_building_collection = bpy.data.collections.get("Semantic_Building")
    if semantic_building_collection is None:
        semantic_building_collection = bpy.data.collections.new("Semantic_Building")
        bpy.context.scene.collection.children.link(semantic_building_collection)

    for obj in bpy.data.objects:
        if obj.type=="MESH" and building_collection not in obj.users_collection:
            add_object_to_collection(obj, semantic_building_collection)
            for mat in obj.data.materials:
                mat.use_backface_culling = False
    
    print("DONE INITIALIZING SCENE")
    print("***********************")
    print()


        
    
    
    building_aabb = get_collection_aabb(building_collection)


    grid_x, grid_y, grid_z = get_grid_points(building_aabb, samples_per_meter=0.5)
    num_pos_samples = functools.reduce(operator.mul, map(len, (grid_x, grid_y, grid_z)), 1)

    grid_roll, grid_pitch, grid_yaw = get_grid_euler(roll_bounds=(math.radians(180),math.radians(180)), roll_samples=1, 
                                                       pitch_bounds=(math.radians(-20),math.radians(20)), pitch_samples=3, 
                                                       yaw_bounds=(0,2*math.pi-(2*math.pi/8)), yaw_samples=8)
    num_rot_samples = functools.reduce(operator.mul, map(len, (grid_roll, grid_pitch, grid_yaw)), 1)
    

    bpy.context.scene.render.film_transparent = True
    bpy.context.scene.render.image_settings.color_mode = 'RGBA'
    bpy.context.scene.render.resolution_x = 960
    bpy.context.scene.render.resolution_y = 720

    
    meta_file = open(os.path.join(OUTPUT_DIR, f"{SCENE_NAME}_meta.csv"),"w")
            
    
    img_i = 0

    # Iterate over uniform grid of positions within scene bounding box
    for pos_i, (x,y,z) in enumerate(itertools.product(grid_x, grid_y, grid_z)):
        
        # Set camera position
        camera_obj.location = Vector((x,y,z))
        
        # Iterate over uniform grid of rotations
        for rot_i,(roll,pitch,yaw) in enumerate(itertools.product(grid_roll, grid_pitch, grid_yaw)):

            # Set camera rotation
            camera_obj.rotation_euler = Euler((pitch,yaw,roll))

            # Update scene view layer to recalculate camera extrensic matrix
            bpy.context.view_layer.update()
            
            # Determine if rejection sampling criteria is met
            # Valid view defined as one where corner and center rays view inner mesh surface and at least 0.25m from camera origin
            if camera_viewing_valid_surface(camera_obj, dist_threshold=0.25):
                meta_file.write(f'{img_i:010},{pos_i:010},{rot_i:010},{x},{y},{z},{roll},{pitch},{yaw}\n')
                meta_file.flush()
                
                img_i += 1

            if rot_i%10==0:
                print(f'    {rot_i}/{num_rot_samples} rotation samples finished rendering')
            
        if pos_i%10==0:
            print(f'{pos_i}/{num_pos_samples} position samples finished rendering')
    
    meta_file.close()



            
SCENE_DIR = '/media/topipari/0CD418EB76995EEF/SegmentationProject/zeroshot_rgbd/datasets/matterport/HM3D/example/00861-GLAQ4DNUx5U'
OUTPUT_DIR = '/media/topipari/0CD418EB76995EEF/SegmentationProject/zeroshot_rgbd/datasets/renders'







render_scene_images(SCENE_DIR, OUTPUT_DIR)


