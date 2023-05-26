import os
import math
import numpy as np
import itertools, functools, operator

import matplotlib.pyplot as plt

import bpy
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
    
    sample_step = bounds_range / (bounds_range * samples_per_meter)
    
    return np.arange(bounds_min[0], bounds_max[0]+1e-10, sample_step[0]), \
            np.arange(bounds_min[1], bounds_max[1]+1e-10, sample_step[1]), \
            np.arange(bounds_min[2], bounds_max[2]+1e-10, sample_step[2])

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
    
    return np.arange(roll_bounds[0], roll_bounds[1]+1e-10, (max(roll_bounds)-min(roll_bounds))/(roll_samples-1)), \
            np.arange(pitch_bounds[0], pitch_bounds[1]+1e-10, (max(pitch_bounds)-min(pitch_bounds))/(pitch_samples-1)), \
            np.arange(yaw_bounds[0], yaw_bounds[1]+1e-10, (max(yaw_bounds)-min(yaw_bounds))/(yaw_samples-1))
            
def euclidean_distance(v1, v2):
    """Calculate euclidean distance between vectors.

    Keyword arguments:
    v1 -- 3D Vector
    v2 -- 3D Vector
    """
    diff = v1 - v2
    return math.sqrt(diff.x**2 + diff.y**2 + diff.z**2)



def get_sphere(pos, name='Basic_Sphere', mat=None):
    sphere_mesh = bpy.data.meshes.new(name)
    sphere_obj = bpy.data.objects.new(name, sphere_mesh)
    sphere_obj.location = Vector(pos)
    sphere_obj.scale = Vector((0.1,0.1,0.1))
    if mat is not None:
        sphere_obj.data.materials.append(sphere_mat)
    return sphere_obj

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

SCENE_FILE = '/Users/top/Downloads/hm3d-example-glb-v0.2/00337-CFVBbU9Rsyb/CFVBbU9Rsyb.glb'
OUTPUT_DIR = '/Users/top/Downloads/hm3d-example-glb-v0.2/renders'






INITIALIZE_SCENE = True
VISUALIZE_GRID = False
RENDER_IMAGES = True


if RENDER_IMAGES:
    os.makedirs(OUTPUT_DIR, exist_ok=True)

print()
print("********************")
print("RESETTING SCENE")


general_collection = bpy.data.collections.get("Collection")
if general_collection is None:
    general_collection = bpy.data.collections.new("Collection")
    bpy.context.scene.collection.children.link(general_collection)

delete_object(bpy.data.objects.get("Camera"))
camera = bpy.data.cameras.new("Camera")
camera.lens = 15
camera.clip_start = 1e-6
camera_obj = bpy.data.objects.new("Camera", camera)
camera_obj.location = Vector((0,0,0))
camera_obj.rotation_mode = 'ZXY'
camera_obj.rotation_euler = Euler((0,0,math.pi))
add_object_to_collection(camera_obj, general_collection)
bpy.context.scene.camera = camera_obj

delete_object(bpy.data.objects.get("Spot_Light"))
spot_light = bpy.data.lights.new(name="Spot_Light", type='SPOT')
spot_light.energy = 50
spot_light.spot_size = math.pi
spot_light.distance = 25
spot_light_obj = bpy.data.objects.new(name="Spot_Light", object_data=spot_light)
spot_light_obj.location = Vector((0,0,0))
spot_light_obj.parent = camera_obj
add_object_to_collection(spot_light_obj, general_collection)



building_collection = bpy.data.collections.get("Building")
delete_collection(building_collection)

delete_object(bpy.data.objects.get("Building_Box"))

initial_sample_collection = bpy.data.collections.get("Initial_Sample_Grid")
delete_collection(initial_sample_collection)

grid_post_coll = bpy.data.collections.get("Accepted_Sample_Grid")
delete_collection(grid_post_coll)

print("DONE RESETTING SCENE")
print("********************")
print()




if INITIALIZE_SCENE:
    print()
    print("***********************")
    print("INITIALIZING SCENE")
    
    bpy.ops.import_scene.gltf(filepath=SCENE_FILE)

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
            obj.select_set(True)
            bpy.context.view_layer.objects.active = obj
    bpy.ops.object.join()
    
    print()
    print("***********************")
    print("DONE INITIALIZING SCENE")


    

num_building_meshes = len(building_collection.all_objects)
building_aabb = get_collection_aabb(building_collection)
building_box = bounding_box("Building_Box", building_aabb, edges=[(0,1),(0,2),(1,3),(2,3),(4,5),(4,6),(5,7),(6,7),(0,4),(1,5),(2,6),(3,7)])
building_collection.objects.link(building_box)


grid_x, grid_y, grid_z = get_grid_points(building_aabb, samples_per_meter=1)
num_pos_samples = functools.reduce(operator.mul, map(len, (grid_x, grid_y, grid_z)), 1)

grid_roll, grid_pitch, grid_yaw = get_grid_euler()
num_rot_samples = functools.reduce(operator.mul, map(len, (grid_roll, grid_pitch, grid_yaw)), 1)


if VISUALIZE_GRID:
    sphere_mat = bpy.data.materials.get("Sphere_Material")
    if sphere_mat is None:
        sphere_mat = bpy.data.materials.new(name="Sphere_Material")
        
    # Create collection for initial and post-rejection grid
    
    initial_sample_collection = bpy.data.collections.get("Initial_Sample_Grid")
    if initial_sample_collection is None:
        initial_sample_collection = bpy.data.collections.new("Initial_Sample_Grid")
        bpy.context.scene.collection.children.link(initial_sample_collection)
    
    grid_post_collection = bpy.data.collections.get("Accepted_Sample_Grid")
    if grid_post_collection is None:
        grid_post_collection = bpy.data.collections.new("Accepted_Sample_Grid")
        bpy.context.scene.collection.children.link(grid_post_collection)
    
    for x,y,z in itertools.product(grid_x, grid_y, grid_z):
        sphere_obj = get_sphere((x,y,z), mat=sphere_mat)
        bpy.context.collection.objects.link(sphere_obj)
        add_object_to_collection(sphere_obj, initial_sample_collection)
        
    
        
    
    



bpy.context.scene.render.resolution_x = 640
bpy.context.scene.render.resolution_y = 480

if RENDER_IMAGES:
    meta_file = open(os.path.join(OUTPUT_DIR,"meta.csv"),"w")



img_i = 0

for light in [5,10,25,50,100,200]:
    spot_light.energy = light
    for pos_i, (x,y,z) in enumerate(itertools.product(grid_x, grid_y, grid_z)):
        if pos_i<67:
            continue
        if pos_i>67:
            break
        has_valid_view = False
        camera_obj.location = Vector((x,y,z))
        for rot_i,(roll,pitch,yaw) in enumerate(itertools.product(grid_roll, grid_pitch, grid_yaw)):
            camera_obj.rotation_euler = Euler((pitch,yaw,roll))
            
            camera_loc = camera_obj.location
            camera_view_dir = camera_obj.matrix_world @ Vector((0,0,-1)) - camera_loc
            camera_view_dir = camera_view_dir.normalized()
            
            res, loc, normal, face_i, object, matrix = bpy.context.scene.ray_cast(bpy.context.view_layer, camera_loc, camera_view_dir)
            normal_dot = camera_view_dir.dot(normal.normalized())
            
            # Valid view if at least one rotation sample looks directly at a surface facing camera
            if normal_dot<0:
                has_valid_view = True
                
                if RENDER_IMAGES:
                    bpy.context.scene.render.filepath = os.path.join(OUTPUT_DIR, f'{scene}_{img_i:05}_{light:05}_{pos_i:05}_{rot_i:05}_RGB.jpg')
                    bpy.ops.render.render(write_still = True)
                    meta_file.write(f'{img_i:05},{light},{pos_i:05},{rot_i:05},{x},{y},{z},{roll},{pitch},{yaw}\n')
                    meta_file.flush()
                    
            if rot_i%10==0:
                print(f'    {rot_i}/{num_rot_samples} rotation samples finished rendering')
                
        if VISUALIZE_GRID and has_valid_view:
            sphere_obj = get_sphere((x,y,z), name="Accept", mat=sphere_mat)
            bpy.context.collection.objects.link(sphere_obj)
            add_object_to_collection(sphere_obj, grid_post_collection)
            
        if pos_i%10==0:
            print(f'{pos_i}/{num_pos_samples} position samples finished rendering')
        
if RENDER_IMAGES:       
    meta_file.close()


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
        
        