#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys
import torch
pyt_version_str=torch.__version__.split("+")[0].replace(".", "")
version_str="".join([
    f"py3{sys.version_info.minor}_cu",
    torch.version.cuda.replace(".",""),
    f"_pyt{pyt_version_str}"
])
get_ipython().system('pip install fvcore iopath')
get_ipython().system('pip install --no-index --no-cache-dir pytorch3d -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/{version_str}/download.html')


# In[1]:


import os
import cv2
from PIL import Image
import open3d as o3d
import matplotlib.pyplot as plt

import torch
import numpy as np


# In[2]:


root = "/media/topipari/0CD418EB76995EEF/ZeroShotSceneSegmentation/zero_shot_scene_segmentation/datasets/raw_data/trajectory_renders/train/00009-vLpv2VX547B"
files = os.listdir(root)
files[:10]


# In[3]:


dep_path = os.path.join(root, '00009-vLpv2VX547B.0000000000.0000000100.0000000000.DEPTH.png')
rgb_path = os.path.join(root, '00009-vLpv2VX547B.0000000000.0000000100.0000000000.RGB.0000000000.png')
sem_path = os.path.join(root, '00009-vLpv2VX547B.0000000000.0000000100.0000000000.SEM.png')


# In[4]:


dep_image = cv2.imread(dep_path, cv2.IMREAD_UNCHANGED)
rgb_image = Image.open(rgb_path).convert('RGB')
sem_image = Image.open(sem_path).convert('RGB')

plt.imshow(dep_image)
plt.show()
plt.imshow(rgb_image)
plt.show()
plt.imshow(sem_image)
plt.show()


# In[5]:


rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
                            o3d.geometry.Image(np.asarray(rgb_image)), 
                            o3d.geometry.Image(np.asarray(dep_image)),
                            convert_rgb_to_intensity=False)


# In[7]:


pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
    rgbd_image, 
    o3d.camera.PinholeCameraIntrinsic(o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault))


# In[33]:


o3d.visualization.draw_geometries([pcd])


# In[29]:


o3d.camera.PinholeCameraIntrinsic(o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault).intrinsic_matrix


# In[93]:


import torch

from pytorch3d.structures import Pointclouds
from pytorch3d.vis.plotly_vis import AxisArgs, plot_batch_individually, plot_scene
from pytorch3d.renderer import (
    look_at_view_transform,
    look_at_rotation,
    PerspectiveCameras, 
    PointsRasterizationSettings,
    PointsRenderer,
    PulsarPointsRenderer,
    PointsRasterizer,
    AlphaCompositor,
    NormWeightedCompositor
)


# In[94]:


def inverse_projection(K, depth_map):
    assert K.shape==(3,3)
    assert depth_map.ndim==3 and depth_map.shape[2]==1
    
    height, width, _ = depth_map.shape
    grid_h, grid_w = torch.meshgrid(torch.arange(height), torch.arange(width), indexing='ij')
    
    pixel_coordinates_homogeneous_map = torch.stack((grid_w, grid_h, torch.ones_like(grid_h)), dim=-1)
    point_projections_nonhomogeneous_map = pixel_coordinates_homogeneous_map * depth_map
    point_projections_nonhomogeneous = point_projections_nonhomogeneous_map.reshape(-1, 3)
    
    points_camera_frame_nonhomogeneous = torch.matmul(point_projections_nonhomogeneous, torch.inverse(K).T)
    points_camera_frame_nonhomogeneous_map = points_camera_frame_nonhomogeneous.reshape(height, width, 3)
    
    return points_camera_frame_nonhomogeneous_map

def transform_points(RT, points_world_frame_nonhomogeneous_map):
    assert RT.shape==(3,4)
    
    height, width, _ = points_world_frame_nonhomogeneous_map.shape
    points_world_frame_homogeneous_map = torch.cat((points_world_frame_nonhomogeneous_map, torch.ones(height,width,1)), dim=2)
    points_camera_frame_nonhomogeneous = torch.matmul(points_world_frame_homogeneous_map.reshape(-1,4), RT.T)
    points_camera_frame_nonhomogeneous_map = points_camera_frame_nonhomogeneous.reshape(height, width, 3)
    return points_camera_frame_nonhomogeneous_map

def projection(K, points_camera_frame_nonhomogeneous_map):
    assert K.shape==(3,3)
    
    height, width, _ = points_camera_frame_nonhomogeneous_map.shape
    points_camera_frame_nonhomogeneous = points_camera_frame_nonhomogeneous_map.reshape(-1,3)
    point_projections_nonhomogeneous = torch.matmul(points_camera_frame_nonhomogeneous, K.T)
    point_projections_nonhomogeneous_map = point_projections_nonhomogeneous.reshape(height, width, 3)
    pixel_coordinates_map = point_projections_nonhomogeneous_map / point_projections_nonhomogeneous_map[:,:,2].reshape(height, width, 1)
    
    pixel_coordinates_map = pixel_coordinates_map[:,:,:2]
    
    return pixel_coordinates_map


# In[95]:


dep_tensor = torch.tensor(dep_image.astype(np.float32)/1000.0)
rgb_tensor = torch.tensor(np.array(rgb_image).astype(np.float32)/255.0)


# In[96]:


plt.imshow(dep_tensor)
plt.show()
plt.imshow(rgb_tensor)
plt.show()


# In[ ]:





# In[ ]:





# In[97]:


for i in range(5):
    K = torch.tensor([[465.6029, 0.0,      320.00],
                      [0.0,      465.6029, 240.00],
                      [0.0,      0.0,      1.0]])
    RT = torch.eye(4)[:3]
    RT[0,3]=0
    RT[1,3]=0
    RT[2,3]=0
    
    

    points_camera_frame_nonhomogeneous_map = inverse_projection(K, dep_tensor.unsqueeze(2))
    points_camera_frame_nonhomogeneous_map_ = transform_points(RT,points_camera_frame_nonhomogeneous_map)
    pixel_coordinates_map = projection(K, points_camera_frame_nonhomogeneous_map_)

    out_dep = torch.zeros_like(dep_tensor)
    out_rgb = torch.zeros_like(rgb_tensor)

    indices = pixel_coordinates_map.reshape(-1,2).round().long()
    indices[indices[:,0]<0, 0]=0
    indices[indices[:,0]>639, 0]=639
    indices[indices[:,1]<0, 1]=0
    indices[indices[:,1]>479, 1]=479
    out_rgb = out_rgb.index_put_((indices[:,1], indices[:,0]), rgb_tensor.reshape(-1,3))

    plt.imshow(out_rgb)
    plt.show()


# In[ ]:





# In[98]:


points= points_camera_frame_nonhomogeneous_map_.reshape(-1,3)#*torch.tensor([[-1,-1,1]])
point_cloud = Pointclouds(points=[points], features=[rgb_tensor.reshape(-1,3)])
point_cloud = point_cloud.cuda()


# In[121]:


R, T = look_at_view_transform(20, 10, 0)

# cameras = PerspectiveCameras(device='cpu', R=RT[:,:3].unsqueeze(0), T=RT[:,3].unsqueeze(0), znear=[0.01], fov=[69], aspect_ratio=[640/480.0], degrees=True)
cameras = PerspectiveCameras(R=RC, focal_length=((465.6029,465.6029),), image_size=((480,640),),principal_point=((320,240),), in_ndc=False, device='cuda')


# Define the settings for rasterization and shading. Here we set the output image to be of size
# 512x512. As we are rendering images for visualization purposes only we will set faces_per_pixel=1
# and blur_radius=0.0. Refer to raster_points.py for explanations of these parameters. 
raster_settings = PointsRasterizationSettings(
    image_size=(480,640), 
    radius = 0.007,
    points_per_pixel = 10
)


# Create a points renderer by compositing points using an alpha compositor (nearer points
# are weighted more heavily). See [1] for an explanation.
rasterizer = PointsRasterizer(cameras=cameras, raster_settings=raster_settings)
renderer = PointsRenderer(
    rasterizer=rasterizer,
    compositor=AlphaCompositor()
)


# In[122]:


for i in range(5):
    K = torch.tensor([[465.6029, 0.0,      320.00],
                      [0.0,      465.6029, 240.00],
                      [0.0,      0.0,      1.0]])
    RT = torch.eye(4)[:3]
    RT[0,3]=0
    RT[1,3]=0
    RT[2,3]=-0.05*i
    
    

    points_camera_frame_nonhomogeneous_map = inverse_projection(K, dep_tensor.unsqueeze(2))
    points_camera_frame_nonhomogeneous_map_ = transform_points(RT,points_camera_frame_nonhomogeneous_map)
    
    points= points_camera_frame_nonhomogeneous_map_.reshape(-1,3)#*torch.tensor([[-1,-1,1]])
    point_cloud = Pointclouds(points=[points], features=[rgb_tensor.reshape(-1,3)])
    point_cloud = point_cloud.cuda()
    
    
    images = renderer(point_cloud)
    plt.figure(figsize=(10, 10))
    plt.imshow(images[0, ..., :3].cpu().numpy())
    plt.axis("off");


# In[76]:


cameras.get_projection_transform().get_matrix().transpose(1,2)[0]


# In[47]:


K


# In[54]:


KK=torch.tensor([[465.6029,   0.0000, 320.0000, 0.0000],
              [  0.0000, 465.6029, 240.0000, 0.0000],
              [  0.0000,   0.0000,   1.0000, 0.0000],
              [  0.0000,   0.0000,   0.0000, 1.0000]]).unsqueeze(0)


# In[56]:


KK.transpose(1, 2)


# In[79]:


plt.imshow(points_camera_frame_nonhomogeneous_map_)


# In[88]:


points_camera_frame_nonhomogeneous_map_[-5:,-5:,0]


# In[104]:


import math
RC=torch.tensor([[math.cos(math.pi), -math.sin(math.pi),0],
             [math.sin(math.pi), math.cos(math.pi), 0],
             [0,0,1]]).unsqueeze(0)


# In[ ]:




