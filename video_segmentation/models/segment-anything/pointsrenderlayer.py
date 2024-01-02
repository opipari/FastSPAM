from typing import Optional, Tuple, Type

import torch

from pytorch3d.structures import Pointclouds
from pytorch3d.renderer import (
    CamerasBase,
    PerspectiveCameras, 
    PointsRasterizationSettings,
    PointsRenderer,
    PointsRasterizer,
    AlphaCompositor
)

class PointsRenderLayer(torch.nn.Module):
    def __init__(
        self,
        raster_settings: Optional[PointsRasterizationSettings] = PointsRasterizationSettings(),
        compositor: Optional[torch.nn.Module] = AlphaCompositor()
    ) -> None:
        super().__init__()
        
        rasterizer = PointsRasterizer(
            cameras=PerspectiveCameras(), 
            raster_settings=raster_settings
        )
        
        self.renderer = PointsRenderer(
            rasterizer=rasterizer,
            compositor=compositor
        )
        
    def set_cameras(
        self, 
        c: Type[CamerasBase]
    ) -> None:
        self.renderer.rasterizer.cameras = c
    
    def set_raster_settings(
        self,
        r: PointsRasterizationSettings
    ) -> None:
        self.renderer.rasterizer.raster_settings = r
    
    def set_raster_image_size(
        self,
        s: Tuple[int, int]
    ) -> None:
        self.renderer.rasterizer.raster_settings.image_size = s
    
    def forward(
        self,
        x: Pointclouds,
        **kwargs
    ) -> torch.Tensor:
        return self.renderer(x, **kwargs)