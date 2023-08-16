from typing import Optional, Type

import numpy as np

import torch
import torch.nn as nn
from torchvision.ops.boxes import batched_nms

from pytorch3d.renderer import CamerasBase
from pytorch3d.structures import Pointclouds

from segment_anything import SamPredictor, SamAutomaticMaskGenerator
from segment_anything.modeling import Sam
from segment_anything.utils.amg import (
    MaskData,
    area_from_rle,
    batch_iterator,
    batched_mask_to_box,
    box_xyxy_to_xywh,
    build_point_grid,
    build_all_layer_point_grids,
    calculate_stability_score,
    coco_encode_rle,
    generate_crop_boxes,
    is_box_near_crop_edge,
    mask_to_rle_pytorch,
    remove_small_regions,
    rle_to_mask,
    uncrop_boxes_xyxy,
    uncrop_masks,
    uncrop_points,
)

from MVPd.utils.MVPdHelpers import get_xy_depth

from pointsrenderlayer import PointsRenderLayer

def uniform_grid_sample_mask(bin_mask, samples=100):
    """
    bin_mask: H x W
    Returns samples_per_inst x 2 
    """
    height, width = bin_mask.shape
    xv, yv = torch.meshgrid(torch.arange(width), torch.arange(height), indexing='xy')
    grid = torch.stack([xv,yv], axis=-1).to(bin_mask.device)
    bin_mask = bin_mask.reshape(-1)
    grid_indices = grid.reshape(-1,2)[bin_mask]
    if grid_indices.shape[0]==0:
        return torch.empty(0,2)
    shuffle_indices = np.random.choice(np.arange(grid_indices.shape[0]), size=min(grid_indices.shape[0], samples), replace=False)
    grid_indices = grid_indices[shuffle_indices]
    if grid_indices.shape[1]<samples:
        replicate_indices = np.random.choice(np.arange(grid_indices.shape[0]), size=samples, replace=True)
        grid_indices = grid_indices[replicate_indices]
    
    return grid_indices

class SAMSelfPrompting(nn.Module):
    def __init__(
        self,
        model: Sam,
        pred_iou_thresh: float = 0.88,
        stability_score_thresh: float = 0.95,
        stability_score_offset: float = 1.0,
        box_nms_thresh: float = 0.7,
        min_mask_region_area: int = 0,
        prompts_per_object: int = 20,
        objects_per_batch: int = 10
    ) -> None:
        super().__init__()
        
        self.predictor = SamPredictor(model)
        self.automatic_mask_generator = SamAutomaticMaskGenerator(
            model=model,
            pred_iou_thresh=pred_iou_thresh,
            stability_score_thresh=stability_score_thresh,
            stability_score_offset=stability_score_offset,
            box_nms_thresh=box_nms_thresh,
            min_mask_region_area=min_mask_region_area
            )
        self.render_layer = PointsRenderLayer()

        self.box_nms_thresh = box_nms_thresh
        self.prompts_per_object = prompts_per_object
        self.objects_per_batch = objects_per_batch
        self.randomize_memory_coordinates = True
        
        self.memory = None

    def _process_image(
        self,
        points_for_image: np.ndarray,
    ) -> torch.Tensor:

        data = MaskData()
        for (points,) in batch_iterator(self.objects_per_batch, points_for_image):
            batch_data = self._process_batch(points)
            data.cat(batch_data)
            del batch_data
        

        # Remove duplicates within this crop.
        keep_by_nms = batched_nms(
            data["boxes"].float(),
            data["iou_preds"],
            torch.zeros_like(data["boxes"][:, 0]),  # categories
            iou_threshold=self.box_nms_thresh,
        )
        data.filter(keep_by_nms)

        return data

    def _process_batch(
        self,
        points: np.ndarray
    ) -> MaskData:
        # orig_h, orig_w = orig_size
        
        # Run model on this batch
        transformed_points = self.predictor.transform.apply_coords(points, self.predictor.original_size)
        in_points = torch.as_tensor(transformed_points, device=self.predictor.device)
        in_labels = torch.ones(in_points.shape[:2], dtype=torch.int, device=in_points.device)
        
        masks, iou_preds, _ = self.predictor.predict_torch(
            in_points,
            in_labels,
            multimask_output=False,
            return_logits=True,
        )

        # Serialize predictions and store in MaskData
        data = MaskData(
            masks=masks.flatten(0, 1),
            iou_preds=iou_preds.flatten(0, 1),
            points=torch.as_tensor(points.repeat(masks.shape[1], axis=0)),
        )
        del masks

        # data["masks_logits"] = data["masks"].clone()
        # Threshold masks and calculate boxes
        data["masks"] = data["masks"] > self.predictor.model.mask_threshold
        data["boxes"] = batched_mask_to_box(data["masks"])

        # Compress to RLE
        # data["masks"] = uncrop_masks(data["masks"], crop_box, orig_h, orig_w)
        # data["rles"] = mask_to_rle_pytorch(data["masks"])
        # del data["masks"]
        
        return data
        
    
    def initialize_prompt_sets(
        self,
        depth: torch.Tensor,
        camera: Type[CamerasBase],
        points: Optional[torch.Tensor] = None,
        masks: Optional[torch.Tensor] = None,
    ) -> None:
        """
        depth: 1 x 1 x H x W
        points: N x self.prompts_per_object x 2 in XY format (i.e. [horizontal, vertical] order)
        masks: N x 1 x H x W one hot encoded binary masks
        """
        assert depth.shape[0]==1
        
        if points is not None:
            assert prompts.shape[1:]==(self.prompts_per_object, 2)
            N = points.shape[0]
            prompt_screen_coords = prompts.clone()
            self._procecss(prompt_screen_coords)
            init_masks = None
        elif masks is not None:
            N = masks.shape[0]
            init_masks = masks.clone()
        else:
            raise NotImplementedError("Prompts must be initialized from points or masks")
        
        xy_depth = get_xy_depth(depth, from_ndc=True).permute(0,2,3,1).reshape(1,-1,3)
        xyz = camera.unproject_points(xy_depth, from_ndc=True, world_coordinates=True)
        seg = init_masks.permute(1,2,3,0).reshape(1,-1,N).to(dtype=xyz.dtype).to(device=xyz.device)
        seg = torch.cat([seg, torch.ones(1,seg.shape[1],1).to(device=xyz.device)], dim=2)
        
        self.memory = Pointclouds(points=xyz, features=seg)
    
    def reset_memory(self):
        self.memory = None

    def get_screen_coords(self, rendered_mem, camera, randomize=False):
        _, h, w = rendered_mem.shape
        new_memory_screen_coords = []
        if randomize:
            for i, bin_mask in enumerate(rendered_mem):
                sampled_coords = uniform_grid_sample_mask(bin_mask, samples=self.prompts_per_object)
                if sampled_coords.shape[0]>0:
                    new_memory_screen_coords.append(sampled_coords)
        else:
            prev_xyz = self.memory.points_list()[0].clone()
            prev_feats = self.memory.features_list()[0].clone()[:,:-1].reshape(h,w,-1).permute(2,0,1)

            proj_xyz = camera.transform_points(prev_xyz)
            proj_xy = torch.tensor([[w,h]])-proj_xyz.reshape(h,w,3)[:,:,:2].cpu().permute(1,0,2)
            proj_coords = proj_xy[self.memory_screen_coords[:,:,0],self.memory_screen_coords[:,:,1]].to(dtype=torch.int)

            for inst_id, (new_xy, old_xy) in enumerate(zip(proj_coords, self.memory_screen_coords)):
                inbound_indices = []
                for i in range(len(new_xy)):
                    if new_xy[i,0]>=0 and new_xy[i,0]<w and \
                        new_xy[i,1]>=0 and new_xy[i,1]<h:
                        old_inst_feat = float(prev_feats[inst_id,old_xy[i,1],old_xy[i,0]].item())
                        new_inst_feat = float(rendered_mem[inst_id,new_xy[i,1],new_xy[i,0]].item())
                        if abs(old_inst_feat-new_inst_feat)<0.1:
                            inbound_indices.append(i)
                valid_coordinates_xy = new_xy[np.array(inbound_indices)]
                
                if valid_coordinates_xy.shape[0]==0:
                    continue
                if valid_coordinates_xy.shape[0]<self.prompts_per_object:
                    num_to_fill = self.prompts_per_object-valid_coordinates_xy.shape[0]
                    # replicate_indices = np.random.choice(np.arange(valid_coordinates_xy.shape[0]), size=num_to_fill, replace=True)
                    # valid_coordinates_xy = torch.cat((valid_coordinates_xy, valid_coordinates_xy[replicate_indices]),dim=0)
                    sampled_coords = uniform_grid_sample_mask(rendered_mem[inst_id], samples=num_to_fill).to(device=valid_coordinates_xy.device)
                    valid_coordinates_xy = torch.cat((valid_coordinates_xy, sampled_coords),dim=0)

                new_memory_screen_coords.append(valid_coordinates_xy)

        return new_memory_screen_coords
            

    def automatic_generate(self, image):
        pred = self.automatic_mask_generator.generate(image)
        pred = {"masks": torch.stack([torch.as_tensor(pr["segmentation"]) for pr in pred], axis=0)}
        return pred

    def forward(
        self,
        image: torch.Tensor,
        depth: torch.Tensor,
        camera: Type[CamerasBase],
        verbose: bool = False
    ):
        image = np.array(image.permute(0,2,3,1).cpu()[0]).astype(np.uint8)

        self.predictor.set_image(image)

        image_size = (int(camera.image_size[0,0].item()), int(camera.image_size[0,1].item()))
        self.render_layer.set_raster_image_size(image_size)
        self.render_layer.set_cameras(camera)

        if self.memory is None:
            pred = self.automatic_generate(image)
            self.memory_screen_coords = self.get_screen_coords(pred["masks"], camera, randomize=True)
        else:
            rendered_mem = self.render_layer(self.memory).permute(0,3,1,2)

            bin_thresh = 0.95
            rendered_new_rgn = rendered_mem[:,-1:] < bin_thresh
            rendered_mem = rendered_mem[:,:-1] >= bin_thresh


            self.memory_screen_coords = self.get_screen_coords(rendered_mem[0], camera, randomize=self.randomize_memory_coordinates)
            
            if len(self.memory_screen_coords)==0:
                self.automatic_generate(image)
                self.memory_screen_coords = self.get_screen_coords(pred["masks"], camera, randomize=True)
            else:
                self.memory_screen_coords = torch.stack(self.memory_screen_coords).to(dtype=torch.int)
                pred = self._process_image(self.memory_screen_coords.cpu().numpy())

                new_rgn_area = rendered_new_rgn.sum().item()
                if new_rgn_area>0:
                    new_rgn_coords = uniform_grid_sample_mask(rendered_new_rgn[0,0], samples=max((32*32)//new_rgn_area, 1)).unsqueeze(1).to(device=self.memory_screen_coords.device) # samples x 1 x 2
                    fill = self._process_image(new_rgn_coords.cpu().numpy())
                    if fill["masks"].shape[0]>0:
                        new_rgn_coords = self.get_screen_coords(fill["masks"], camera, randomize=True)
                        new_rgn_coords = torch.stack(new_rgn_coords).to(dtype=torch.int).to(device=self.memory_screen_coords.device)
                    else:
                        new_rgn_coords = torch.empty(0,self.prompts_per_object,2).to(dtype=torch.int)
                        
                    merged_pred_fill = MaskData(
                        masks=torch.cat([pred["masks"], fill["masks"]],axis=0),
                        iou_preds=torch.cat([pred["iou_preds"], fill["iou_preds"]],axis=0),
                        boxes=torch.cat([pred["boxes"], fill["boxes"]],axis=0).float(),
                        points=torch.cat([self.memory_screen_coords, new_rgn_coords],axis=0)
                    )

                    keep_by_nms = batched_nms(
                        merged_pred_fill["boxes"].float(),
                        merged_pred_fill["iou_preds"],
                        torch.zeros_like(merged_pred_fill["boxes"][:, 0]),  # categories
                        iou_threshold=self.box_nms_thresh,
                    )

                    merged_pred_fill.filter(keep_by_nms)
                    pred = merged_pred_fill
                    self.memory_screen_coords = pred["points"]

        
        xy_depth = get_xy_depth(depth, from_ndc=True).permute(0,2,3,1).reshape(1,-1,3)
        xyz = camera.unproject_points(xy_depth, from_ndc=True, world_coordinates=True)
        seg = pred["masks"].permute(1,2,0).flatten(0,1).unsqueeze(0).to(dtype=xyz.dtype).to(device=xyz.device)
        seg = torch.cat([seg, torch.ones(1,seg.shape[1],1).to(device=xyz.device)], dim=2)
        
        self.memory = Pointclouds(points=xyz, features=seg)

        self.predictor.reset_image()

        pred["rles"] = mask_to_rle_pytorch(pred["masks"])

        output = {"coco_rle": [coco_encode_rle(rle) for rle in pred["rles"]],
                  "memory_coordinates": self.memory_screen_coords
                 }
        
        if verbose:
            output["rendered_memory"] = rendered_mem

        return output