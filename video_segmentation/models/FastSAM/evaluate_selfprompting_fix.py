import os
import json
import argparse

import time
from tqdm import tqdm
import math

from PIL import Image
import numpy as np
import torch
import torch.nn as nn

import matplotlib.pyplot as plt

from panopticapi.utils import id2rgb

from MVPd.utils.MVPdataset import MVPDataset, MVPVideo, MVPdCategories, video_collate
from MVPd.utils.MVPdHelpers import get_xy_depth, get_cameras
from pytorch3d.structures import Pointclouds

from fastsam import FastSAM, FastSAMPrompt

from video_segmentation.metrics import utils as metric_utils


def get_dataset(dataset_config):
    dataset = MVPDataset(root=dataset_config["root"],
                        split=dataset_config["split"],
                        window_size = 0)

    return dataset
    
def get_model(model_config):
    model = FastSelfPrompting(model_config)
    return model



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


def mask_to_point(binmask):
    y_center, x_center = np.argwhere(binmask.cpu().numpy()==1).sum(0)/(binmask.cpu().numpy()==1).sum()
    y_center, x_center = int(y_center), int(x_center)

    if not binmask[y_center,x_center].item():
        x_center, y_center = uniform_grid_sample_mask(binmask, samples=1)[0]
        y_center, x_center = int(y_center), int(x_center)

    return [x_center, y_center] # (w x h) (horizontal x vertical)


class FastSelfPrompting(nn.Module):
    def __init__(
        self,
        config
    ) -> None:
        super().__init__()
        
        self.model = FastSAM(config["checkpoint"])
        self.config = config
        self.reset_memory()

    def reset_memory(self):
        self.memory = None


    def reprojected_memory_coords(self, image, depth, camera):
        if self.memory is None or self.memory_coords is None or len(self.memory_coords)==0:
            return None
        h, w, _ = image.shape

        prev_xyz = self.memory.points_list()[0].clone()
        proj_xyz = camera.transform_points(prev_xyz)
        proj_xy = torch.tensor([[w,h]])-proj_xyz.reshape(h,w,3)[:,:,:2].cpu().permute(1,0,2)
        # plt.imshow(proj_xy[:,:,0])
        # plt.savefig('bbin.png')
        # plt.close()
        # plt.imshow(proj_xy[:,:,1])
        # plt.savefig('bbbin.png')
        # plt.close()
        proj_coords = proj_xy[self.memory_coords[:,0],self.memory_coords[:,1]].to(dtype=torch.int)        

        
        # plt.imshow(depth[0,0].cpu().numpy())
        # plt.scatter(self.memory_coords[:,0], self.memory_coords[:,1], marker='o', color="blue")
        # plt.scatter(proj_coords[:,0], proj_coords[:,1], marker='o', color="red")
        # plt.savefig('bin.png')
        # plt.close()

        inbound_coords = ((proj_coords[:,0]>=0) & (proj_coords[:,0]<w)) & ((proj_coords[:,1]>=0) & (proj_coords[:,1]<h))
        proj_coords = proj_coords[inbound_coords]

        # plt.imshow(depth[0,0].cpu().numpy())
        # plt.scatter(self.memory_coords[:,0], self.memory_coords[:,1], marker='o', color="blue")
        # plt.scatter(proj_coords[:,0], proj_coords[:,1], marker='o', color="red")
        # plt.savefig('bin_.png')
        # plt.close()
        if len(proj_coords)==0:
            return None
        else:
            return proj_coords



    def forward(self, image, depth, camera):
        input = Image.fromarray(image)
        input = input.convert("RGB")
        everything_results = self.model(
            input,
            device=0,
            retina_masks=self.config["retina"],
            imgsz=self.config["imgsz"],
            conf=self.config["conf"],
            iou=self.config["iou"]
            )
        prompt_process = FastSAMPrompt(input, everything_results, device=0)
        

        points_ = self.reprojected_memory_coords(image, depth, camera)
        if points_ is None:
            ann = prompt_process.everything_prompt()
            if len(ann)>0:
                ann = ann.to(torch.bool)
                ann = ann[torch.argsort(torch.sum(ann,(1,2)),descending=True)]

        else:
            ann = []
            all_ann = prompt_process.everything_prompt()
            if len(all_ann)>0:
                points = points_[:1]
                ann = torch.as_tensor(prompt_process.point_prompt(points=points, pointlabel=[1]))
                for p in points_[1:]:
                    mask = torch.as_tensor(prompt_process.point_prompt(points=[p], pointlabel=[1]))
                    max_iou, max_arg = torch.max(metric_utils.pairwise_segment_metric(mask, ann, metric='IOU'), dim=1)
                    max_iou, max_arg = max_iou.item(), max_arg.item()
                    if max_iou>=self.config["iou"]:
                        ann[max_arg] = torch.logical_or(ann[max_arg], mask[0])
                    else:
                        ann = torch.cat([ann, mask])
                        points = torch.cat([points, p.unsqueeze(0)])

                
                all_ann = all_ann.to(torch.bool)
                all_ann = all_ann[torch.argsort(torch.sum(all_ann,(1,2)),descending=True)]
                added_ann = torch.any(ann,dim=0,keepdim=True).to(all_ann.device)
                
                pairwise_inter = metric_utils.pairwise_segment_metric(all_ann, added_ann, metric="Intersection")
                inter_over_i = pairwise_inter / torch.sum(all_ann,(1,2)).reshape(-1,1).to(pairwise_inter.device)
                inter_over_iou = (inter_over_i>=self.config["iou"]).reshape(-1)
                new_ann = all_ann[~inter_over_iou]
                new_points_ = torch.as_tensor([mask_to_point(ar) for ar in new_ann])

                for p in new_points_:
                    mask = torch.as_tensor(prompt_process.point_prompt(points=[p], pointlabel=[1]))
                    max_iou, max_arg = torch.max(metric_utils.pairwise_segment_metric(mask,ann,metric='IOU'), dim=1)
                    max_iou, max_arg = max_iou.item(), max_arg.item()
                    if max_iou>=self.config["iou"]:
                        ann[max_arg] = torch.logical_or(ann[max_arg], mask[0])
                    else:
                        ann = torch.cat([ann, mask])
                        points = torch.cat([points, p.unsqueeze(0)])

                non_empty = torch.sum(ann,(1,2))>0
                points = points[non_empty]
                ann = ann[non_empty]

                sort_order = torch.argsort(torch.sum(ann,(1,2)),descending=True)
                points = points[sort_order]
                ann = ann[sort_order]
            


        if len(ann)>0:
            self.memory_coords = torch.as_tensor([mask_to_point(ar) for ar in ann])
            ann = metric_utils.mask_to_rle_pytorch(ann)
        else:
            self.memory_coords = None
        coco_rle = [metric_utils.coco_encode_rle(rle) for rle in ann]


        xy_depth = get_xy_depth(depth, from_ndc=True).permute(0,2,3,1).reshape(1,-1,3)
        xyz = camera.unproject_points(xy_depth, from_ndc=True, world_coordinates=True)        
        self.memory = Pointclouds(points=xyz)

        return coco_rle


def evaluation_process(index, nprocs, config, output_dir):

    dataset = get_dataset(config["dataset"])

    model = get_model(config['model'])


    if nprocs>0:
        is_per_proc = math.ceil(len(dataset)/nprocs)
        i_start = index*is_per_proc
        i_end = min((index+1)*is_per_proc, len(dataset))
        inds = list(range(i_start, i_end))
        dataset = torch.utils.data.Subset(dataset, inds)
        print(len(dataset))
    rand_inds = [1367,  100,  983,   83, 1586,  896,  683, 1598, 1092, 1020,  435,  747,
        1497,  484,  473,  367,  622,   14, 1290, 1414,  459,  740,  111, 1383,
        1189, 1698, 1280, 1584,  286,  229, 1279,  463, 1396, 1305,  500,  214,
        1513,  108,  536, 1729,  240, 1184, 1061,  722,  513,  650,   59,  128,
         180, 1028,   96,  970, 1488, 1470, 1128,  615,  274, 1155,  519,  280,
         938, 1547,  234,  971,  620,  677,  659,  227, 1135, 1001,  150,  586,
         825, 1541,    4,  178, 1285,  209,  966,  153,  857,  580,  633,  954,
        1520,  852, 1580,  348, 1457,  223,  205,  824,   56,  511, 1400,  879,
        1632,  116, 1007, 1145,  717, 1207, 1164,  772, 1002, 1315, 1114, 1206,
        1011,   44, 1374, 1505, 1269, 1049,  351,  373, 1510,  177, 1427, 1193,
         853,   41, 1583,  648, 1391,  829, 1485, 1682, 1362,   64, 1328, 1262,
        1228, 1499, 1492,  778, 1641, 1297,  791, 1349,  483, 1168, 1324, 1201,
         979,   88, 1010, 1590, 1208, 1132, 1714,  173,  893, 1460,  477, 1096,
         847,  216, 1601,  624,  119,  667,  527,  237,  221, 1496,  322,  989,
        1614,  329,  305,  122, 1401,  251]
    # print("Within evaluation process")
    with torch.no_grad():
        for vi, video in enumerate(tqdm(dataset, position=0, disable=index!=0)):
            # vitime = time.time()
            if vi not in rand_inds:
                continue
            first_sample = next(iter(video))
            video_name = first_sample['meta']['video_name']
            out_dir = os.path.join(output_dir, config['experiment_name'], 'panomasksRLE', video_name)
            
            if os.path.exists(os.path.join(out_dir)):
                continue
            os.makedirs(out_dir, exist_ok=True)

            model.reset_memory()
            for idx, sample in enumerate(tqdm(video, position=1, disable=index!=0)):
                # Load metadata
                video_name = sample['meta']['video_name']
                out_dir = os.path.join(output_dir, config['experiment_name'], 'panomasksRLE', video_name)
                out_file = sample['meta']['window_names'][0].split('.')[0]+'.pt'

                if os.path.exists(os.path.join(out_dir, out_file)):
                    continue

                # Run SAM
                image = np.array(sample['observation']['image'][0]).astype(np.uint8) # 480 x 640 x 3

                # image = torch.tensor(sample['observation']['image']).permute(0,3,1,2).to('cuda')
                depth = torch.tensor(sample['observation']['depth']).unsqueeze(1).to('cuda')
                camera = get_cameras(sample['camera']['K'],
                                    sample['camera']['W2V_pose'],
                                    sample['meta']['image_size']).to('cuda')


                coco_rle = model(image, depth, camera)
                
                torch.save({"coco_rle":coco_rle}, os.path.join(out_dir, out_file))


            # print(f"Finished processing {vi}/{len(dataset)}: ", video_name, f"in {time.time()-vitime}s", flush=True)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config-path", dest="config_path")
    parser.add_argument("--output-path", dest="output_path")
    args = parser.parse_args()

    config = json.load(open(args.config_path, 'r'))

    nprocs = 1
    torch.multiprocessing.spawn(evaluation_process, args=(nprocs, config, args.output_path), nprocs=nprocs)

    
