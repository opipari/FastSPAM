from typing import Any, Dict, List, Tuple

import torch
from torch.nn import functional as F
import torchvision as tv

from segment_anything.modeling import Sam

from MVPd2SA1B import sample_point_from_mask

class ISam(Sam):

    def __init__(
        self,
        sam: Sam,
        interactive_iterations = 8,
        mask_iterations = 2
    ) -> None:
        """
        SAM predicts object masks from an image and input prompts.

        Arguments:
          image_encoder (ImageEncoderViT): The backbone used to encode the
            image into image embeddings that allow for efficient mask prediction.
          prompt_encoder (PromptEncoder): Encodes various types of input prompts.
          mask_decoder (MaskDecoder): Predicts masks from the image embeddings
            and encoded prompts.
          pixel_mean (list(float)): Mean values for normalizing pixels in the input image.
          pixel_std (list(float)): Std values for normalizing pixels in the input image.
        """
        super().__init__(
            image_encoder = sam.image_encoder,
            prompt_encoder = sam.prompt_encoder,
            mask_decoder = sam.mask_decoder,
            pixel_mean = sam.pixel_mean,
            pixel_std = sam.pixel_std,
        )
        self.interactive_iterations = interactive_iterations
        self.mask_iterations = mask_iterations
        self.reset_image()


    def reset_image(self) -> None:
        """Resets the currently set image."""
        self.is_image_set = False
        self.features = None
    
    def set_image(
        self,
        batched_input: List[Dict[str, Any]]
    ) -> None:
        self.reset_image()

        input_images = torch.stack([self.preprocess(x["image"]) for x in batched_input], dim=0)
        self.features = self.image_encoder(input_images)
        self.is_image_set = True

    def calculate_loss(
        self,
        image_record: Dict[str, Any],
        output_record: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        true_masks = F.interpolate(image_record["masks"].unsqueeze(1), 
                                        image_record["original_size"], mode="bilinear", align_corners=False)
        pred_masks = output_record["masks"].float()
        pred_logits = self.postprocess_masks(
            output_record["low_res_logits"],
            input_size=image_record["masks"].shape[-2:],
            original_size=image_record["original_size"],
        )

        true_inter = torch.sum(torch.logical_and(true_masks, pred_masks), dim=(2,3))
        true_union = torch.sum(torch.logical_or(true_masks, pred_masks), dim=(2,3))
        true_iou = true_inter/true_union
        true_iou = true_iou.detach()

        loss_iou = torch.mean(F.mse_loss(output_record["iou_predictions"], true_iou, reduction='none'), dim=1)

        loss_focal = torch.mean(tv.ops.sigmoid_focal_loss(pred_logits, true_masks.repeat((1,pred_logits.shape[1],1,1))), dim=(2,3))
        loss_dice = (2. * true_inter) / (torch.sum(pred_masks, dim=(2,3)) + torch.sum(true_masks, dim=(2,3)))
        loss_mask, mask_arg = torch.min((20 * loss_focal) + loss_dice, dim=1)
        return torch.mean(loss_mask + loss_iou), mask_arg
      
    def forward_step(
        self,
        batched_input: List[Dict[str, Any]],
        multimask_output: bool,
    ) -> List[Dict[str, torch.Tensor]]:
        """
        Predicts masks end-to-end from provided images and prompts.
        If prompts are not known in advance, using SamPredictor is
        recommended over calling the model directly.

        Arguments:
          batched_input (list(dict)): A list over input images, each a
            dictionary with the following keys. A prompt key can be
            excluded if it is not present.
              'image': The image as a torch tensor in 3xHxW format,
                already transformed for input to the model.
              'original_size': (tuple(int, int)) The original size of
                the image before transformation, as (H, W).
              'point_coords': (torch.Tensor) Batched point prompts for
                this image, with shape BxNx2. Already transformed to the
                input frame of the model.
              'point_labels': (torch.Tensor) Batched labels for point prompts,
                with shape BxN.
              'boxes': (torch.Tensor) Batched box inputs, with shape Bx4.
                Already transformed to the input frame of the model.
              'mask_inputs': (torch.Tensor) Batched mask inputs to the model,
                in the form Bx1xHxW.
          multimask_output (bool): Whether the model should predict multiple
            disambiguating masks, or return a single mask.

        Returns:
          (list(dict)): A list over input images, where each element is
            as dictionary with the following keys.
              'masks': (torch.Tensor) Batched binary mask predictions,
                with shape BxCxHxW, where B is the number of input prompts,
                C is determined by multimask_output, and (H, W) is the
                original size of the image.
              'iou_predictions': (torch.Tensor) The model's predictions
                of mask quality, in shape BxC.
              'low_res_logits': (torch.Tensor) Low resolution logits with
                shape BxCxHxW, where H=W=256. Can be passed as mask input
                to subsequent iterations of prediction.
        """
        
        image_embeddings = self.features

        outputs = []
        for image_record, curr_embedding in zip(batched_input, image_embeddings):
            if "point_coords" in image_record:
                points = (image_record["point_coords"], image_record["point_labels"])
            else:
                points = None
            sparse_embeddings, dense_embeddings = self.prompt_encoder(
                points=points,
                boxes=image_record.get("boxes", None),
                masks=image_record.get("mask_inputs", None),
            )
            low_res_masks, iou_predictions = self.mask_decoder(
                image_embeddings=curr_embedding.unsqueeze(0),
                image_pe=self.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=multimask_output,
            )
            masks = self.postprocess_masks(
                low_res_masks,
                input_size=image_record["masks"].shape[-2:],
                original_size=image_record["original_size"],
            )
            masks = masks > self.mask_threshold
            outputs.append(
                {
                    "masks": masks,
                    "iou_predictions": iou_predictions,
                    "low_res_logits": low_res_masks,
                }
            )

        loss = 0
        mask_args = []
        for image_record, output_record in zip(batched_input, outputs):
            loss_, mask_arg_ = self.calculate_loss(image_record, output_record)
            loss += loss_
            output_record["mask_arg"] = mask_arg_

        loss /= len(batched_input)

        return outputs, loss

    def forward_interactive(
        self, 
        batched_input: List[Dict[str, Any]],
        multimask_output: bool,
    ):
        if not self.is_image_set:
            self.set_image(batched_input)

        batched_output, loss = self.forward_step(batched_input, multimask_output)

        with torch.no_grad():
            for image_record, output_record in zip(batched_input, batched_output):
                true_masks = image_record["masks"].bool()
                pred_masks = output_record["masks"][torch.arange(len(output_record["masks"])), output_record["mask_arg"]].unsqueeze(1).float()
                pred_masks = F.interpolate(pred_masks, 
                                          image_record["masks"].shape[-2:], mode="bilinear", align_corners=False).squeeze(1).bool()
                error_region = torch.ne(pred_masks, true_masks)
                
                # Sample point
                point_samples = []
                point_labels = []
                for i in range(len(pred_masks)):
                    if error_region[i].sum().item()>0:
                        point = sample_point_from_mask(error_region[i].cpu())
                    else:
                        point = sample_point_from_mask(true_masks[i].cpu())
                    point_samples.append(point)
                    point_labels.append(true_masks[i, point[0,1], point[0,0]].unsqueeze(0))
                point_samples = torch.stack(point_samples).to(dtype=torch.float32)
                point_labels = torch.stack(point_labels).to(dtype=torch.float32)

                
                if 'boxes' in image_record:
                    del image_record['boxes']
                if 'mask_inputs' in image_record:
                    del image_record['mask_inputs']
                if 'point_coords' in image_record:
                    del image_record['point_coords']
                if 'point_labels' in image_record:
                    del image_record['point_labels']
                image_record['point_coords'] = point_samples.to(device=image_record["masks"].device)
                image_record['point_labels'] = point_labels.to(device=image_record["masks"].device)

                image_record['mask_inputs'] = output_record["low_res_logits"][torch.arange(len(pred_masks)), output_record["mask_arg"]].unsqueeze(1).detach()

                del output_record["mask_arg"]
                # import matplotlib.pyplot as plt
                # fig, ax = plt.subplots(len(pred_masks),3, figsize=(3,len(pred_masks)*3))
                # for i in range(len(pred_masks)):
                #     ax[i,0].imshow(true_masks[i].bool().cpu())
                #     ax[i,1].imshow(pred_masks[i].bool().cpu())
                #     ax[i,2].imshow(error_region[i].bool().cpu())
                #     circle = plt.Circle((point_samples[i,0][0], point_samples[i,0][1]), 10, facecolor='r')
                #     ax[i,0].add_artist(circle)
                #     circle = plt.Circle((point_samples[i,0][0], point_samples[i,0][1]), 10, facecolor='r')
                #     ax[i,1].add_artist(circle)
                #     circle = plt.Circle((point_samples[i,0][0], point_samples[i,0][1]), 10, facecolor='r')
                #     ax[i,2].add_artist(circle)
                # plt.show()
        
        return batched_input, loss

    def forward_noninteractive(
        self, 
        batched_input: List[Dict[str, Any]],
        multimask_output: bool,
    ):

        if not self.is_image_set:
            self.set_image(batched_input)

        with torch.no_grad():
            for image_record in batched_input:
                assert 'mask_inputs' in image_record, "Mask inputs must be given as input for noninteractive step"
                if 'boxes' in image_record:
                    del image_record['boxes']
                if 'point_coords' in image_record:
                    del image_record['point_coords']
                if 'point_labels' in image_record:
                    del image_record['point_labels']

        batched_output, loss = self.forward_step(batched_input, multimask_output)
        
        with torch.no_grad():
            for image_record, output_record in zip(batched_input, batched_output):
                image_record['mask_inputs'] = output_record["low_res_logits"][torch.arange(len(output_record["low_res_logits"])), output_record["mask_arg"]].unsqueeze(1).detach()
                del output_record["mask_arg"]

        return batched_output, loss
