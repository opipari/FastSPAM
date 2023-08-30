
class InitConfig():
    def __init__(self):
        self.experiment_name = "train_automatic_sam_vit_b"
        self.output_dir = "video_panoptic_segmentation/models/segment-anything/results"
        self.model_type = "vit_b"
        self.sam_checkpoint = "./video_panoptic_segmentation/models/segment-anything/segment-anything/checkpoints/sam_vit_b_01ec64.pth"
        
        self.use_augmentation = False
        self.total_iterations = 10000
        self.eval_every = 1000
        self.eval_iterations = 50
        self.learning_rate = 8e-4
        self.adam_betas = (0.9, 0.999)
        self.weight_decay = 0.1
        self.warmup_iters = 250
        self.lr_decay_iters = 8000
        self.num_mask_samples = 8
        self.min_box_area_pcnt = 0.001
        self.seed = 0

if __name__=="__main__":
    import sys

    if len(sys.argv)==2:
        print(getattr(InitConfig(), sys.argv[1]))