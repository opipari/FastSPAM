
class InitConfig():
    def __init__(self):
        self.experiment_name = "train_automatic_sam_vit_b"
        self.output_dir = "video_panoptic_segmentation/models/segment-anything/results"
        self.model_type = "vit_b"
        self.sam_checkpoint = "./video_panoptic_segmentation/models/segment-anything/segment-anything/checkpoints/sam_vit_b_01ec64.pth"
        
        self.use_augmentation = False
        self.total_iterations = 90000
        self.eval_every = 5000
        self.eval_iterations = 50
        self.learning_rate = 8e-4
        self.adam_betas = (0.9, 0.999)
        self.weight_decay = 0.1
        self.warmup_iters = 250
        self.lr_decay_iters = [60000, 86666]
        self.num_mask_samples = 16
        self.min_box_area_pcnt = 0.001
        self.seed = 0

if __name__=="__main__":
    import sys

    if len(sys.argv)==2:
        print(getattr(InitConfig(), sys.argv[1]))