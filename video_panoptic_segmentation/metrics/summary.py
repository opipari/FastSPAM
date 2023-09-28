
import json
results = json.load(open('video_panoptic_segmentation/models/segment-anything/results/evaluate_pretrained_sam_automatic/metrics_sam_automatic.json','r'))
# results = json.load(open('video_panoptic_segmentation/models/segment-anything/results/evaluate_pretrained_sam_automatic_vit_b/metrics_sam_automatic_vit_b.json','r'))


mask_inter = 0
mask_prec_den = 0
mask_rec_den = 0

bound_inter = 0
bound_prec_den = 0
bound_rec_den = 0

for video_name in results:
	for sample_results in results[video_name]:
		for ind in sample_results:
			mask_inter_, mask_prec_, mask_rec_, bound_inter_, bound_prec_, bound_rec_ = sample_results[ind]["met"]


			mask_inter += mask_inter_
			mask_prec_den += mask_prec_
			mask_rec_den += mask_rec_

			bound_inter += bound_inter_
			bound_prec_den += bound_prec_
			bound_rec_den += bound_rec_

print("Mask")
mask_prec = mask_inter/mask_prec_den
mask_rec = mask_inter/mask_rec_den
print(f"Precision: {mask_prec}")
print(f"Recall: {mask_rec}")
print(f"F-Score: {(2*mask_prec*mask_rec)/(mask_prec+mask_rec)}")

print("Boundary")
bound_prec = bound_inter/bound_prec_den
bound_rec = bound_inter/bound_rec_den
print(f"Precision: {bound_prec}")
print(f"Recall: {bound_rec}")
print(f"F-Score: {(2*bound_prec*bound_rec)/(bound_prec+bound_rec)}")