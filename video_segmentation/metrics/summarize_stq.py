import os
import torch

epsilon=1e-15
root = "/home/ANT.AMAZON.COM/topipari/Downloads/evaluate_stq_300length/"
print(os.listdir(root))

for moddir in sorted(os.listdir(root)):
	data0100 = torch.load(os.path.join(root,moddir,"res.json"))
	

	print(moddir,data0100["aq"],data0100["iou"],data0100["stq"], len(data0100['stq_per_seq']))#, data0100["gt300_per_seq"].sum())