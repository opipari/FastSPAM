from ultralytics import YOLO

epochs = 2

# for i in range(epochs):
#     if i==0:
model = YOLO(model="yolov8x-seg.yaml", \
             ).load('FastSAM-x.pt')
    # else:
    #     model = YOLO(model="yolov8x-seg.yaml", \
    #                  ).load(f'fastsam_{i-1}/test/weights/last.pt')

model.train(data=f"sa.yaml", \
            epochs=2, \
            batch=16, \
            imgsz=1024, \
            overlap_mask=False, \
            save=True, \
            save_period=1, \
            device='0',\
            project=f'fastsam', \
            name='test', 
            val=False,
            cache='disk')