# PAOT Benchmark Data

#### Static Images

  - [MSRA10K](https://mmcheng.net/msra10k/)
      ```
      aws s3 cp s3://prism-intern-anthony/raw_data/pretraining/static/MSRA10K/MSRA10K_Imgs_GT.zip ./zero_shot_scene_segmentation/datasets/raw_data/pretraining/static/MSRA10K/ && \
        unzip ./zero_shot_scene_segmentation/datasets/raw_data/pretraining/static/MSRA10K/MSRA10K_Imgs_GT.zip -d ./zero_shot_scene_segmentation/datasets/raw_data/pretraining/static/MSRA10K/ && \
          rm ./zero_shot_scene_segmentation/datasets/raw_data/pretraining/static/MSRA10K/MSRA10K_Imgs_GT.zip
      ```
  - [ECSSD](http://www.cse.cuhk.edu.hk/leojia/projects/hsaliency/dataset.html)
      ```
      aws s3 sync s3://prism-intern-anthony/raw_data/pretraining/static/ECSSD/ ./zero_shot_scene_segmentation/datasets/raw_data/pretraining/static/ECSSD/ && \
        unzip ./zero_shot_scene_segmentation/datasets/raw_data/pretraining/static/ECSSD/images.zip -d ./zero_shot_scene_segmentation/datasets/raw_data/pretraining/static/ECSSD/ && \
          rm ./zero_shot_scene_segmentation/datasets/raw_data/pretraining/static/ECSSD/images.zip
        unzip ./zero_shot_scene_segmentation/datasets/raw_data/pretraining/static/ECSSD/ground_truth_mask.zip -d ./zero_shot_scene_segmentation/datasets/raw_data/pretraining/static/ECSSD/ && \
          rm ./zero_shot_scene_segmentation/datasets/raw_data/pretraining/static/ECSSD/ground_truth_mask.zip
      ```
  - [PASCAL-S](http://cbs.ic.gatech.edu/salobj/download/salObj.zip)
      ```
      aws s3 cp s3://prism-intern-anthony/raw_data/pretraining/static/PASCAL-S/salObj.zip ./zero_shot_scene_segmentation/datasets/raw_data/pretraining/static/PASCAL-S/ && \
        unzip ./zero_shot_scene_segmentation/datasets/raw_data/pretraining/static/PASCAL-S/salObj.zip -d ./zero_shot_scene_segmentation/datasets/raw_data/pretraining/static/PASCAL-S/ && \
          rm ./zero_shot_scene_segmentation/datasets/raw_data/pretraining/static/PASCAL-S/salObj.zip
      ```
  - [PASCAL VOC2012](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/)
      ```
      aws s3 cp s3://prism-intern-anthony/raw_data/pretraining/static/PASCALVOC2012/VOCtrainval_11-May-2012.tar ./zero_shot_scene_segmentation/datasets/raw_data/pretraining/static/PASCALVOC2012/ && \
        tar -xvf ./zero_shot_scene_segmentation/datasets/raw_data/pretraining/static/PASCALVOC2012/VOCtrainval_11-May-2012.tar -C ./zero_shot_scene_segmentation/datasets/raw_data/pretraining/static/PASCALVOC2012/ && \
          rm ./zero_shot_scene_segmentation/datasets/raw_data/pretraining/static/PASCALVOC2012/VOCtrainval_11-May-2012.tar && \
            mv ./zero_shot_scene_segmentation/datasets/raw_data/pretraining/static/PASCALVOC2012/VOCdevkit/VOC2012/* ./zero_shot_scene_segmentation/datasets/raw_data/pretraining/static/PASCALVOC2012/ && \
              rm -r ./zero_shot_scene_segmentation/datasets/raw_data/pretraining/static/PASCALVOC2012/VOCdevkit/
      ``` 
  - [COCO 2017](https://cocodataset.org/#download)
    
    See here for confirmation of 2017: https://github.com/xmlyqing00/AFB-URR/issues/15

      ```
      aws s3 sync s3://prism-intern-anthony/raw_data/pretraining/static/COCO/ ./zero_shot_scene_segmentation/datasets/raw_data/pretraining/static/COCO/ && \
        unzip ./zero_shot_scene_segmentation/datasets/raw_data/pretraining/static/COCO/train2017.zip -d ./zero_shot_scene_segmentation/datasets/raw_data/pretraining/static/COCO/ && \
          rm ./zero_shot_scene_segmentation/datasets/raw_data/pretraining/static/COCO/train2017.zip
        unzip ./zero_shot_scene_segmentation/datasets/raw_data/pretraining/static/COCO/val2017.zip -d ./zero_shot_scene_segmentation/datasets/raw_data/pretraining/static/COCO/ && \
          rm ./zero_shot_scene_segmentation/datasets/raw_data/pretraining/static/COCO/val2017.zip
        unzip ./zero_shot_scene_segmentation/datasets/raw_data/pretraining/static/COCO/annotations_trainval2017.zip -d ./zero_shot_scene_segmentation/datasets/raw_data/pretraining/static/COCO/ && \
          rm ./zero_shot_scene_segmentation/datasets/raw_data/pretraining/static/COCO/annotations_trainval2017.zip
      ```

  - Preprocess 
      ```
      source ./envs/paot-benchmark/bin/activate
      
      python ./zero_shot_scene_segmentation/models/AFB-URR/AFB-URR/unify_pretrain_dataset.py --name MSRA10K --src ./zero_shot_scene_segmentation/datasets/raw_data/pretraining/static/MSRA10K/ --dst ./zero_shot_scene_segmentation/datasets/raw_data/pretraining/static/unified/ --palette ./zero_shot_scene_segmentation/models/AFB-URR/AFB-URR/assets/mask_palette.png
      
      python ./zero_shot_scene_segmentation/models/AFB-URR/AFB-URR/unify_pretrain_dataset.py --name ECSSD --src ./zero_shot_scene_segmentation/datasets/raw_data/pretraining/static/ECSSD/ --dst ./zero_shot_scene_segmentation/datasets/raw_data/pretraining/static/unified/ --palette ./zero_shot_scene_segmentation/models/AFB-URR/AFB-URR/assets/mask_palette.png
      
      python ./zero_shot_scene_segmentation/models/AFB-URR/AFB-URR/unify_pretrain_dataset.py --name PASCAL-S --src ./zero_shot_scene_segmentation/datasets/raw_data/pretraining/static/PASCAL-S/ --dst ./zero_shot_scene_segmentation/datasets/raw_data/pretraining/static/unified/ --palette ./zero_shot_scene_segmentation/models/AFB-URR/AFB-URR/assets/mask_palette.png
      
      python ./zero_shot_scene_segmentation/models/AFB-URR/AFB-URR/unify_pretrain_dataset.py --name PASCALVOC2012 --src ./zero_shot_scene_segmentation/datasets/raw_data/pretraining/static/PASCALVOC2012/ --dst ./zero_shot_scene_segmentation/datasets/raw_data/pretraining/static/unified/ --palette ./zero_shot_scene_segmentation/models/AFB-URR/AFB-URR/assets/mask_palette.png
      
      python ./zero_shot_scene_segmentation/models/AFB-URR/AFB-URR/unify_pretrain_dataset.py --name COCO --src ./zero_shot_scene_segmentation/datasets/raw_data/pretraining/static/COCO/ --dst ./zero_shot_scene_segmentation/datasets/raw_data/pretraining/static/unified/ --palette ./zero_shot_scene_segmentation/models/AFB-URR/AFB-URR/assets/mask_palette.png

      ln -s $PWD/zero_shot_scene_segmentation/datasets/raw_data/pretraining/static/unified/* ./zero_shot_scene_segmentation/models/aot-benchmark/paot-benchmark/datasets/Static/
      ```

#### DAVIS

<details open>
<summary>Download From AWS Cloud</summary>
    
```
aws s3 cp s3://prism-intern-anthony/raw_data/pretraining/DAVIS/DAVIS-2017-trainval-480p.zip ./zero_shot_scene_segmentation/datasets/raw_data/pretraining/DAVIS/ && \
  unzip ./zero_shot_scene_segmentation/datasets/raw_data/pretraining/DAVIS/DAVIS-2017-trainval-480p.zip -d ./zero_shot_scene_segmentation/datasets/raw_data/pretraining/DAVIS/ && \
    rm ./zero_shot_scene_segmentation/datasets/raw_data/pretraining/DAVIS/DAVIS-2017-trainval-480p.zip
```

</details>


<details>
<summary> Download From Open Internet</summary>
    
```
wget https://data.vision.ee.ethz.ch/csergi/share/davis/DAVIS-2017-trainval-480p.zip -P ./zero_shot_scene_segmentation/datasets/raw_data/ && \
  unzip ./zero_shot_scene_segmentation/datasets/raw_data/DAVIS-2017-trainval-480p.zip -d ./zero_shot_scene_segmentation/datasets/raw_data/ && \
    rm ./zero_shot_scene_segmentation/datasets/raw_data/DAVIS-2017-trainval-480p.zip
```
</details>


#### YouTube-VOS

<details open>
<summary>Download From AWS Cloud</summary>
    
```
aws s3 cp s3://prism-intern-anthony/raw_data/DAVIS-2017-trainval-480p.zip ./zero_shot_scene_segmentation/datasets/raw_data/ && \
  unzip ./zero_shot_scene_segmentation/datasets/raw_data/DAVIS-2017-trainval-480p.zip -d ./zero_shot_scene_segmentation/datasets/raw_data/ && \
    rm ./zero_shot_scene_segmentation/datasets/raw_data/DAVIS-2017-trainval-480p.zip
```

</details>


<details>
<summary> Download From Open Internet</summary>

Follow download instructions from [aot-benchmark](https://github.com/yoxu515/aot-benchmark#getting-started)

Link for 2018: https://drive.google.com/drive/folders/1bI5J1H3mxsIGo7Kp-pPZU8i6rnykOw7f

Link for 2019: https://drive.google.com/drive/folders/1BWzrCWyPEmBEKm0lOHe5KLuBuQxUSwqz

</details>




#### [VIPOSeg-Benchmark](https://aihabitat.org/datasets/hm3d-semantics/)


<details open>
<summary>Download From AWS Cloud</summary>
    
```
export UNZIP_DISABLE_ZIPBOMB_DETECTION=TRUE && \
  aws s3 sync s3://prism-intern-anthony/raw_data/VIPOSeg/ ./zero_shot_scene_segmentation/datasets/raw_data/VIPOSeg/ && \
    unzip ./zero_shot_scene_segmentation/datasets/raw_data/VIPOSeg/VIPOSeg_train.zip -d ./zero_shot_scene_segmentation/datasets/raw_data/VIPOSeg/ && \
      rm ./zero_shot_scene_segmentation/datasets/raw_data/VIPOSeg/VIPOSeg_train.zip && \
    unzip ./zero_shot_scene_segmentation/datasets/raw_data/VIPOSeg/VIPOSeg_valid.zip -d ./zero_shot_scene_segmentation/datasets/raw_data/VIPOSeg/ && \
      rm ./zero_shot_scene_segmentation/datasets/raw_data/VIPOSeg/VIPOSeg_valid.zip && \
unset UNZIP_DISABLE_ZIPBOMB_DETECTION


ln -s $PWD/zero_shot_scene_segmentation/datasets/raw_data/VIPOSeg/ ./zero_shot_scene_segmentation/models/aot-benchmark/paot-benchmark/datasets/
```

</details>


<details>
<summary> Download From Open Internet</summary>
    
```
export UNZIP_DISABLE_ZIPBOMB_DETECTION=TRUE && \
  source ./envs/data-processing/bin/activate && \
    gdown 1GdhTyV8s6wJi8HnlncBWoI2gb_CmrbS1 -O ./zero_shot_scene_segmentation/datasets/raw_data/VIPOSeg/ && \
      unzip ./zero_shot_scene_segmentation/datasets/raw_data/VIPOSeg/VIPOSeg_train.zip -d ./zero_shot_scene_segmentation/datasets/raw_data/VIPOSeg/ && \
        rm ./zero_shot_scene_segmentation/datasets/raw_data/VIPOSeg/VIPOSeg_train.zip && \
    gdown 1E6cB6FqXhLKT6N5_NEXO7QckwH45IWU2 -O zero_shot_scene_segmentation/datasets/raw_data/VIPOSeg/ && \
      unzip ./zero_shot_scene_segmentation/datasets/raw_data/VIPOSeg/VIPOSeg_valid.zip -d ./zero_shot_scene_segmentation/datasets/raw_data/VIPOSeg/ && \
        rm ./zero_shot_scene_segmentation/datasets/raw_data/VIPOSeg/VIPOSeg_valid.zip && \
  deactivate && \
unset UNZIP_DISABLE_ZIPBOMB_DETECTION
```

</details>