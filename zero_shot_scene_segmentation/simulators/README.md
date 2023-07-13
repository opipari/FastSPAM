
```
{
  "videos"        : [video],
  "annotations"   : [video_annotation],
  "categories"    : [license],
  "instances"     : [instance],
}


video{
  "video_id"       : str,
  "images"         : [image],
}

video_annotation{
  "video_id"        : str,
  "annotations"     : [annotation],
}


image{
  "id"              : int,
  "width"           : int,
  "height"          : int,
  "file_name"       : str,
  "depth_file_name" : str,
  "scene_id"        : str,
  "camera_position" : [float, float, float], # Position(X, Y, Z)
  "camera_rotation" : [float, float, float, float], # Quaternion(W, X, Y, Z)
}

annotation{
  "image_id"        : int,
  "file_name"       : str,
  "segments_info"   : [segment_info],
}

segment_info{
  "id"              : int,
  "category_id"     : int,
  "area"            : int,
  "bbox"            : [x,y,width,height],
  "iscrowd"         : 0 or 1,
  "instance_id"     : int,
}

instance{
  "id"              : int,
  "category_id"     : int,
  "raw_category"    : str,
  "color"           : [R,G,B], # Color from Matterport source scene
  "scene_id"        : str,
}

category{
  "id"              : int,
  "name"            : str,
  "supercategory"   : str,
  "isthing"         : 0 or 1,
  "color"           : [R,G,B],
}


```
