# Preprocess the gait index from video.

We first use the YOLO series method to get the gait cycle index from video, and save the results to .json file.

The .sjon mapping file will be used in the after process, for training the model.

## JSON file

🗒️ Because the .pt file is too large, so I use the json file to store the information of the gait cycle index.

## Format

**This script is used to the segmentation_dataset_512 dataset.**

To define the gait cycle in the video, and save the gait cycle index to json file.

The json file include: (in dict)

``` python   
{
    "video_name": the video name,
    "video_path": the video path, relative path from /workspace/skeleton/data/segmentation_dataset_512,
    "frame_count": the raw frames of the video,
    "label": the label of the video,
    "disease": the disease of the video,
    "gait_cycle_index_bbox": the gait cycle index,
    "bbox_none_index": the bbox none index, when use yolo to get the bbox, some frame will not get the bbox.
    "bbox": the bbox, [n, 4] (cxcywh)
}
```

## Pipeline

The python file flow is:

``` mermaid
graph LR
    main.py --> preprocess.py --> yolov8.py
```

The data process pipeline is:

``` mermaid
graph LR
    A[Video] --> B[YOLO] --> C[BBOX] 
    B --> D[Keypoint]
    C --> E[Mix Method]
    D --> E[Mix Method]
    E --> F[Save to .json file]
```

## Usage

``` bash
python main.py
```

### Info 
The code is from https://github.com/ChenKaiXuSan/Skeleton_ASD_PyTorch/tree/master/prepare_skeleton_dataset and https://github.com/ChenKaiXuSan/Skeleton_ASD_PyTorch/tree/master/prepare_gait_cycle_index


<!-- The results will be saved in the `/workspace/dataset/seg_skeleton_pkl` folder. -->
<!-- The json file will be saved in the `/workspace/dataset/seg_skeleton_json` folder. -->
The train dataset will be saved in the `/workspace/dataset/pose_attn_map_dataset` folder.