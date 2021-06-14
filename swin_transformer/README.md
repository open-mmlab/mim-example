# Swin Transformer for Object Detection and Segmentation

This is an unofficial implementation of Swin Transformer.
It implements Swin Transformer for object detection and segmentation tasks to show how we can use [MIM](https://github.com/open-mmlab/mim) to accelerate the research projects.

## Requirements

- MIM 0.1.0
- MMCV-full v1.3.5
- MMDetection v2.13.0
- MMSegmentation v0.14.0
- timm

You can install them after installing mim through the following commands

```bash
pip install openmim  # install mim through pypi
pip install timm  # swin transformer relies timm
mim install mmcv-full==1.3.5  # install mmcv
MKL_THREADING_LAYER=GNU mim install mmdet==2.13.0  # install mmdet to run object detection
MKL_THREADING_LAYER=GNU mim install mmsegmentation=0.14.0  # install mmseg to run semantic segmentation
```

**Note**: `MKL_THREADING_LAYER=GNU` is workaround according to the [issue](https://github.com/pytorch/pytorch/issues/37377).

## Explaination

Because MMDetection and MMSegmentation inherits the model registry in MMCV since v2.12.0 and v0.13.0, we only need the implementation of swin transformer and add it into the model registry of MMCV. Then we can use it for object detection and segmentation by modifying configs.

The implementation of Swin Transformer and its pre-trained models are taken from the [official implementation](https://github.com/microsoft/Swin-Transformer)

## Usages

Assume now you are in the directory under `swin_transformer`, to run it with mmdet and slurm, we can use the command as below

```bash
PYTHONPATH='.':$PYTHONPATH mim train mmdet configs/swin_mask_rcnn/mask_rcnn_swim-t-p4-w7_fpn_fp16_1x_coco.py --work-dir ../work_dir/mask_rcnn_swim-t-p4-w7_fpn_fp16_1x_coco.py --launcher slurm --partition $PARTITION --gpus 8 --gpus-per-node 8  --srun-args ${SRUN_ARGS}
```

To run it with mmseg, we can use the command as below

```bash
PYTHONPATH='.':$PYTHONPATH mim train mmseg configs/upernet/upernet_swin-t_512x512_160k_8x2_ade20k.py --work-dir ../work_dir/upernet_swin-t_512x512_160k_8x2_ade20k.py --launcher slurm --partition $PARTITION --gpus 8 --gpus-per-node 8 --srun-args ${SRUN_ARGS}
```


## Results

### ADE20K

| Backbone | Method | Crop Size | Lr Schd | mIoU | Config | Download |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Swin-T | UPerNet | 512x512 | 160K | 44.3 | [config](swin_transformer/configs/swin_upernet/upernet_swin-t_512x512_160k_8x2_ade20k.py) | [model]() &#124;  [log]() |

### COCO

| Backbone | Method | Lr Schd | Bbox mAP | Mask mAP| Config | Download |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Swin-T | Mask R-CNN | 1x| 42.6| 39.5 |[config](swin_transformer/configs/swin_mask_rcnn/mask_rcnn_swim-t-p4-w7_fpn_1x_coco.py) | [model]() &#124;  [log]() |
| Swin-T | Mask R-CNN | FP16 1x| 42.5|39.3 |[config](swin_transformer/configs/swin_mask_rcnn/mask_rcnn_swim-t-p4-w7_fpn_fp16_1x_coco.py) | [model]() &#124;  [log]() |
