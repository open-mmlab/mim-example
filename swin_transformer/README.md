# Swin Transformer for Object Detection and Segmentation

This is an unofficial implementation of Swin Transformer.
It implements Swin Transformer for object detection and segmentation tasks to show how we can use [MIM](https://github.com/open-mmlab/mim) to accelerate the research projects.

## Requirements

- MIM>=0.1.1
- MMCV-full v1.3.5
- MMDetection v2.13.0
- MMSegmentation v0.14.0
- timm

You can install them after installing mim through the following commands

```bash
pip install openmim>=0.1.1  # install mim through pypi
pip install timm  # swin transformer relies timm
mim install mmcv-full==1.3.5  # install mmcv
MKL_THREADING_LAYER=GNU mim install mmdet==2.13.0  # install mmdet to run object detection
MKL_THREADING_LAYER=GNU mim install mmsegmentation=0.14.0  # install mmseg to run semantic segmentation
```

**Note**: `MKL_THREADING_LAYER=GNU` is a workaround according to the [issue](https://github.com/pytorch/pytorch/issues/37377).

## Explaination

Because MMDetection and MMSegmentation inherits the model registry in MMCV since v2.12.0 and v0.13.0, respectively, we only need one implementation of swin transformer and add it into the model registry of MMCV. Then we can use it for object detection and segmentation by modifying configs.


### Step 1: implement Swin Transformer

The implementation of Swin Transformer and its pre-trained models are taken from the [official implementation](https://github.com/microsoft/Swin-Transformer).
The key file structure is as below:

```
swin_transformer
    |---- configs
            |---- swin_mask_rcnn  # config files to run with MMDetection
                        |---- mask_rcnn_swim-t-p4-w7_fpn_1x_coco.py
                        |---- mask_rcnn_swim-t-p4-w7_fpn_fp16_1x_coco.py
            |---- swin_upernet  # config files to run with MMSegmentation
                        |---- upernet_swin-t_512x512_160k_8x2_ade20k.py
    |---- swin
            |---- swin_checkpoint.py  # for checkout loading
            |---- swin_transformer.py  # implementation of swin transformer
```

### Step 2: register Swin Transformer into model registry

The key step that allow MMDet and MMSeg to use a unique implementation of Swin Transformer is to register the backbone into the registry in MMCV.

```python
from mmcv.cnn import MODELS


@MODELS.register_module()
class SwinTransformer(nn.Module):
    # code implementation
    def __init__(self, *args, **kwargs):
        super().__init__()
```

It essentially builds a mapping as below

```python
'SwinTransformer' -> <class 'SwinTransformer'>
```

Because MMDetection and MMSegmentation inherits the model registry in MMCV since v2.12.0 and v0.13.0, their `MODELS` registries are under descendants of the `MODELS` registry in MMCV. Therefore, such a mapping in MMDet/MMSeg becomes

```python
'mmcv.SwinTransformer' -> <class 'SwinTransformer'>
```

To enable the `MODEL.build()` in MMDet/MMSeg to correctly find the implementation of `SwinTransformer`, we need to specify the scope of the module by `mmcv.SwinTransformer` as you will see in the configs.

### Step 3: use Swin Transformer through config

To use Swin Transformer, we can simply use the config and the build function

```python
module_cfg = dict(type='mmcv.SwinTransformer')
module = build_backbone(module_cfg)
```

To run it with MMDetection or MMSegmentation, we need to define the model backbone as below

```python
model = dict(
    type='MaskRCNN',
    pretrained='./pretrain/swin/swin_tiny_patch4_window7_224.pth',
    backbone=dict(type='mmcv.SwinTransformer'))

custom_imports = dict(
    imports=['swin.swin_transformer'], allow_failed_imports=False)
```

## Usages

Assume now you are in the directory under `swin_transformer`, to run it with mmdet and slurm, we can use the command as below

```bash
PYTHONPATH='.':$PYTHONPATH mim train mmdet configs/swin_mask_rcnn/mask_rcnn_swim-t-p4-w7_fpn_fp16_1x_coco.py \--work-dir ../work_dir/mask_rcnn_swim-t-p4-w7_fpn_fp16_1x_coco.py --launcher slurm --partition $PARTITION --gpus 8 --gpus-per-node 8  --srun-args $SRUN_ARGS
```

To run it with mmseg, we can use the command as below

```bash
PYTHONPATH='.':$PYTHONPATH mim train mmseg configs/upernet/upernet_swin-t_512x512_160k_8x2_ade20k.py --work-dir ../work_dir/upernet_swin-t_512x512_160k_8x2_ade20k.py --launcher slurm --partition $PARTITION --gpus 8 --gpus-per-node 8 --srun-args $SRUN_ARGS
```

## Results

### ADE20K

| Backbone | Method | Crop Size | Lr Schd | mIoU | Config | Download |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Swin-T | UPerNet | 512x512 | 160K | 44.3 | [config](/configs/swin_upernet/upernet_swin-t_512x512_160k_8x2_ade20k.py) | [model]() &#124;  [log]() |

### COCO

| Backbone | Method | Lr Schd | Bbox mAP | Mask mAP| Config | Download |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Swin-T | Mask R-CNN | 1x| 42.6| 39.5 |[config](/configs/swin_mask_rcnn/mask_rcnn_swim-t-p4-w7_fpn_1x_coco.py) | [model](https://download.openmmlab.com/mim-example/swin_transformer/swin_mask_rcnn/mask_rcnn_swim-t-p4-w7_fpn_1x_coco/mask_rcnn_swim-t-p4-w7_fpn_1x_coco_20210612_135948-bf3d7aa4.pth) &#124;  [log](https://download.openmmlab.com/mim-example/swin_transformer/swin_mask_rcnn/mask_rcnn_swim-t-p4-w7_fpn_1x_coco/mask_rcnn_swim-t-p4-w7_fpn_1x_coco_20210612_135948.log.json) |
| Swin-T | Mask R-CNN | FP16 1x| 42.5|39.3 |[config](/configs/swin_mask_rcnn/mask_rcnn_swim-t-p4-w7_fpn_fp16_1x_coco.py) | [model](https://download.openmmlab.com/mim-example/swin_transformer/swin_mask_rcnn/mask_rcnn_swim-t-p4-w7_fpn_fp16_1x_coco/mask_rcnn_swim-t-p4-w7_fpn_fp16_1x_coco_20210612_135948-6434d76f.pth) &#124;  [log](https://download.openmmlab.com/mim-example/swin_transformer/swin_mask_rcnn/mask_rcnn_swim-t-p4-w7_fpn_fp16_1x_coco/mask_rcnn_swim-t-p4-w7_fpn_fp16_1x_coco_20210612_135948.log.json) |
