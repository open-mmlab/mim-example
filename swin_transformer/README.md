# Swin Transformer for Object Detection and Segmentation

This is an unofficial implementation of Swin Transformer.
It implements Swin Transformer for object detection and segmentation tasks to show how we can use [MIM](https://github.com/open-mmlab/mim) to accelerate the research projects.

## Requirements

- MMCV-full v1.3.4
- MMDetection v2.12.0
- MMSegmentation v0.13.0

You can install them after installing mim through the following commands

```bash
mim install mmcv==1.3.4
mim install mmdet==2.12.0
mim install mmsegmentation=0.13.0
```

## Explaination

Because MMDetection and MMSegmentation inherits the model registry in MMCV since v2.12.0 and v0.13.0, we only need the implementation of swin transformer and add it into the model registry of MMCV. Then we can use it for object detection and segmentation by modifying configs.

The implementation of Swin Transformer and its pre-trained models are taken from the [official implementation]()
