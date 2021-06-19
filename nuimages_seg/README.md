# Use nuImages Dataset for Semantic Segmentation

In this example, we will show how to integrate a new dataset with OpenMMLab projects to perform vision tasks.
Here we take the nuImages dataset as an example and show how to use it for semantic segmentation with MMSegmentation.
There are basically four steps to go as below, and we will do it step by step.

1. Preprocess the dataset
2. Implement a new dataset class
3. Modify config file to use it
4. Train and test a model

The key files are listed as below

```
project
├── configs
│   ├── _base_
│   │   ├── datasets
│   │   │   └── nuimages.py
│   │   ├── default_runtime.py
│   │   ├── models
│   │   │   └── pspnet_r50-d8.py
│   │   └── schedules
│   │       └── schedule_80k.py
│   └── pspnet
│       └── pspnet_r18-d8_512x1024_80k_nuim.py
├── nuim_converter.py
└── nuim_dataset.py
```

## Preprocess the dataset

According to the [official documentation](https://www.nuscenes.org/nuimages).
The semantic masks are stored in the two json files and we need to extract and convert them to the segmentation maps in PNG format used in training.

```bash
python -u nuim_converter.py \
    --data-root $DATA \
    --versions $VERSIONS \
    --out-dir $OUT \
    --nproc $NUM_WORKERS
```

Arguments description:

- `DATA`: the root of raw data path, by default it is `data/nuimages/`
- `VERSIONS`: versions of the dataset, can be the combination of `v1.0-val`, `v1.0-train`, `v1.0-mini`, `v1.0-test`.
- `OUT`: the output path of the extracted annotation files, by default it is `data/nuimages/annotations`
- `NUM_WORKERS`: the number of parallel workers that extract the segmentation map

After extracting the segmentation maps, you can get the png masks and the json file containing the data split.

```
project
├── nuimages
│   ├── annotations
│   │   ├── nuimages_v1.0-mini.json
│   │   ├── nuimages_v1.0-train.json
│   │   ├── nuimages_v1.0-val.json
│   │   ├── nuimages_v1.0-val2400.json
│   │   ├── nuimages_v1.0-test.json
│   │   └── semantic_masks
│   │       ├── xxxxx.png
│   │       └── xxxxx.png
│   └── samples
```

## Implement a new dataset class

Then we need to implement a new dataset class `NuImagesDataset`, the key implementation of the class is as below.

```python
import os.path as osp

import mmcv
from mmcv.utils import print_log
from mmseg.datasets import CustomDataset
from mmseg.datasets.builder import DATASETS
from mmseg.utils import get_root_logger


@DATASETS.register_module()
class NuImagesDataset(CustomDataset):
    CLASSES = ()

    def load_annotations(self, img_dir, img_suffix, ann_dir,
                         seg_map_suffix, split):

        annotations = mmcv.load(split)
        img_infos = []
        for img in annotations['images']:
            img_info = dict(filename=img['file_name'])
            seg_map = img_info['filename'].replace(
                img_suffix, seg_map_suffix)
            img_info['ann'] = dict(
                seg_map=osp.join('semantic_masks', seg_map))
            img_infos.append(img_info)

        print_log(
            f'Loaded {len(img_infos)} images from {ann_dir}',
            logger=get_root_logger())
        return img_infos

```

### Explanation

1. The first important thing is to register this class into the `DATASET` registry by `@DATASETS.register_module()`. With this decorator, no matter where this file is put, it will register the `NuImagesDataset` into the `DATASET` registry in MMSegmentation, as long as this file is imported by python.

2. `NuImagesDataset` inherits from `CustomDataset` and only needs to override the `load_annotations` function to load the json file we produced in the previous step.
This function converts the annotations to the middle format in MMSegmentation.
The other functionalities of the dataset class like evaluation are implemented in the `CustomDataset` and can be reused here.

## Modify Config

We can implement a base config of nuImages dataset thus it can be used in multiple configs of different models.
The base config is implemented in `configs/_base_/datasets/nuimages.py`

```python
dataset_type = 'NuImagesDataset'
data_root = 'data/nuimages/'
train_pipeline = [
    ...
]
test_pipeline = [
    ...
]
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='',
        ann_dir='annotations/',
        split='annotations/nuimages_v1.0-train.json',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ...),
    test=dict(
        type=dataset_type,
        ...))

custom_imports = dict(
    imports=['nuim_dataset'],
    allow_failed_imports=False)
```

The key step is to define the `custom_imports` so that MMCV will import the file specified in the list of `imports` when loading the config.
This will load the file `nuim_dataset.py` we implemented in the previous step
so that the `NuImagesDataset` can be registered into the `DATASET` registry in MMSegmentation.

The whole config to run a model can be found in `configs/pspnet/pspnet_r18-d8_512x1024_80k_nuim.py` which defined the model and dataset to run.

```python
_base_ = [
    '../_base_/models/pspnet_r50-d8.py',
    '../_base_/datasets/nuimages.py',
    '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_80k.py'
]

model = dict(
    pretrained='open-mmlab://resnet18_v1c',
    backbone=dict(depth=18),
    decode_head=dict(
        in_channels=512,
        channels=128,
        num_classes=25),
    auxiliary_head=dict(
        in_channels=256,
        channels=64,
        num_classes=25))

# use cityscapes pre-trained models
load_from = (
    'https://download.openmmlab.com/mmsegmentation/v0.5/pspnet/'
    'pspnet_r18-d8_512x1024_80k_cityscapes/'
    'pspnet_r18-d8_512x1024_80k_cityscapes_20201225_021458-09ffa746.pth')

```

## Train and test the model

Finally, we can train the model through the following command

```bash
PYTHONPATH='.'$PYTHONPATH mim train mmseg \
    configs/pspnet/pspnet_r18-d8_512x1024_80k_nuim.py
    --work-dir $WORK_DIR \
    --launcher slurm -G 8 -p $PARTITION
```

We can test the model through the following command

```bash
PYTHONPATH='.'$PYTHONPATH mim test mmseg \
    configs/pspnet/pspnet_r18-d8_512x1024_80k_nuim.py
    --checkpoint $WORK_DIR/latest.pth \
    --launcher slurm -G 8 -p $PARTITION \
    --eval mIoU
```
