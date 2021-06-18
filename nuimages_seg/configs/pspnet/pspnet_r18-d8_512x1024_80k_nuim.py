_base_ = [
    '../_base_/models/pspnet_r50-d8.py', '../_base_/datasets/nuimages.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_80k.py'
]

model = dict(
    pretrained='https://download.openmmlab.com/pretrain/third_party/resnet18_v1c-b5776b93.pth',
    backbone=dict(depth=18),
    decode_head=dict(
        in_channels=512,
        channels=128,
        num_classes=25),
    auxiliary_head=dict(in_channels=256, channels=64, num_classes=25))

# use cityscapes pre-trained models
load_from = 'https://download.openmmlab.com/mmsegmentation/v0.5/pspnet/pspnet_r18-d8_512x1024_80k_cityscapes/pspnet_r18-d8_512x1024_80k_cityscapes_20201225_021458-09ffa746.pth'  # noqa
evaluation = dict(interval=80000, metric='mIoU')
