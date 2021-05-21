# model settings
custom_imports = dict(imports=['custom_net'], allow_failed_imports=False)

model = dict(
    type='ImageClassifier',
    backbone=dict(type='CustomNet', in_channels=3),
    neck=dict(type='GlobalAveragePooling'),
    head=dict(
        type='LinearClsHead',
        num_classes=10,
        in_channels=64,
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
    ))

# dataset settings
dataset_type = 'ImageNet'
img_norm_cfg = dict(
    mean=[125.307, 122.961, 113.8575],
    std=[51.5865, 50.847, 51.255],
    to_rgb=True)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='RandomResizedCrop', size=32, backend='pillow'),
    dict(type='RandomFlip', flip_prob=0.5, direction='horizontal'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='ToTensor', keys=['gt_label']),
    dict(type='Collect', keys=['img', 'gt_label'])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', size=(32, -1), backend='pillow'),
    dict(type='CenterCrop', crop_size=32),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='Collect', keys=['img'])
]

data = dict(
    samples_per_gpu=16,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        data_prefix='/tmp/test_dataset',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        data_prefix='/tmp/test_dataset',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        data_prefix='/tmp/test_dataset',
        pipeline=test_pipeline))

# optimizer
optimizer = dict(type='SGD', lr=0.1, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(policy='step', step=[100, 150])
runner = dict(type='EpochBasedRunner', max_epochs=1)

# checkpoint saving
checkpoint_config = dict(interval=1)
# yapf:disable
log_config = dict(
    interval=100,
    hooks=[
        dict(type='TextLoggerHook'),
    ])
# yapf:enable
# You can register your own hooks like this
# custom_hooks=[dict(type='EMAHook')]

dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
