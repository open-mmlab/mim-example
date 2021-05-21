# MMClassification with Custom Backbone

English | [简体中文](README_zh-CN.md)

This example demonstrates how to train / test a model with a custom backbone using MMClassification. All you need to do is to create two files:

1. A python file that defines the custom backbone
2. A config file that defines the model architecture, training and testing settings

### 1. Define and register the backbone

```python
from mmcli.utils import exit_with_error

try:
    # Import the backbone registry from mmcls
    from mmcls.models.builder import BACKBONES
    from mmcv.cnn import ConvModule, constant_init, kaiming_init
    from torch import nn
except ImportError:
    exit_with_error('Please install mmcls, mmcv, torch to run this example.')


# Use the decorator to register the new backbone in mmcls BACKBONES registry
@BACKBONES.register_module()
class CustomNet(nn.Module):
    # The definition of the custom backbone.
    # Three methods should be implemented: __init__, init_weights and forward
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels

        conv_param = dict(
            kernel_size=3,
            stride=2,
            conv_cfg=dict(type='Conv2d'),
            norm_cfg=dict(type='BN2d'),
            act_cfg=dict(type='ReLU'))

        self.conv1 = ConvModule(3, 16, **conv_param)
        self.conv2 = ConvModule(16, 32, **conv_param)
        self.conv3 = ConvModule(32, 64, **conv_param)

    def init_weights(self, pretrained=None):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                kaiming_init(m)
            elif isinstance(m, nn.BatchNorm2d):
                constant_init(m, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return x

```

### 2. Add a config file to use the custom backbone

```python
# At first, you need to import the custom_net in the config file, to register the custom backbone in mmcls
custom_imports = dict(imports=['custom_net'], allow_failed_imports=False)

model = dict(
    type='ImageClassifier',
    # Use the custom backbone
    backbone=dict(type='CustomNet', in_channels=3),
    neck=dict(type='GlobalAveragePooling'),
    head=dict(
        type='LinearClsHead',
        num_classes=10,
        in_channels=64,
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
    ))

# The rest part of the config
```

After that, you can train / test the classifier based on the custom backbone with following commands:

```shell
# Training
mim train mmcls custom_net_config.py --gpus 1 --work-dir MyExperiment
# Testing
mim test mmcls custom_net_config.py --checkpoint ckpt.pth --gpus 1 --metrics accuracy
```

Here we directly use [MIM](https://github.com/open-mmlab/mim) to launch the training and testing.
Actually, MIM provides more fascinating functionalities than that, for more details, please refer to the [documentation of MIM](https://mim.readthedocs.io/en/latest/)
