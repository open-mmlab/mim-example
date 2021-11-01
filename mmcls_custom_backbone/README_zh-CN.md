# 在 MMClassification 中使用自定义主干网络

[English](README.md) | 简体中文

这个示例演示了如何在 MMClassification 中使用自定义主干网络进行训练和测试。为达到这一目的，只需新建两个文件：

1. 一个定义了新的主干网络的 python 文件
2. 一个配置文件，定义了基于新的主干网络的识别模型，以及训练和测试的设定

### 1. 定义新的主干网络，并在 MMClassification 中注册

```python
from mim.utils import exit_with_error

try:
    # 导入 mmcls 中的注册表
    from mmcls.models.builder import BACKBONES
    from mmcv.cnn import ConvModule, constant_init, kaiming_init
    from torch import nn
except ImportError:
    exit_with_error('Please install mmcls, mmcv, torch to run this example.')


# 利用装饰器将新的主干网络注册在 mmcls 的 BACKBONES 注册表中
@BACKBONES.register_module()
class CustomNet(nn.Module):
    # 定义新的主干网络，需要实现 3 个方法：__init__, init_weights, forward

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

### 2. 添加配置文件来使用新的主干网络

```python
# 首先，需要在配置文件中导入 custom_net，将新的主干网络注册在 mmcls 中
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

# 配置文件的其余部分
```

完成之后，可以利用如下命令训练并测试新的识别模型：

```shell
# 当前工作路径为 `mim-example/mmcls_custom_backbone`
# 训练
PYTHONPATH=$PWD:$PYTHONPATH mim train mmcls custom_net_config.py --gpus 1 --work-dir MyExperiment
# 测试
PYTHONPATH=$PWD:$PYTHONPATH mim test mmcls custom_net_config.py --checkpoint ckpt.pth --gpus 1 --metrics accuracy
```

在这里，我们直接使用 [MIM](https://github.com/open-mmlab/mim) 启动了训练和测试.
实际上， MIM 提供了更多激动人心的功能, 请查看 [MIM 的文档](https://openmim.readthedocs.io/en/latest/)来了解更多细节。
