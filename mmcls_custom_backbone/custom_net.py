from mim.utils import exit_with_error

try:
    from mmcls.models.builder import BACKBONES
    from mmcv.cnn import ConvModule, constant_init, kaiming_init
    from torch import nn
except ImportError:
    exit_with_error('Please install mmcls, mmcv, torch to run this example.')


@BACKBONES.register_module()
class CustomNet(nn.Module):

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
