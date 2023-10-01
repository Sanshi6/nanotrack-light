import torch
import torch.nn as nn
from nanotrack.models.backbone.MobileV3Large import MobileOneBlock
import torch.nn.functional as F


# if kernel = 1, stride = 2, lost a lot information.


class SEBlock(nn.Module):
    """ Squeeze and Excite module.

        Pytorch implementation of `Squeeze-and-Excitation Networks` -
        https://arxiv.org/pdf/1709.01507.pdf
    """

    def __init__(self,
                 in_channels: int,
                 rd_ratio: float = 0.0625) -> None:
        """ Construct a Squeeze and Excite Module.

        :param in_channels: Number of input channels.
        :param rd_ratio: Input channel reduction ratio.
        """
        super(SEBlock, self).__init__()
        self.activate = nn.Hardsigmoid()
        self.reduce = nn.Conv2d(in_channels=in_channels,
                                out_channels=int(in_channels * rd_ratio),
                                kernel_size=1,
                                stride=1,
                                bias=True)
        self.expand = nn.Conv2d(in_channels=int(in_channels * rd_ratio),
                                out_channels=in_channels,
                                kernel_size=1,
                                stride=1,
                                bias=True)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """ Apply forward pass. """
        b, c, h, w = inputs.size()
        x = F.avg_pool2d(inputs, kernel_size=[h, w])
        x = self.reduce(x)
        x = F.relu(x)
        x = self.expand(x)
        x = self.activate(x)
        # x = nn.sigmoid(x)
        x = x.view(-1, c, 1, 1)
        return inputs * x


class MobileV3Block(nn.Module):
    def __init__(self, in_channel, exp_channel, out_channel, kernel, stride, use_activate, use_se):
        super(MobileV3Block, self).__init__()
        self.padding = kernel // 2

        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=exp_channel, kernel_size=1, stride=1)
        self.bn1 = nn.BatchNorm2d(exp_channel)
        if use_activate == 'ReLU6':
            self.act1 = nn.ReLU6(inplace=True)
        else:
            self.act1 = nn.Hardswish(inplace=True)

        self.conv2 = nn.Conv2d(in_channels=exp_channel, out_channels=exp_channel, kernel_size=kernel, stride=stride,
                               padding=self.padding, groups=exp_channel)
        self.bn2 = nn.BatchNorm2d(exp_channel)
        if use_activate == 'ReLU6':
            self.act2 = nn.ReLU6(inplace=True)
        else:
            self.act2 = nn.Hardswish(inplace=True)

        if use_se:
            self.se1 = SEBlock(exp_channel)
        else:
            self.se1 = nn.Identity()

        self.conv3 = nn.Conv2d(in_channels=exp_channel, out_channels=out_channel, kernel_size=1, stride=1)
        self.bn3 = nn.BatchNorm2d(out_channel)
        if use_activate == 'ReLU6':
            self.act3 = nn.ReLU6(inplace=True)
        else:
            self.act3 = nn.Hardswish(inplace=True)

        self.conv4 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel, kernel_size=kernel, stride=1,
                               padding=self.padding, groups=out_channel)
        self.bn4 = nn.BatchNorm2d(out_channel)
        if use_activate == 'ReLU6':
            self.act4 = nn.ReLU6(inplace=True)
        else:
            self.act4 = nn.Hardswish(inplace=True)
        if use_se:
            self.se1 = SEBlock(exp_channel)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')
                # m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.act2(x)

        x = self.se1(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.act3(x)
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.act4(x)
        return x


class MobileRepV3(nn.Module):
    def __init__(self, inference_mode):
        super(MobileRepV3, self).__init__()
        self.MobileRepV3Block = nn.Sequential(
            # 3 -> 16
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.Hardswish(inplace=True),

            # 16 -> 16 -> 16
            MobileV3Block(in_channel=16, exp_channel=16, out_channel=16, kernel=3, stride=1,
                          use_activate='ReLU', use_se=False),

            # 16 -> 64 -> 24
            MobileV3Block(in_channel=16, exp_channel=64, out_channel=24, kernel=3, stride=2,
                          use_activate='ReLU', use_se=False),

            # 24 -> 72 -> 24
            MobileV3Block(in_channel=24, exp_channel=72, out_channel=24, kernel=3, stride=1,
                          use_activate='ReLU', use_se=False),

            # 24 -> 72 -> 40
            MobileV3Block(in_channel=24, exp_channel=72, out_channel=40, kernel=5, stride=2,
                          use_activate='ReLU', use_se=True),

            # 40 -> 120 -> 40
            MobileV3Block(in_channel=40, exp_channel=120, out_channel=40, kernel=5, stride=1,
                          use_activate='ReLU', use_se=True),

            # 40 -> 120 -> 40
            MobileV3Block(in_channel=40, exp_channel=120, out_channel=40, kernel=5, stride=1,
                          use_activate='ReLU', use_se=True),

            # 40 -> 240 -> 80
            MobileV3Block(in_channel=40, exp_channel=240, out_channel=80, kernel=3, stride=2,
                          use_activate='HardSigmoid', use_se=False),

            # 80 -> 200 -> 80
            MobileV3Block(in_channel=80, exp_channel=240, out_channel=80, kernel=3, stride=1,
                          use_activate='HardSigmoid', use_se=False),

            # 80 -> 184 -> 80
            MobileV3Block(in_channel=80, exp_channel=184, out_channel=80, kernel=3, stride=1,
                          use_activate='HardSigmoid', use_se=False),

            # 80 -> 184 -> 80
            MobileV3Block(in_channel=80, exp_channel=184, out_channel=80, kernel=3, stride=1,
                          use_activate='HardSigmoid', use_se=False),

            # 80 -> 480 -> 112
            MobileV3Block(in_channel=80, exp_channel=480, out_channel=112, kernel=3, stride=1,
                          use_activate='HardSigmoid', use_se=True),

            # 112 -> 672 -> 112
            MobileV3Block(in_channel=112, exp_channel=672, out_channel=112, kernel=3, stride=1,
                          use_activate='HardSigmoid', use_se=True),

            # 112 -> 672 -> 160
            MobileV3Block(in_channel=112, exp_channel=672, out_channel=160, kernel=5, stride=1,
                          use_activate='HardSigmoid', use_se=True),

            # 160 -> 960 -> 160
            MobileV3Block(in_channel=160, exp_channel=960, out_channel=160, kernel=5, stride=1,
                          use_activate='HardSigmoid', use_se=True),

            # 160 -> 960 -> 160
            MobileV3Block(in_channel=160, exp_channel=960, out_channel=160, kernel=5, stride=1,
                          use_activate='HardSigmoid', use_se=True),

            nn.Conv2d(in_channels=160, out_channels=960, kernel_size=1, stride=1),


            # MobileOneBlock(in_channels=16, out_channels=72, kernel_size=1, stride=1,
            #                padding=0, groups=1, inference_mode=inference_mode, use_se=False, num_conv_branches=1),
            # MobileOneBlock(in_channels=72, out_channels=72, kernel_size=3, stride=2,
            #                padding=1, groups=72, inference_mode=inference_mode, use_se=False, num_conv_branches=1),
            #
            # MobileOneBlock(in_channels=72, out_channels=24, kernel_size=1, stride=1,
            #                padding=0, groups=1, inference_mode=inference_mode, use_se=False, num_conv_branches=1),
            # MobileOneBlock(in_channels=24, out_channels=24, kernel_size=3, stride=1,
            #                padding=1, groups=24, inference_mode=inference_mode, use_se=False, num_conv_branches=1),
            #
            # MobileOneBlock(in_channels=24, out_channels=88, kernel_size=1, stride=1,
            #                padding=0, groups=1, inference_mode=inference_mode, use_se=False, num_conv_branches=1),
            # MobileOneBlock(in_channels=88, out_channels=88, kernel_size=3, stride=1,
            #                padding=1, groups=88, inference_mode=inference_mode, use_se=False, num_conv_branches=1),
            #
            # MobileOneBlock(in_channels=88, out_channels=24, kernel_size=1, stride=1,
            #                padding=0, groups=1, inference_mode=inference_mode, use_se=False, num_conv_branches=1),
            # MobileOneBlock(in_channels=24, out_channels=24, kernel_size=3, stride=1,
            #                padding=1, groups=24, inference_mode=inference_mode, use_se=False, num_conv_branches=1),
            #
            # MobileOneBlock(in_channels=24, out_channels=96, kernel_size=1, stride=1,
            #                padding=0, groups=1, inference_mode=inference_mode, use_se=False, num_conv_branches=1),
            # MobileOneBlock(in_channels=96, out_channels=96, kernel_size=3, stride=1,
            #                padding=1, groups=96, inference_mode=inference_mode, use_se=False, num_conv_branches=1),
            #
            # MobileOneBlock(in_channels=96, out_channels=40, kernel_size=1, stride=1,
            #                padding=0, groups=1, inference_mode=inference_mode, use_se=False, num_conv_branches=1),
            # MobileOneBlock(in_channels=40, out_channels=40, kernel_size=3, stride=2,
            #                padding=1, groups=40, inference_mode=inference_mode, use_se=False, num_conv_branches=1),
            #
            # MobileOneBlock(in_channels=40, out_channels=240, kernel_size=1, stride=1,
            #                padding=0, groups=1, inference_mode=inference_mode, use_se=False, num_conv_branches=1),
            # MobileOneBlock(in_channels=240, out_channels=240, kernel_size=3, stride=1,
            #                padding=1, groups=240, inference_mode=inference_mode, use_se=False, num_conv_branches=1),
            #
            # MobileOneBlock(in_channels=240, out_channels=40, kernel_size=1, stride=1,
            #                padding=0, groups=1, inference_mode=inference_mode, use_se=False, num_conv_branches=1),
            # MobileOneBlock(in_channels=40, out_channels=40, kernel_size=3, stride=1,
            #                padding=1, groups=40, inference_mode=inference_mode, use_se=False, num_conv_branches=1),
            #
            # MobileOneBlock(in_channels=40, out_channels=240, kernel_size=1, stride=1,
            #                padding=0, groups=1, inference_mode=inference_mode, use_se=False, num_conv_branches=1),
            # MobileOneBlock(in_channels=240, out_channels=240, kernel_size=3, stride=1,
            #                padding=1, groups=240, inference_mode=inference_mode, use_se=False, num_conv_branches=1),
            #
            # MobileOneBlock(in_channels=240, out_channels=40, kernel_size=1, stride=1,
            #                padding=0, groups=1, inference_mode=inference_mode, use_se=False, num_conv_branches=1),
            # MobileOneBlock(in_channels=40, out_channels=40, kernel_size=3, stride=1,
            #                padding=1, groups=40, inference_mode=inference_mode, use_se=False, num_conv_branches=1),
            #
            # MobileOneBlock(in_channels=40, out_channels=120, kernel_size=1, stride=1,
            #                padding=0, groups=1, inference_mode=inference_mode, use_se=False, num_conv_branches=1),
            # MobileOneBlock(in_channels=120, out_channels=120, kernel_size=3, stride=1,
            #                padding=1, groups=120, inference_mode=inference_mode, use_se=False, num_conv_branches=1),
            #
            # MobileOneBlock(in_channels=120, out_channels=48, kernel_size=1, stride=1,
            #                padding=0, groups=1, inference_mode=inference_mode, use_se=False, num_conv_branches=1),
            # MobileOneBlock(in_channels=48, out_channels=48, kernel_size=3, stride=1,
            #                padding=1, groups=48, inference_mode=inference_mode, use_se=False, num_conv_branches=1),
            #
            # MobileOneBlock(in_channels=48, out_channels=144, kernel_size=1, stride=1,
            #                padding=0, groups=1, inference_mode=inference_mode, use_se=False, num_conv_branches=1),
            # MobileOneBlock(in_channels=144, out_channels=144, kernel_size=3, stride=1,
            #                padding=1, groups=144, inference_mode=inference_mode, use_se=False, num_conv_branches=1),
            #
            # MobileOneBlock(in_channels=144, out_channels=48, kernel_size=1, stride=1,
            #                padding=0, groups=1, inference_mode=inference_mode, use_se=False, num_conv_branches=1),
            # MobileOneBlock(in_channels=48, out_channels=48, kernel_size=3, stride=1,
            #                padding=1, groups=48, inference_mode=inference_mode, use_se=False, num_conv_branches=1),
            #
            # MobileOneBlock(in_channels=48, out_channels=288, kernel_size=1, stride=1,
            #                padding=0, groups=1, inference_mode=inference_mode, use_se=False, num_conv_branches=1),
            # MobileOneBlock(in_channels=288, out_channels=288, kernel_size=3, stride=1,
            #                padding=1, groups=288, inference_mode=inference_mode, use_se=False, num_conv_branches=1),
            #
            # MobileOneBlock(in_channels=288, out_channels=96, kernel_size=1, stride=1,
            #                padding=0, groups=1, inference_mode=inference_mode, use_se=False, num_conv_branches=1),
            # MobileOneBlock(in_channels=96, out_channels=96, kernel_size=3, stride=2,
            #                padding=1, groups=96, inference_mode=inference_mode, use_se=False, num_conv_branches=1),
            #
            # MobileOneBlock(in_channels=96, out_channels=576, kernel_size=1, stride=1,
            #                padding=0, groups=1, inference_mode=inference_mode, use_se=False, num_conv_branches=1),
            # MobileOneBlock(in_channels=576, out_channels=576, kernel_size=3, stride=1,
            #                padding=1, groups=576, inference_mode=inference_mode, use_se=False, num_conv_branches=1),
            #
            # MobileOneBlock(in_channels=576, out_channels=96, kernel_size=1, stride=1,
            #                padding=0, groups=1, inference_mode=inference_mode, use_se=False, num_conv_branches=1),
            # MobileOneBlock(in_channels=96, out_channels=96, kernel_size=3, stride=1,
            #                padding=1, groups=96, inference_mode=inference_mode, use_se=False, num_conv_branches=1),
            #
            # MobileOneBlock(in_channels=96, out_channels=576, kernel_size=1, stride=1,
            #                padding=0, groups=1, inference_mode=inference_mode, use_se=False, num_conv_branches=1),
            # MobileOneBlock(in_channels=576, out_channels=576, kernel_size=3, stride=1,
            #                padding=1, groups=576, inference_mode=inference_mode, use_se=False, num_conv_branches=1),
            #
            # nn.Conv2d(in_channels=576, out_channels=96, kernel_size=1, stride=1, padding=0, groups=1),
            # nn.BatchNorm2d(num_features=96),
            #
            # # MobileOneBlock(in_channels=576, out_channels=96, kernel_size=1, stride=1,
            # #                padding=0, groups=1, inference_mode=inference_mode, use_se=False, num_conv_branches=1),
            # # nn.Conv2d(in_channels=96, out_channels=96, kernel_size=3, stride=1, padding=1, groups=96),
            # # nn.BatchNorm2d(num_features=96),
            # # MobileOneBlock(in_channels=48, out_channels=48, kernel_size=3, stride=1,
            # #                padding=1, groups=48, inference_mode=inference_mode, use_se=False, num_conv_branches=1),

        )

    def forward(self, x):
        x = self.MobileRepV3Block(x)
        return x


# Give the model a name by adding it as a module attribute
# MobileRepV3 = nn.Module()
# MobileRepV3.add_module('block', MobileRepV3Block)
from torchsummary import summary

if __name__ == "__main__":
    # x = torch.randn(1, 3, 255, 255)
    net = MobileRepV3(inference_mode=False).cuda()
    summary(net, (3, 255, 255))
    # out = net(x)
    # print(net)
