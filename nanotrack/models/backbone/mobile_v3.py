import torch
import torch.nn as nn
import math

__all__ = ['mobilenetv3_small']


def _make_divisible(v, divisor, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


##(1) Original training method
# class h_sigmoid(nn.Module):
#     def __init__(self, inplace=True):
#         super(h_sigmoid, self).__init__()
#         self.relu = nn.ReLU6(inplace=inplace)

#     def forward(self, x):
#         return self.relu(x + 3) / 6

# class h_swish(nn.Module):
#     def __init__(self, inplace=True):
#         super(h_swish, self).__init__()
#         self.sigmoid = h_sigmoid(inplace=inplace)

#     def forward(self, x):
#         return x * self.sigmoid(x)

# (2) If you want to run faster, before you convert to onnx model, you should use the following operators，since onnx can better optimize them
class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.hard_sigmoid = nn.Hardsigmoid(inplace=inplace)

    def forward(self, x):
        return self.hard_sigmoid(x)


class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.hard_swish = nn.Hardswish(inplace=True)

    def forward(self, x):
        return self.hard_swish(x)


class SELayer(nn.Module):
    def __init__(self, channel, reduction=4):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, _make_divisible(channel // reduction, 8)),
            nn.ReLU(inplace=True),
            nn.Linear(_make_divisible(channel // reduction, 8), channel),
            h_sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


def conv_3x3_bn(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        h_swish()
    )


def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        h_swish()
    )


class InvertedResidual(nn.Module):
    def __init__(self, inp, hidden_dim, oup, kernel_size, stride, use_se, use_hs):
        super(InvertedResidual, self).__init__()
        assert stride in [1, 2]

        self.identity = stride == 1 and inp == oup

        if inp == hidden_dim:
            self.conv = nn.Sequential(
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, kernel_size, stride, (kernel_size - 1) // 2, groups=hidden_dim,
                          bias=False),
                nn.BatchNorm2d(hidden_dim),
                h_swish() if use_hs else nn.ReLU(inplace=True),
                # Squeeze-and-Excite
                SELayer(hidden_dim) if use_se else nn.Identity(),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )
        else:
            self.conv = nn.Sequential(
                # pw
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                h_swish() if use_hs else nn.ReLU(inplace=True),
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, kernel_size, stride, (kernel_size - 1) // 2, groups=hidden_dim,
                          bias=False),
                nn.BatchNorm2d(hidden_dim),
                # Squeeze-and-Excite
                SELayer(hidden_dim) if use_se else nn.Identity(),
                h_swish() if use_hs else nn.ReLU(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )

    def forward(self, x):
        if self.identity:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileNetV3(nn.Module):
    def __init__(self, cfgs, mode, num_classes=1000, width_mult=1., used_layers=[2, 3, 4]):
        super(MobileNetV3, self).__init__()
        # setting of inverted residual blocks
        self.cfgs = cfgs
        assert mode in ['large', 'small']

        # building first layer
        input_channel = _make_divisible(16 * width_mult, 8)
        layers = [conv_3x3_bn(3, input_channel, 2)]

        # building inverted residual blocks
        block = InvertedResidual
        for k, t, c, use_se, use_hs, s in self.cfgs:
            output_channel = _make_divisible(c * width_mult, 8)
            exp_size = _make_divisible(input_channel * t, 8)
            layers.append(block(input_channel, exp_size, output_channel, k, s, use_se, use_hs))
            input_channel = output_channel
        self.features = nn.Sequential(*layers)

        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


def mobilenetv3_small(**kwargs):
    """
    Constructs a MobileNetV3-Small model 
    """
    cfgs = [  #
        # k, t, c, SE, HS, s 
        [3, 1, 16, 1, 0, 2],
        [3, 4.5, 24, 0, 0, 2],
        [3, 3.67, 24, 0, 0, 1],
        [5, 4, 40, 1, 1, 2],
        [5, 6, 40, 1, 1, 1],
        [5, 6, 40, 1, 1, 1],
        [5, 3, 48, 1, 1, 1],
        [5, 3, 48, 1, 1, 1],
    ]

    return MobileNetV3(cfgs, mode='small', **kwargs)


def mobilenetv3_small_v3(**kwargs):
    """
    Constructs a MobileNetV3-Small model 
    """
    cfgs = [
        # k, t, c, SE, HS, s 
        [3, 1, 16, 1, 0, 2],
        [3, 4.5, 24, 0, 0, 2],
        [3, 3.67, 24, 0, 0, 1],
        [5, 4, 40, 1, 1, 2],
        [5, 6, 40, 1, 1, 1],
        [5, 6, 40, 1, 1, 1],
        [5, 3, 48, 1, 1, 1],
        [5, 3, 48, 1, 1, 1],
        [5, 6, 96, 1, 1, 1],  # s=2 --> s=1
    ]

    return MobileNetV3(cfgs, mode='small', **kwargs)


from torchsummary import summary

if __name__ == '__main__':
    # print(_make_divisible(13, 11))
    net = mobilenetv3_small().cuda()
    summary(net, (3, 127, 127))
    print(net)

"""
MobileNetV3(
  (features): Sequential(
    (0): Sequential(
      (0): Conv2d(3, 16, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
      (1): BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (2): h_swish(
        (hard_swish): Hardswish()
      )
    )
    (1): InvertedResidual(
      (conv): Sequential(
        (0): Conv2d(16, 16, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), groups=16, bias=False)
        (1): BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (2): ReLU(inplace=True)
        (3): SELayer(
          (avg_pool): AdaptiveAvgPool2d(output_size=1)
          (fc): Sequential(
            (0): Linear(in_features=16, out_features=8, bias=True)
            (1): ReLU(inplace=True)
            (2): Linear(in_features=8, out_features=16, bias=True)
            (3): h_sigmoid(
              (hard_sigmoid): Hardsigmoid()
            )
          )
        )
        (4): Conv2d(16, 16, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (5): BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
    (2): InvertedResidual(
      (conv): Sequential(
        (0): Conv2d(16, 72, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (1): BatchNorm2d(72, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (2): ReLU(inplace=True)
        (3): Conv2d(72, 72, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), groups=72, bias=False)
        (4): BatchNorm2d(72, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (5): Identity()
        (6): ReLU(inplace=True)
        (7): Conv2d(72, 24, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (8): BatchNorm2d(24, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
    (3): InvertedResidual(
      (conv): Sequential(
        (0): Conv2d(24, 88, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (1): BatchNorm2d(88, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (2): ReLU(inplace=True)
        (3): Conv2d(88, 88, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=88, bias=False)
        (4): BatchNorm2d(88, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (5): Identity()
        (6): ReLU(inplace=True)
        (7): Conv2d(88, 24, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (8): BatchNorm2d(24, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
    (4): InvertedResidual(
      (conv): Sequential(
        (0): Conv2d(24, 96, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (1): BatchNorm2d(96, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (2): h_swish(
          (hard_swish): Hardswish()
        )
        (3): Conv2d(96, 96, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2), groups=96, bias=False)
        (4): BatchNorm2d(96, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (5): SELayer(
          (avg_pool): AdaptiveAvgPool2d(output_size=1)
          (fc): Sequential(
            (0): Linear(in_features=96, out_features=24, bias=True)
            (1): ReLU(inplace=True)
            (2): Linear(in_features=24, out_features=96, bias=True)
            (3): h_sigmoid(
              (hard_sigmoid): Hardsigmoid()
            )
          )
        )
        (6): h_swish(
          (hard_swish): Hardswish()
        )
        (7): Conv2d(96, 40, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (8): BatchNorm2d(40, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
    (5): InvertedResidual(
      (conv): Sequential(
        (0): Conv2d(40, 240, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (1): BatchNorm2d(240, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (2): h_swish(
          (hard_swish): Hardswish()
        )
        (3): Conv2d(240, 240, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=240, bias=False)
        (4): BatchNorm2d(240, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (5): SELayer(
          (avg_pool): AdaptiveAvgPool2d(output_size=1)
          (fc): Sequential(
            (0): Linear(in_features=240, out_features=64, bias=True)
            (1): ReLU(inplace=True)
            (2): Linear(in_features=64, out_features=240, bias=True)
            (3): h_sigmoid(
              (hard_sigmoid): Hardsigmoid()
            )
          )
        )
        (6): h_swish(
          (hard_swish): Hardswish()
        )
        (7): Conv2d(240, 40, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (8): BatchNorm2d(40, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
    (6): InvertedResidual(
      (conv): Sequential(
        (0): Conv2d(40, 240, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (1): BatchNorm2d(240, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (2): h_swish(
          (hard_swish): Hardswish()
        )
        (3): Conv2d(240, 240, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=240, bias=False)
        (4): BatchNorm2d(240, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (5): SELayer(
          (avg_pool): AdaptiveAvgPool2d(output_size=1)
          (fc): Sequential(
            (0): Linear(in_features=240, out_features=64, bias=True)
            (1): ReLU(inplace=True)
            (2): Linear(in_features=64, out_features=240, bias=True)
            (3): h_sigmoid(
              (hard_sigmoid): Hardsigmoid()
            )
          )
        )
        (6): h_swish(
          (hard_swish): Hardswish()
        )
        (7): Conv2d(240, 40, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (8): BatchNorm2d(40, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
    (7): InvertedResidual(
      (conv): Sequential(
        (0): Conv2d(40, 120, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (1): BatchNorm2d(120, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (2): h_swish(
          (hard_swish): Hardswish()
        )
        (3): Conv2d(120, 120, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=120, bias=False)
        (4): BatchNorm2d(120, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (5): SELayer(
          (avg_pool): AdaptiveAvgPool2d(output_size=1)
          (fc): Sequential(
            (0): Linear(in_features=120, out_features=32, bias=True)
            (1): ReLU(inplace=True)
            (2): Linear(in_features=32, out_features=120, bias=True)
            (3): h_sigmoid(
              (hard_sigmoid): Hardsigmoid()
            )
          )
        )
        (6): h_swish(
          (hard_swish): Hardswish()
        )
        (7): Conv2d(120, 48, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (8): BatchNorm2d(48, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
    (8): InvertedResidual(
      (conv): Sequential(
        (0): Conv2d(48, 144, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (1): BatchNorm2d(144, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (2): h_swish(
          (hard_swish): Hardswish()
        )
        (3): Conv2d(144, 144, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=144, bias=False)
        (4): BatchNorm2d(144, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (5): SELayer(
          (avg_pool): AdaptiveAvgPool2d(output_size=1)
          (fc): Sequential(
            (0): Linear(in_features=144, out_features=40, bias=True)
            (1): ReLU(inplace=True)
            (2): Linear(in_features=40, out_features=144, bias=True)
            (3): h_sigmoid(
              (hard_sigmoid): Hardsigmoid()
            )
          )
        )
        (6): h_swish(
          (hard_swish): Hardswish()
        )
        (7): Conv2d(144, 48, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (8): BatchNorm2d(48, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
  )
)

kernel size, stride, padding, size, Rf, stride
3,2,1, size: 128 Rf: 3 stride: 2
3,2,1, size: 64 Rf: 7 stride: 4
3,2,1, size: 32 Rf: 15 stride: 8
3,1,1, size: 32 Rf: 31 stride: 8
5,2,2, size: 16 Rf: 63 stride: 16
5,1,2, size: 16 Rf: 127 stride: 16
5,1,2, size: 16 Rf: 191 stride: 16
5,1,2, size: 16 Rf: 255 stride: 16
5,1,2, size: 16 Rf: 319 stride: 16
"""













































