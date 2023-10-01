import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Tuple


def _initialize_weights(self):
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
        x = torch.sigmoid(x)
        x = x.view(-1, c, 1, 1)
        return inputs * x


class MobileOneBlock(nn.Module):
    """ MobileOne building block.

        This block has a multi-branched architecture at train-time
        and plain-CNN style architecture at inference time
        For more details, please refer to our paper:
        `An Improved One millisecond Mobile Backbone` -
        https://arxiv.org/pdf/2206.04040.pdf
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int,
                 stride: int = 1,
                 padding: int = 0,
                 dilation: int = 1,
                 groups: int = 1,
                 inference_mode: bool = False,
                 use_se: bool = False,
                 num_conv_branches: int = 1,
                 use_act: str = 'ReLU') -> None:
        """ Construct a MobileOneBlock module.

        :param in_channels: Number of channels in the input.
        :param out_channels: Number of channels produced by the block.
        :param kernel_size: Size of the convolution kernel.
        :param stride: Stride size.
        :param padding: Zero-padding size.
        :param dilation: Kernel dilation factor.
        :param groups: Group number.
        :param inference_mode: If True, instantiates model in inference mode.
        :param use_se: Whether to use SE-ReLU activations.
        :param num_conv_branches: Number of linear conv branches.
        :param use_act: choose activate function
        """

        # kernel = 5 / 3, activate function = ReLU6 / hard sigmoid
        super(MobileOneBlock, self).__init__()
        self.use_act = use_act
        self.inference_mode = inference_mode
        self.groups = groups
        self.stride = stride
        self.kernel_size = kernel_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_conv_branches = num_conv_branches

        # Check if SE-ReLU is requested
        if use_se:
            self.se = SEBlock(out_channels)
        else:
            self.se = nn.Identity()
        if use_act == 'ReLU':
            self.activation = nn.ReLU()
        else:
            self.activation = h_swish()

        if inference_mode:
            self.reparam_conv = nn.Conv2d(in_channels=in_channels,
                                          out_channels=out_channels,
                                          kernel_size=kernel_size,
                                          stride=stride,
                                          padding=padding,
                                          dilation=dilation,
                                          groups=groups,
                                          bias=True)
        else:
            # Re-parameterizable skip connection
            self.rbr_skip = nn.BatchNorm2d(num_features=in_channels) \
                if out_channels == in_channels and stride == 1 else None

            # Re-parameterizable conv branches
            rbr_conv = list()
            for _ in range(self.num_conv_branches):
                rbr_conv.append(self._conv_bn(kernel_size=kernel_size,
                                              padding=padding))
            self.rbr_conv = nn.ModuleList(rbr_conv)

            # Re-parameterizable scale branch
            self.rbr_scale = None
            if kernel_size > 1:
                self.rbr_scale = self._conv_bn(kernel_size=1,
                                               padding=0)
        _initialize_weights(self)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """ Apply forward pass. """
        # Inference mode forward pass.
        if self.inference_mode:
            return self.activation(self.se(self.reparam_conv(x)))

        # Multi-branched train-time forward pass.
        # Skip branch output
        identity_out = 0
        if self.rbr_skip is not None:
            identity_out = self.rbr_skip(x)

        # Scale branch output
        scale_out = 0
        if self.rbr_scale is not None:
            scale_out = self.rbr_scale(x)

        # Other branches
        out = scale_out + identity_out
        for ix in range(self.num_conv_branches):
            out += self.rbr_conv[ix](x)

        return self.activation(self.se(out))

    def reparameterize(self):
        """ Following works like `RepVGG: Making VGG-style ConvNets Great Again` -
        https://arxiv.org/pdf/2101.03697.pdf. We re-parameterize multi-branched
        architecture used at training time to obtain a plain CNN-like structure
        for inference.
        """
        if self.inference_mode:
            return
        kernel, bias = self._get_kernel_bias()
        self.reparam_conv = nn.Conv2d(in_channels=self.rbr_conv[0].conv.in_channels,
                                      out_channels=self.rbr_conv[0].conv.out_channels,
                                      kernel_size=self.rbr_conv[0].conv.kernel_size,
                                      stride=self.rbr_conv[0].conv.stride,
                                      padding=self.rbr_conv[0].conv.padding,
                                      dilation=self.rbr_conv[0].conv.dilation,
                                      groups=self.rbr_conv[0].conv.groups,
                                      bias=True)
        self.reparam_conv.weight.data = kernel
        self.reparam_conv.bias.data = bias

        # Delete un-used branches
        for para in self.parameters():
            para.detach_()
        self.__delattr__('rbr_conv')
        self.__delattr__('rbr_scale')
        if hasattr(self, 'rbr_skip'):
            self.__delattr__('rbr_skip')

        self.inference_mode = True

    def _get_kernel_bias(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """ Method to obtain re-parameterized kernel and bias.
        Reference: https://github.com/DingXiaoH/RepVGG/blob/main/repvgg.py#L83

        :return: Tuple of (kernel, bias) after fusing branches.
        """
        # get weights and bias of scale branch
        kernel_scale = 0
        bias_scale = 0
        if self.rbr_scale is not None:
            kernel_scale, bias_scale = self._fuse_bn_tensor(self.rbr_scale)
            # Pad scale branch kernel to match conv branch kernel size.
            pad = self.kernel_size // 2
            kernel_scale = torch.nn.functional.pad(kernel_scale,
                                                   [pad, pad, pad, pad])

        # get weights and bias of skip branch
        kernel_identity = 0
        bias_identity = 0
        if self.rbr_skip is not None:
            kernel_identity, bias_identity = self._fuse_bn_tensor(self.rbr_skip)

        # get weights and bias of conv branches
        kernel_conv = 0
        bias_conv = 0
        for ix in range(self.num_conv_branches):
            _kernel, _bias = self._fuse_bn_tensor(self.rbr_conv[ix])
            kernel_conv += _kernel
            bias_conv += _bias

        kernel_final = kernel_conv + kernel_scale + kernel_identity
        bias_final = bias_conv + bias_scale + bias_identity
        return kernel_final, bias_final

    def _fuse_bn_tensor(self, branch) -> Tuple[torch.Tensor, torch.Tensor]:
        """ Method to fuse batchnorm layer with preceeding conv layer.
        Reference: https://github.com/DingXiaoH/RepVGG/blob/main/repvgg.py#L95

        :param branch:
        :return: Tuple of (kernel, bias) after fusing batchnorm.
        """
        if isinstance(branch, nn.Sequential):
            kernel = branch.conv.weight
            running_mean = branch.bn.running_mean
            running_var = branch.bn.running_var
            gamma = branch.bn.weight
            beta = branch.bn.bias
            eps = branch.bn.eps
        else:
            assert isinstance(branch, nn.BatchNorm2d)
            if not hasattr(self, 'id_tensor'):
                input_dim = self.in_channels // self.groups
                kernel_value = torch.zeros((self.in_channels,
                                            input_dim,
                                            self.kernel_size,
                                            self.kernel_size),
                                           dtype=branch.weight.dtype,
                                           device=branch.weight.device)
                for i in range(self.in_channels):
                    kernel_value[i, i % input_dim,
                                    self.kernel_size // 2,
                                    self.kernel_size // 2] = 1
                self.id_tensor = kernel_value
            kernel = self.id_tensor
            running_mean = branch.running_mean
            running_var = branch.running_var
            gamma = branch.weight
            beta = branch.bias
            eps = branch.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std

    def _conv_bn(self,
                 kernel_size: int,
                 padding: int) -> nn.Sequential:
        """ Helper method to construct conv-batchnorm layers.

        :param kernel_size: Size of the convolution kernel.
        :param padding: Zero-padding size.
        :return: Conv-BN module.
        """
        mod_list = nn.Sequential()
        mod_list.add_module('conv', nn.Conv2d(in_channels=self.in_channels,
                                              out_channels=self.out_channels,
                                              kernel_size=kernel_size,
                                              stride=self.stride,
                                              padding=padding,
                                              groups=self.groups,
                                              bias=False))
        mod_list.add_module('bn', nn.BatchNorm2d(num_features=self.out_channels))
        return mod_list


class MobileRepV3(nn.Module):
    """ MobileOne Model

        Pytorch implementation of `An Improved One millisecond Mobile Backbone` -
        https://arxiv.org/pdf/2206.04040.pdf
    """

    def __init__(self,
                 cfg: list,
                 inference_mode: bool = False,
                 num_conv_branches: int = 1) -> None:
        """ Construct MobileOne model.

        :param num_blocks_per_stage: List of number of blocks per stage.
        :param num_classes: Number of classes in the dataset.
        :param width_multipliers: List of width multiplier for blocks in a stage.
        :param inference_mode: If True, instantiates model in inference mode.
        :param use_se: Whether to use SE-ReLU activations.
        :param num_conv_branches: Number of linear conv branches.
        """
        super().__init__()

        self.inference_mode = inference_mode
        self.num_conv_branches = num_conv_branches
        self.cfgs = cfg
        # Build stages
        self.stage0 = MobileOneBlock(in_channels=3, out_channels=16,
                                     kernel_size=3, stride=2, padding=1,
                                     inference_mode=self.inference_mode)

        layers = []
        # in c,   out c,  kernel,     stride,     SE,     HS
        for in_channel, out_channel, kernel, stride, SE, HS in self.cfgs:
            layers.extend(self._make_stage(in_channel=in_channel, out_channel=out_channel, kernel=kernel, stride=stride,
                                           use_se=SE, use_act=HS))
        self.features = nn.Sequential(*layers)

        self.stage1 = MobileOneBlock(in_channels=160, out_channels=960,
                                     kernel_size=1, stride=1, padding=0,
                                     inference_mode=self.inference_mode)

        _initialize_weights(self)

    def _make_stage(self,
                    in_channel: int,
                    out_channel: int,
                    kernel: int,
                    stride: int,
                    use_se: bool,
                    use_act: str,
                    ) -> list:
        """ Build a stage of MobileOne model.

        :param planes: Number of output channels.
        :param num_blocks: Number of blocks in this stage.
        :param num_se_blocks: Number of SE blocks in this stage.
        :return: A stage of MobileOne model.
        """
        # Get strides for all layers
        blocks = []
        if kernel == 5:
            self.padding = 2
        else:
            self.padding = 1

        # Pointwise conv
        blocks.append(MobileOneBlock(in_channels=in_channel,
                                     out_channels=out_channel,
                                     kernel_size=1,
                                     stride=1,
                                     padding=0,
                                     groups=1,
                                     inference_mode=self.inference_mode,
                                     use_se=use_se,
                                     num_conv_branches=self.num_conv_branches,
                                     use_act=use_act))
        # Depth wise conv
        blocks.append(MobileOneBlock(in_channels=out_channel,
                                     out_channels=out_channel,
                                     kernel_size=kernel,
                                     stride=stride,
                                     padding=self.padding,
                                     groups=out_channel,
                                     inference_mode=self.inference_mode,
                                     use_se=use_se,
                                     num_conv_branches=self.num_conv_branches,
                                     use_act=use_act))

        # return nn.Sequential(*blocks)
        return blocks

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """ Apply forward pass. """
        x = self.stage0(x)
        x = self.features(x)
        x = self.stage1(x)
        return x


def RepMobileNetv3(**kwargs):
    """
    Constructs a MobileNetV3-Small model
    """
    cfgs = [  #
        #   in c,   out c,  kernel,     stride,     SE,     HS
        # 16 -> 16 -> 16, ReLU, stride: 1, kernel:3, SE: False
        [16, 16, 3, 1, False, 'ReLU'],
        [16, 16, 3, 1, False, 'ReLU'],

        # 16 -> 64 -> 24, ReLU, stride:2, kernel: 3, SE: False
        [16, 64, 3, 2, False, 'ReLU'],
        [64, 24, 3, 1, False, 'ReLU'],

        # 24 -> 72 -> 24, ReLU, stride:1, kernel:3, SE: False
        [24, 72, 3, 1, False, 'ReLU'],
        [72, 24, 3, 1, False, 'ReLU'],

        # 24 -> 72 -> 40, ReLU, stride:2, kernel:5, SE: True
        [24, 72, 5, 2, False, 'ReLU'],
        [72, 40, 3, 1, False, 'ReLU'],

        # 40 -> 120 -> 40, ReLU, stride:1, kernel:5, SE: True
        [40, 120, 5, 1, False, 'ReLU'],
        [120, 40, 3, 1, False, 'ReLU'],

        # 40 -> 120 -> 40, ReLU, stride:1, kernel:5, SE: True
        [40, 120, 5, 1, False, 'ReLU'],
        [120, 40, 3, 1, False, 'ReLU'],

        # 40 -> 240 -> 80, HS, stride:2, kernel:3, SE: False
        [40, 240, 3, 2, False, 'HS'],
        [240, 80, 3, 1, False, 'ReLU'],

        # 80 -> 200 -> 80, HS, stride:1, kernel:3, SE: False
        [80, 200, 3, 1, False, 'HS'],
        [200, 80, 3, 1, False, 'ReLU'],

        # 80 -> 184 -> 80, HS, stride:1, kernel:3, SE: False
        [80, 184, 3, 1, False, 'HS'],
        [184, 80, 3, 1, False, 'HS'],

        # 80 -> 184 -> 80, HS, stride:1, kernel:3, SE: False
        [80, 184, 3, 1, False, 'HS'],
        [184, 80, 3, 1, False, 'HS'],

        # 80 -> 480 -> 112, HS, stride:1, kernel:3, SE: True
        [80, 480, 3, 1, False, 'HS'],
        [480, 112, 3, 1, False, 'HS'],

        # 112 -> 672 -> 112, HS, stride:1, kernel:3, SE: True
        [112, 672, 3, 1, False, 'HS'],
        [672, 112, 3, 1, False, 'HS'],

        # 112 -> 672 -> 160, HS, stride:2, kernel:5, SE: True
        [112, 672, 5, 1, False, 'HS'],
        [672, 160, 3, 1, False, 'HS'],

        # 160 -> 960 -> 160, HS, stride:1, kernel:5, SE: True
        [160, 960, 5, 1, False, 'HS'],
        [960, 160, 3, 1, False, 'HS'],

        # 160 -> 960 -> 160, HS, stride:1, kernel:5, SE: True
        [160, 960, 5, 1, False, 'HS'],
        [960, 160, 3, 1, False, 'HS'],
    ]

    return MobileRepV3(cfg=cfgs, inference_mode=True, num_conv_branches=2)


if __name__ == "__main__":
    from torchsummary import summary

    net = RepMobileNetv3().cuda()
    summary(net, (3, 255, 255))
    # net = RepMobileNetv3()
    # data = torch.rand(1, 3, 255, 255)
    # out = net(data)
    # print(out)
