from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import os

import cv2
import torch
import numpy as np

import sys

from nanotrack.models.backbone.RLightTrack1 import reparameterize_model

sys.path.append(os.getcwd())

from tqdm import tqdm

from nanotrack.core.config import cfg
from nanotrack.models.model_builder import ModelBuilder
from nanotrack.tracker.tracker_builder import build_tracker
from nanotrack.utils.bbox import get_axis_aligned_bbox
from nanotrack.utils.model_load import load_pretrain
from toolkit.datasets import DatasetFactory
from toolkit.utils.region import vot_overlap, vot_float2str
import torch.onnx
from ptflops import get_model_complexity_info
import torch.nn as nn
from bin.eval import eval
from torchsummary import summary


parser = argparse.ArgumentParser(description='nanotrack')

parser.add_argument('--config', default='./models/config/SubNet.yaml', type=str, help='config file')  # Rep_config.yaml

parser.add_argument('--snapshot', default='models/snapshot/test.pth', type=str, help='snapshot of models to eval')

args = parser.parse_args()

torch.set_num_threads(1)


class AlexNet(nn.Module):
    def __init__(self, num_classes=1000):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


def main():
    cfg.merge_from_file(args.config)

    # create model
    model = ModelBuilder(cfg)

    # load model
    model = load_pretrain(model, args.snapshot).eval()
    model.backbone = reparameterize_model(model.backbone)
    backbone = model.backbone
    flops, params = get_model_complexity_info(backbone, (3, 255, 255), as_strings=True, print_per_layer_stat=False)
    print(f"模型的计算量（FLOPs）: {flops}")
    summary(backbone, (3, 255, 255), device='cpu')

    # for export backbone model
    # dynamic_axes_23 = {
    #     'in': [2, 3],
    #     'out': [2, 3]
    # }

    # picture = torch.rand(1, 3, 255, 255)
    # torch.onnx.export(model.backbone,  # model being run
    #                   picture,  # model input (or a tuple for multiple inputs)
    #                   "backbone.onnx",  # where to save the model (can be a file or file-like object)
    #                   export_params=True,  # store the trained parameter weights inside the model file
    #                   # opset_version=10,  # the ONNX version to export the model to
    #                   do_constant_folding=True,  # whether to execute constant folding for optimization
    #                   input_names=['input'],  # the model's input names
    #                   output_names=['output'],  # the model's output names
    #                   dynamic_axes=dynamic_axes_23)

    # picture = torch.rand(1, 3, 255, 255)
    # torch.onnx.export(model.backbone,  # model being run
    #                   picture,  # model input (or a tuple for multiple inputs)
    #                   "backbone_255.onnx",  # where to save the model (can be a file or file-like object)
    #                   export_params=True,  # store the trained parameter weights inside the model file
    #                   # opset_version=10,  # the ONNX version to export the model to
    #                   do_constant_folding=True,  # whether to execute constant folding for optimization
    #                   input_names=['input'],  # the model's input names
    #                   output_names=['output'],  # the model's output names
    #                   )

    # for export head model
    # template = torch.rand(1, 1024, 8, 8)
    # search = torch.rand(1, 1024, 16, 16)
    # dummy_input = (template, search)
    # input_names = ['input1', 'input2']
    # output_names = ['output1', 'output2']
    # torch.onnx.export(model.ban_head,  # model being run
    #                   dummy_input,  # model input (or a tuple for multiple inputs)
    #                   "head.onnx",  # where to save the model (can be a file or file-like object)
    #                   export_params=True,  # store the trained parameter weights inside the model file
    #                   # opset_version=10,  # the ONNX version to export the model to
    #                   do_constant_folding=True,  # whether to execute constant folding for optimization
    #                   input_names=input_names,  # the model's input names
    #                   output_names=output_names,  # the model's output names
    #                   # dynamic_axes=dynamic_axes_23  # variable lenght axes
    #                   )
    # print(model)


if __name__ == '__main__':
    main()
