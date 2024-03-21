from torch2trt import torch2trt
import argparse
import os
import torch
import sys
from nanotrack.models.backbone.RLightTrack1 import reparameterize_model
sys.path.append(os.getcwd())
from nanotrack.core.config import cfg
from nanotrack.models.model_builder import ModelBuilder
from nanotrack.utils.model_load import load_pretrain


parser = argparse.ArgumentParser(description='nanotrack')

parser.add_argument('--dataset', default='OTB100', type=str, help='datasets')

parser.add_argument('--tracker_name', '-t', default='MobileOne', type=str, help='tracker name')

parser.add_argument('--config', default='./models/config/Rep_config.yaml', type=str, help='config file')

parser.add_argument('--snapshot', default='models/snapshot/test.pth', type=str, help='snapshot of models to eval')

parser.add_argument('--save_path', default='./results', type=str, help='snapshot of models to eval')

parser.add_argument('--video', default='', type=str, help='eval one special video')

parser.add_argument('--vis', action='store_true', help='whether v isualzie Ray_result')

parser.add_argument('--gpu_id', default='not_set', type=str, help="gpu id")

parser.add_argument('--tracker_path', '-p', default='./results', type=str, help='tracker Ray_result path')

parser.add_argument('--num', '-n', default=4, type=int, help='number of thread to eval')

parser.add_argument('--show_video_level', '-s', dest='show_video_level', action='store_true')

parser.set_defaults(show_video_level=False)

args = parser.parse_args()

if args.gpu_id != 'not_set':
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

torch.set_num_threads(1)


def run():
    cfg.merge_from_file(args.config)

    # create model
    model = ModelBuilder(cfg)

    # load model
    model = load_pretrain(model, args.snapshot).eval().cuda()
    model.backbone = reparameterize_model(model.backbone)

    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

    # tensorrt 模型保存路径
    device_name = torch.cuda.get_device_name(device)
    device_name = device_name.replace(' ', '_').lower()
    out_path = f'output/checkpoints/mixed_second_finetune_acc_97P7.{device_name}.pth'

    # tensorrt 输入样本
    height, width = 32, 492  # 简单起见，暂时不考虑动态输入，这里设置成测试样本大小
    x = torch.ones((1, 1, height, width)).to(device)

    # 设置 TensorRT 模型内存上限
    max_workspace_size = 1 << 32  # 2^30 bytes = 1GB

    # 模型转换与保存
    model_trt = torch2trt(model, [x], max_workspace_size=max_workspace_size)
    torch.save(model_trt.state_dict(), out_path)


if __name__ == '__main__':
    run()
