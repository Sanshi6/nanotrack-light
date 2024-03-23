from __future__ import absolute_import

# 在 Windows 系统， ray 对于多线程 Pool 库的支持很差，程序会卡住不动，可以用 for 代替这个操作。

import time
# ray library
import ray
from cv2 import cv2
from ray import tune
from ray.air import RunConfig
from ray.tune import Experiment, TuneConfig
from ray.tune.search.bayesopt import BayesOptSearch
from ray.tune.search.optuna import OptunaSearch
from ray.tune.search.ax import AxSearch
from hyperopt import hp
from pprint import pprint

from nanotrack.sub_models.backbone.subnet import reparameterize_model
# model library
from toolkit.datasets import DatasetFactory
from toolkit.utils.region import vot_overlap, vot_float2str
from nanotrack.models.model_builder import ModelBuilder
from nanotrack.tracker.tracker_builder import build_tracker
from nanotrack.utils.bbox import get_axis_aligned_bbox
from nanotrack.utils.model_load import load_pretrain
from nanotrack.core.config import cfg

# os library
import os
import argparse
import numpy as np
import sys
from easydict import EasyDict as edict
from bin.eval import eval
from tqdm import tqdm

sys.path.append("..")

# parser
parser = argparse.ArgumentParser(description='parameters for Ocean tracker')
parser.add_argument('--snapshot', default=r'E:\SiamProject\NanoTrack\models\snapshot\test.pth', type=str, help='snapshot of model')
parser.add_argument('--config', default=r'E:\SiamProject\NanoTrack\models\config\Rep_config.yaml', type=str)
parser.add_argument('--cache_dir', default=r'E:\SiamProject\NanoTrack\TPE_results', type=str, help='directory to store cache')
parser.add_argument('--gpu_nums', default=1, type=int, help='gpu numbers')  # gpu
parser.add_argument('--trial_per_gpu', default=8, type=int, help='trail per gpu')  # trial num
parser.add_argument('--dataset', default='VOT2018', type=str, help='dataset')  # datasets
args = parser.parse_args()
print('==> However TPE is slower than GENE')

# receive parameter
info = edict()
info.config = args.config
info.cache_dir = args.cache_dir
info.gpu_nums = args.gpu_nums
info.trial_per_gpu = args.trial_per_gpu

# # cfg build
# cfg.merge_from_file(args.config)
#
# # model build  -> pretrained model
# model = ModelBuilder(cfg)
# print("enter.")
# model = load_pretrain(model, args.snapshot).cuda().eval()
# model.backbone = reparameterize_model(model.backbone)
#
# # tracker builder
# tracker = build_tracker(model, cfg)
#
# # dataset build
# info.dataset = args.dataset
# info.datasetPath = os.path.join('.', 'datasets', info.dataset)
# dataset = DatasetFactory.create_dataset(name=info.dataset, dataset_root=info.datasetPath, load_img=False)
# print('pretrained model has been loaded')
# print(os.environ['CUDA_VISIBLE_DEVICES'])

# test
def test(tracker, cfgs, dataset):
    # save path // eval(cf), cf parameter setting

    # args: dataset
    #       args.save_path, args.dataset, args.tracker_name,
    #                                   'baseline', video.name
    #     parser.add_argument('--tracker_path', '-p', default='./results', type=str,
    #                         help='tracker Ray_result path')
    #     parser.add_argument('--dataset', '-d', default=dataset, type=str,
    #                         help='dataset name')
    #     parser.add_argument('--num', '-n', default=4, type=int,
    #                         help='number of thread to eval')
    #     parser.add_argument('--tracker_name', '-t', default=tracker_name,
    #                         type=str, help='tracker name')
    #     parser.add_argument('--show_video_level', '-s', dest='show_video_level',
    #                         action='store_true')
    #     parser.set_defaults(show_video_level=False)
    #
    #
    total_lost = 0
    avg_speed = 0
    for v_idx, video in tqdm(enumerate(dataset)):
        # if args.video != '':
        #     if video.name != args.video:
        #         continue
        frame_counter = 0
        lost_number = 0
        toc = 0
        pred_bboxes = []

        for idx, (img, gt_bbox) in enumerate(video):
            # print(img.shape)
            if len(gt_bbox) == 4:
                gt_bbox = [gt_bbox[0], gt_bbox[1],
                           gt_bbox[0], gt_bbox[1] + gt_bbox[3] - 1,
                           gt_bbox[0] + gt_bbox[2] - 1, gt_bbox[1] + gt_bbox[3] - 1,
                           gt_bbox[0] + gt_bbox[2] - 1, gt_bbox[1]]
            tic = cv2.getTickCount()
            if idx == frame_counter:
                cx, cy, w, h = get_axis_aligned_bbox(np.array(gt_bbox))
                gt_bbox_ = [cx - (w - 1) / 2, cy - (h - 1) / 2, w, h]  # [topx,topy,w,h]
                tracker.init(img, gt_bbox_)
                pred_bbox = gt_bbox_
                pred_bboxes.append(1)
            elif idx > frame_counter:
                outputs = tracker.track(img)
                pred_bbox = outputs['bbox']
                if cfg.MASK.MASK:
                    pred_bbox = outputs['polygon']
                overlap = vot_overlap(pred_bbox, gt_bbox, (img.shape[1], img.shape[0]))
                if overlap > 0:
                    pred_bboxes.append(pred_bbox)
                else:
                    pred_bboxes.append(2)
                    frame_counter = idx + 5
                    lost_number += 1
            else:
                pred_bboxes.append(0)
            toc += cv2.getTickCount() - tic
            if idx == 0:
                cv2.destroyAllWindows()
            if cfgs.vis and idx > frame_counter:                # visual tracker box
                cv2.polylines(img, [np.array(gt_bbox, np.int).reshape((-1, 1, 2))],
                              True, (0, 255, 0), 3)
                if cfg.MASK.MASK:
                    cv2.polylines(img, [np.array(pred_bbox, np.int).reshape((-1, 1, 2))],
                                  True, (0, 255, 255), 3)
                else:
                    bbox = list(map(int, pred_bbox))
                    cv2.rectangle(img, (bbox[0], bbox[1]),
                                  (bbox[0] + bbox[2], bbox[1] + bbox[3]), (0, 255, 255), 3)
                cv2.putText(img, str(idx), (40, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                cv2.putText(img, str(lost_number), (40, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                cv2.imshow(video.name, img)
                cv2.waitKey(1)
        toc /= cv2.getTickFrequency()
        # save results
        video_path = os.path.join(cfgs.save_path, cfgs.dataset, cfgs.tracker_name,
                                  'baseline', video.name)
        if not os.path.isdir(video_path):
            os.makedirs(video_path)

        result_path = os.path.join(video_path, '{}_001.txt'.format(video.name))
        with open(result_path, 'w') as f:
            for x in pred_bboxes:
                if isinstance(x, int):
                    f.write("{:d}\n".format(x))
                else:
                    f.write(','.join([vot_float2str("%.4f", i) for i in x]) + '\n')

        total_lost += lost_number
        avg_speed += idx / toc

    print('Speed: {:3.1f}fps'.format(avg_speed / 60))
    eao = eval(cfgs)
    return eao


class Cfgs:
    def __init__(self):
        self.save_path = None
        self.dataset = None
        self.tracker_name = None
        self.vis = False
        self.tracker_path = None
        self.show_video_level = False
        self.num = 1

# fitness function
def fitness(config):
    # for test
    # cfgs = Cfgs()
    # cfgs.dataset = "VOT2018"
    # cfgs.tracker_path = r"E:/SiamProject/NanoTrack/results"
    # cfgs.tracker_name = "p_0.385_s_0.385_w_0.409"
    # cfgs.save_path = r"E:/SiamProject/NanoTrack/results"
    # eao = eval(cfgs)

    # cfg build
    cfg.merge_from_file(args.config)
    # update config for tune
    cfg.TRACK.PENALTY_K = config["penalty_k"]
    cfg.TRACK.LR = config['scale_lr']
    cfg.TRACK.WINDOW_INFLUENCE = config['window_influence']
    # print("config=", config)
    # print(cfg.TRACK.PENALTY_K, cfg.TRACK.LR, cfg.TRACK.WINDOW_INFLUENCE)


    # model build  -> pretrained model
    model = ModelBuilder(cfg)
    # print("enter.")
    model = load_pretrain(model, args.snapshot).cuda().eval()
    model.backbone = reparameterize_model(model.backbone)

    # tracker builder
    tracker = build_tracker(model, cfg)

    # dataset build
    info.dataset = args.dataset
    info.datasetPath = os.path.join('E:/SiamProject/NanoTrack', 'datasets', info.dataset)
    dataset = DatasetFactory.create_dataset(name=info.dataset, dataset_root=info.datasetPath, load_img=False)
    print('pretrained model has been loaded')
    # print(os.environ['CUDA_VISIBLE_DEVICES'])

    # update tracker
    # tracker = build_tracker(model, cfg)
    cfgs = Cfgs()
    cfgs.dataset = "VOT2018"
    cfgs.tracker_path = r"E:/SiamProject/NanoTrack/results"
    cfgs.tracker_name = "p_{:.3f}_s_{:.3f}_w_{:.3f}".format(config['scale_lr'], config['scale_lr'], config['window_influence'])
    cfgs.save_path = r"E:/SiamProject/NanoTrack/results"

    # TODO: config -> tracker -> return eao value, refer test.py

    eao = test(tracker, cfgs, dataset)
    print(
        "penalty_k: {0}, scale_lr: {1}, window_influence: {2}, eao: {3}".format(
            cfg.TRACK.PENALTY_K, cfg.TRACK.LR, cfg.TRACK.WINDOW_INFLUENCE, eao))
    return {"eao": eao}


if __name__ == "__main__":
    ray.init(num_gpus=args.gpu_nums, num_cpus=args.gpu_nums * 8, object_store_memory=8000000000)

    params = {
        "penalty_k": tune.quniform(0.001, 0.2, 0.001),
        "scale_lr": tune.quniform(0.001, 0.8, 0.001),
        "window_influence": tune.quniform(0.001, 0.65, 0.001),
    }

    tuner = tune.Tuner(fitness, param_space=params,
                       run_config=RunConfig(name="my_tune_run", storage_path=r"E:\SiamProject\NanoTrack\Ray_result", ),
                       tune_config=TuneConfig(num_samples=800, mode='max', metric='eao',
                                              max_concurrent_trials=8, search_alg=OptunaSearch()))

    results = tuner.fit()
    print(results.get_best_result(metric="eao", mode="max").config)
