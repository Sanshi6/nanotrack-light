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

# model library
from toolkit.datasets import DatasetFactory
from toolkit.utils.region import vot_overlap, vot_float2str
from nanotrack.models.model_builder import ModelBuilder
from nanotrack.tracker.tracker_builder import build_tracker
from nanotrack.utils.bbox import get_axis_aligned_bbox
from nanotrack.utils.model_load import load_pretrain
from nanotrack.core.config import cfg
from nanotrack.models.backbone.MobileOne import reparameterize_model

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
parser.add_argument('--snapshot', default=r'E:\SiamProject\NanoTrack\models\snapshot\test.pth', type=str,
                    help='snapshot of model')
parser.add_argument('--config', default=r'E:\SiamProject\NanoTrack\models\config\Rep_config.yaml', type=str)
parser.add_argument('--cache_dir', default=r'E:\SiamProject\NanoTrack\TPE_results', type=str,
                    help='directory to store cache')
parser.add_argument('--gpu_nums', default=1, type=int, help='gpu numbers')  # gpu
parser.add_argument('--trial_per_gpu', default=8, type=int, help='trail per gpu')  # trial num
parser.add_argument('--dataset', default='OTB100', type=str, help='dataset')  # datasets
parser.add_argument('--video', default='', type=str, help='eval one special video')
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
    for v_idx, video in tqdm(enumerate(dataset)):
        # if args.video != '':
        #     # test one special video
        #     if video.name != args.video:
        #         continue
        toc = 0
        pred_bboxes = []
        scores = []
        track_times = []
        for idx, (img, gt_bbox) in enumerate(video):
            tic = cv2.getTickCount()
            if idx == 0:
                cx, cy, w, h = get_axis_aligned_bbox(np.array(gt_bbox))
                gt_bbox_ = [cx - (w - 1) / 2, cy - (h - 1) / 2, w, h]  # [topx,topy,w,h]
                tracker.init(img, gt_bbox_)
                pred_bbox = gt_bbox_
                scores.append(None)
                if 'VOT2018-LT' == cfgs.dataset:
                    pred_bboxes.append([1])
                else:
                    pred_bboxes.append(pred_bbox)
            else:
                outputs = tracker.track(img)
                pred_bbox = outputs['bbox']
                pred_bboxes.append(pred_bbox)
                # scores.append(outputs['best_score'])
            toc += cv2.getTickCount() - tic
            track_times.append((cv2.getTickCount() - tic) / cv2.getTickFrequency())
            if idx == 0:
                cv2.destroyAllWindows()
            # if args.vis and idx > 0:
            #     gt_bbox = list(map(int, gt_bbox))
            #     pred_bbox = list(map(int, pred_bbox))
            #     cv2.rectangle(img, (gt_bbox[0], gt_bbox[1]),
            #                   (gt_bbox[0] + gt_bbox[2], gt_bbox[1] + gt_bbox[3]), (0, 255, 0), 3)
            #     cv2.rectangle(img, (pred_bbox[0], pred_bbox[1]),
            #                   (pred_bbox[0] + pred_bbox[2], pred_bbox[1] + pred_bbox[3]), (0, 255, 255), 3)
            #     cv2.putText(img, str(idx), (40, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            #     cv2.imshow(video.name, img)
                cv2.waitKey(1)
        toc /= cv2.getTickFrequency()

        # save results
        model_path = os.path.join(cfgs.save_path, cfgs.dataset, cfgs.tracker_name)
        if not os.path.isdir(model_path):
            os.makedirs(model_path)
        result_path = os.path.join(model_path, '{}.txt'.format(video.name))
        with open(result_path, 'w') as f:
            for x in pred_bboxes:
                f.write(','.join([str(i) for i in x]) + '\n')
    eao = eval(cfgs)
    return eao

#  because save path or other parameter have been change in per experiment, so use Cfgs alternative the cfgs.
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
    cfgs.dataset = "OTB100"
    cfgs.tracker_path = r"E:/SiamProject/NanoTrack/results"
    cfgs.tracker_name = "p_{:.3f}_s_{:.3f}_w_{:.3f}".format(config['scale_lr'], config['scale_lr'],
                                                            config['window_influence'])
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
# Current best trial: 21e45116 with eao=0.6745109496429366 and params={'penalty_k': 0.010000000000000002, 'scale_lr': 0.2, 'window_influence': 0.265}

# test


# Current best trial: 21e45116 with eao=0.6745109496429366 and params={'penalty_k': 0.010000000000000002, 'scale_lr': 0.2, 'window_influence': 0.265}
# ╭──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
# │ Trial name         status         penalty_k     scale_lr     window_influence     iter     total time (s)        eao │
# ├──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
# │ fitness_38c98f2d   TERMINATED         0.023        0.243                0.279        1            3495.49   0.663604 │
# │ fitness_d2dc8ca5   TERMINATED         0.197        0.675                0.138        1            4670.04   0.645868 │
# │ fitness_83770d3e   TERMINATED         0.029        0.253                0.336        1            4657.17   0.662778 │
# │ fitness_d4dbf6c0   TERMINATED         0.107        0.711                0.222        1            4614.36   0.651963 │
# │ fitness_d1a8d67a   TERMINATED         0.026        0.395                0.219        1            4725.4    0.66961  │
# │ fitness_a04407be   TERMINATED         0.078        0.143                0.441        1            2265.32   0.655042 │
# │ fitness_e7b30e6d   TERMINATED         0.007        0.475                0.012        1            2181.11   0.656671 │
# │ fitness_73a03c0d   TERMINATED         0.069        0.413                0.613        1            2196.09   0.643832 │
# │ fitness_3775f909   TERMINATED         0.025        0.472                0.018        1            2047.62   0.650505 │
# │ fitness_a25db061   TERMINATED         0.162        0.033                0.28         1            2045.38   0.629688 │
# │ fitness_ae68b50a   TERMINATED         0.081        0.785                0.452        1            2109.25   0.649679 │
# │ fitness_d7545059   TERMINATED         0.084        0.703                0.324        1            3494.12   0.649217 │
# │ fitness_bbee2eb8   TERMINATED         0.022        0.046                0.257        1            1919.99   0.642059 │
# │ fitness_5a622ba7   TERMINATED         0.049        0.754                0.061        1            1929.71   0.66292  │
# │ fitness_935ac864   TERMINATED         0.193        0.579                0.291        1            2005.45   0.651809 │
# │ fitness_9e4554c3   TERMINATED         0.142        0.081                0.151        1            4534.89   0.652441 │
# │ fitness_5420f870   TERMINATED         0.051        0.497                0.282        1            4588.2    0.657517 │
# │ fitness_cd596eb4   TERMINATED         0.127        0.26                 0.636        1            4654.93   0.637475 │
# │ fitness_0cff5f39   TERMINATED         0.044        0.288                0.167        1            3923.2    0.666682 │
# │ fitness_fc15e351   TERMINATED         0.054        0.286                0.153        1            1966.64   0.667059 │
# │ fitness_35fb411f   TERMINATED         0.117        0.274                0.38         1            1973.99   0.66028  │
# │ fitness_61052c2f   TERMINATED         0.12         0.275                0.176        1            2064      0.65882  │
# │ fitness_cf887b2a   TERMINATED         0.122        0.266                0.182        1            3491.71   0.656894 │
# │ fitness_3b8476d7   TERMINATED         0.114        0.276                0.143        1            2160.5    0.652963 │
# │ fitness_a828c432   TERMINATED         0.052        0.345                0.145        1            2200.92   0.667255 │
# │ fitness_c8956be0   TERMINATED         0.058        0.358                0.112        1            2799.26   0.661459 │
# │ fitness_66cf505d   TERMINATED         0.002        0.375                0.097        1            2269.79   0.652794 │
# │ fitness_42e33517   TERMINATED         0.052        0.359                0.088        1            2184.98   0.646327 │
# │ fitness_bbc8dfb5   TERMINATED         0.055        0.35                 0.09         1            2167.25   0.646132 │
# │ fitness_86eac7b2   TERMINATED         0.051        0.338                0.098        1            2183.32   0.661284 │
# │ fitness_325f71a4   TERMINATED         0.049        0.347                0.083        1            2175.29   0.646773 │
# │ fitness_d5186f69   TERMINATED         0.005        0.382                0.123        1            2145.7    0.656354 │
# │ fitness_90827976   TERMINATED         0.003        0.363                0.09         1            2123.5    0.654076 │
# │ fitness_aca55187   TERMINATED         0.062        0.374                0.083        1            2158.52   0.652151 │
# │ fitness_2fe9d4d2   TERMINATED         0.093        0.198                0.219        1            3364.84   0.653082 │
# │ fitness_bdccd203   TERMINATED         0.037        0.164                0.208        1            2040.37   0.662877 │
# │ fitness_050812bb   TERMINATED         0.097        0.167                0.216        1            2046.72   0.660649 │
# │ fitness_0a017181   TERMINATED         0.095        0.15                 0.227        1            2023.31   0.660896 │
# │ fitness_854972d7   TERMINATED         0.038        0.186                0.194        1            1808.61   0.668351 │
# │ fitness_c666b606   TERMINATED         0.094        0.121                0.198        1            1814.03   0.661048 │
# │ fitness_6e25f429   TERMINATED         0.035        0.173                0.217        1            1815.99   0.655906 │
# │ fitness_d95885df   TERMINATED         0.029        0.178                0.212        1            1892.88   0.663071 │
# │ fitness_42af871b   TERMINATED         0.096        0.171                0.221        1            1793.67   0.660656 │
# │ fitness_3f49fcdd   TERMINATED         0.037        0.204                0.243        1            1797.28   0.656729 │
# │ fitness_3b378571   TERMINATED         0.039        0.212                0.171        1            1790.75   0.6609   │
# │ fitness_92e5dee8   TERMINATED         0.019        0.207                0.046        1            1806.65   0.650999 │
# │ fitness_4e1907d9   TERMINATED         0.021        0.204                0.242        1            1813.54   0.661152 │
# │ fitness_678f2427   TERMINATED         0.018        0.427                0.039        1            1808.7    0.652804 │
# │ fitness_5bd5afe6   TERMINATED         0.021        0.432                0.153        1            3132.44   0.656864 │
# │ fitness_67f79fd6   TERMINATED         0.017        0.434                0.046        1            1900.19   0.661374 │
# │ fitness_1ed556a8   TERMINATED         0.019        0.424                0.149        1            1793.31   0.649451 │
# │ fitness_a614fb2c   TERMINATED         0.072        0.433                0.04         1            1810.83   0.642833 │
# │ fitness_7a470080   TERMINATED         0.016        0.45                 0.151        1            1794.15   0.655106 │
# │ fitness_51b04f00   TERMINATED         0.016        0.434                0.37         1            1802.9    0.659984 │
# │ fitness_1cfed20d   TERMINATED         0.071        0.426                0.312        1            1813.67   0.657591 │
# │ fitness_6ca8eb36   TERMINATED         0.065        0.529                0.324        1            1808.8    0.656511 │
# │ fitness_d1bb9bf9   TERMINATED         0.071        0.311                0.151        1            1897.11   0.664909 │
# │ fitness_3cbd5f6d   TERMINATED         0.072        0.616                0.349        1            1796.38   0.647567 │
# │ fitness_c396b7f7   TERMINATED         0.066        0.311                0.356        1            1801.09   0.666466 │
# │ fitness_0f5f4561   TERMINATED         0.071        0.317                0.33         1            1793.38   0.665707 │
# │ fitness_c44a9a03   TERMINATED         0.043        0.318                0.327        1            3104.07   0.667827 │
# │ fitness_9128e325   TERMINATED         0.044        0.306                0.336        1            1805.61   0.662445 │
# │ fitness_d7dd268d   TERMINATED         0.066        0.311                0.273        1            1812.63   0.664582 │
# │ fitness_200806a3   TERMINATED         0.044        0.323                0.257        1            1805.65   0.668593 │
# │ fitness_79c5b00c   TERMINATED         0.043        0.305                0.559        1            1798.05   0.654938 │
# │ fitness_7c6d47b9   TERMINATED         0.043        0.322                0.267        1            1899.7    0.663509 │
# │ fitness_a89883ef   TERMINATED         0.042        0.308                0.443        1            1809.89   0.651246 │
# │ fitness_8fc9d6f9   TERMINATED         0.031        0.303                0.514        1            1794.88   0.662199 │
# │ fitness_6a2e10dd   TERMINATED         0.084        0.309                0.427        1            1802.54   0.660839 │
# │ fitness_221c201e   TERMINATED         0.044        0.246                0.452        1            1810.82   0.659818 │
# │ fitness_d86b4236   TERMINATED         0.031        0.248                0.422        1            1809.94   0.655912 │
# │ fitness_ef810b7d   TERMINATED         0.028        0.247                0.261        1            1789.12   0.667293 │
# │ fitness_dd998c8f   TERMINATED         0.031        0.238                0.182        1            1801.79   0.662388 │
# │ fitness_ef3fd48a   TERMINATED         0.028        0.235                0.186        1            1898.25   0.667474 │
# │ fitness_1b3f6220   TERMINATED         0.08         0.394                0.396        1            1792.73   0.651575 │
# │ fitness_2e479ead   TERMINATED         0.084        0.244                0.302        1            3125.2    0.652828 │
# │ fitness_8c6bfa2a   TERMINATED         0.061        0.234                0.191        1            1808.34   0.655508 │
# │ fitness_2a68e67d   TERMINATED         0.03         0.397                0.182        1            1816.48   0.657157 │
# │ fitness_1689e72a   TERMINATED         0.06         0.397                0.304        1            1811.32   0.660604 │
# │ fitness_88050c79   TERMINATED         0.059        0.232                0.292        1            1794.62   0.665154 │
# │ fitness_01ed7d91   TERMINATED         0.058        0.274                0.119        1            1809.05   0.661109 │
# │ fitness_58f93ad4   TERMINATED         0.01         0.282                0.29         1            1796.31   0.669783 │
# │ fitness_c58cd5ee   TERMINATED         0.026        0.231                0.293        1            1893.7    0.665766 │
# │ fitness_9b78518b   TERMINATED         0.057        0.124                0.303        1            1808.69   0.654863 │
# │ fitness_abedb3b3   TERMINATED         0.01         0.103                0.299        1            1812.15   0.663423 │
# │ fitness_91bc210c   TERMINATED         0.008        0.12                 0.247        1            1809.58   0.659952 │
# │ fitness_b4d074c1   TERMINATED         0.025        0.337                0.125        1            3120.27   0.651687 │
# │ fitness_0d3318b1   TERMINATED         0.008        0.115                0.13         1            1797.9    0.657291 │
# │ fitness_f923b210   TERMINATED         0.025        0.279                0.125        1            1802.28   0.655145 │
# │ fitness_2d12816d   TERMINATED         0.011        0.116                0.245        1            1805.5    0.657481 │
# │ fitness_b977ef6f   TERMINATED         0.013        0.107                0.251        1            1897.84   0.661559 │
# │ fitness_1b378790   TERMINATED         0.013        0.333                0.249        1            1807.2    0.659035 │
# │ fitness_4a404d6c   TERMINATED         0.012        0.336                0.248        1            1815.43   0.668879 │
# │ fitness_7a49a681   TERMINATED         0.026        0.274                0.257        1            1809.82   0.661606 │
# │ fitness_72cd88d3   TERMINATED         0.025        0.281                0.258        1            1795.23   0.668278 │
# │ fitness_9bed43b3   TERMINATED         0.048        0.332                0.265        1            1798.53   0.659809 │
# │ fitness_43a36066   TERMINATED         0.052        0.336                0.264        1            1803.45   0.659503 │
# │ fitness_21def0c7   TERMINATED         0.048        0.337                0.201        1            1897.72   0.662832 │
# │ fitness_099b9f1d   TERMINATED         0.049        0.286                0.167        1            1805.07   0.664953 │
# │ fitness_dbf3106a   TERMINATED         0.001        0.342                0.207        1            1809.9    0.659352 │
# │ fitness_b6dff385   TERMINATED         0.047        0.342                0.198        1            1813.57   0.663384 │
# │ fitness_3192520f   TERMINATED         0.048        0.365                0.231        1            3122.51   0.66855  │
# │ fitness_aab76d1c   TERMINATED         0.035        0.361                0.232        1            1789.89   0.658929 │
# │ fitness_f922c0d7   TERMINATED         0.036        0.365                0.23         1            1804.14   0.662452 │
# │ fitness_75b58d29   TERMINATED         0.161        0.262                0.282        1            1811.07   0.665578 │
# │ fitness_fd934af7   TERMINATED         0.033        0.378                0.227        1            1899.16   0.665771 │
# │ fitness_a49ef90d   TERMINATED         0.001        0.359                0.278        1            1807.95   0.661037 │
# │ fitness_94136ce3   TERMINATED         0.18         0.363                0.228        1            1812.03   0.651918 │
# │ fitness_1b499620   TERMINATED         0.036        0.374                0.23         1            1816.47   0.654098 │
# │ fitness_ef9106d9   TERMINATED         0.038        0.258                0.281        1            1791.16   0.665313 │
# │ fitness_81d7fa5a   TERMINATED         0.033        0.291                0.274        1            1794.3    0.661784 │
# │ fitness_3c9a779b   TERMINATED         0.039        0.38                 0.32         1            1804.3    0.65779  │
# │ fitness_9c3ff5fd   TERMINATED         0.039        0.186                0.163        1            1894.63   0.663651 │
# │ fitness_1a7be3ef   TERMINATED         0.039        0.288                0.28         1            3119.62   0.664604 │
# │ fitness_59e8e0db   TERMINATED         0.04         0.187                0.317        1            1807.22   0.657215 │
# │ fitness_f45ba08f   TERMINATED         0.041        0.454                0.323        1            1810.59   0.665161 │
# │ fitness_18f67cb1   TERMINATED         0.041        0.465                0.167        1            1808.61   0.657752 │
# │ fitness_9f608815   TERMINATED         0.028        0.221                0.316        1            1790.89   0.660228 │
# │ fitness_0904320d   TERMINATED         0.021        0.41                 0.161        1            1796.67   0.657059 │
# │ fitness_bd11ff8d   TERMINATED         0.027        0.223                0.159        1            1810.35   0.66758  │
# │ fitness_1ba713cd   TERMINATED         0.022        0.214                0.185        1            1895.04   0.669222 │
# │ fitness_0a162752   TERMINATED         0.053        0.22                 0.193        1            1807.12   0.659857 │
# │ fitness_b0708cfc   TERMINATED         0.024        0.293                0.211        1            1807.5    0.663164 │
# │ fitness_ddeeb3c1   TERMINATED         0.024        0.288                0.137        1            1805.91   0.668582 │
# │ fitness_96c70cb0   TERMINATED         0.022        0.005                0.139        1            1797.33   0.59103  │
# │ fitness_fe0ff10f   TERMINATED         0.053        0.151                0.189        1            1792.61   0.666832 │
# │ fitness_11df9fed   TERMINATED         0.015        0.16                 0.187        1            1803.87   0.664623 │
# │ fitness_9e174751   TERMINATED         0.005        0.153                0.105        1            3116.81   0.651646 │
# │ fitness_370e7de2   TERMINATED         0.023        0.222                0.189        1            1896.08   0.66699  │
# │ fitness_b3203201   TERMINATED         0.023        0.202                0.211        1            1803.7    0.664119 │
# │ fitness_98789bf0   TERMINATED         0.017        0.256                0.138        1            1815.78   0.65978  │
# │ fitness_3db1e487   TERMINATED         0.016        0.153                0.187        1            1815.65   0.661463 │
# │ fitness_6cad87ab   TERMINATED         0.015        0.151                0.185        1            1793.87   0.659358 │
# │ fitness_bdd998ae   TERMINATED         0.016        0.198                0.256        1            1796.24   0.665098 │
# │ fitness_1851e690   TERMINATED         0.017        0.198                0.176        1            1803.04   0.655548 │
# │ fitness_9e23ccf3   TERMINATED         0.007        0.249                0.258        1            1893.6    0.665074 │
# │ fitness_e8362886   TERMINATED         0.006        0.257                0.257        1            1803.06   0.665536 │
# │ fitness_9b6291d8   TERMINATED         0.007        0.255                0.26         1            1809.37   0.659174 │
# │ fitness_1ac3a1f6   TERMINATED         0.005        0.264                0.241        1            1812.16   0.663784 │
# │ fitness_b4cca141   TERMINATED         0.03         0.318                0.24         1            3128.23   0.659545 │
# │ fitness_cc42d5f1   TERMINATED         0.029        0.322                0.343        1            1797.23   0.664835 │
# │ fitness_575ddbc9   TERMINATED         0.01         0.324                0.234        1            1796.52   0.667933 │
# │ fitness_b4b49fe6   TERMINATED         0.029        0.323                0.344        1            1805.82   0.655157 │
# │ fitness_0dd746dc   TERMINATED         0.029        0.321                0.237        1            1899.42   0.666172 │
# │ fitness_542aa7c2   TERMINATED         0.027        0.316                0.071        1            1811.73   0.657535 │
# │ fitness_f92c04bb   TERMINATED         0.029        0.324                0.343        1            1807.19   0.662529 │
# │ fitness_6167fdf5   TERMINATED         0.03         0.302                0.219        1            1807.95   0.667841 │
# │ fitness_5d86b20c   TERMINATED         0.028        0.322                0.216        1            1794.4    0.670051 │
# │ fitness_78b383a4   TERMINATED         0.029        0.326                0.237        1            1792.72   0.664448 │
# │ fitness_13e0fcda   TERMINATED         0.033        0.304                0.217        1            1808.18   0.661928 │
# │ fitness_d2dd5ea4   TERMINATED         0.021        0.298                0.219        1            1801.62   0.658731 │
# │ fitness_40c70ae4   TERMINATED         0.021        0.294                0.219        1            3115.5    0.661894 │
# │ fitness_e260b411   TERMINATED         0.011        0.297                0.215        1            1892.5    0.659267 │
# │ fitness_c39efc9b   TERMINATED         0.108        0.299                0.224        1            1811.85   0.662992 │
# │ fitness_d8179bc7   TERMINATED         0.106        0.297                0.214        1            1807.61   0.657971 │
# │ fitness_f2f333c8   TERMINATED         0.104        0.298                0.218        1            1796.33   0.662197 │
# │ fitness_75db0a13   TERMINATED         0.034        0.293                0.222        1            1800.56   0.656248 │
# │ fitness_6050f386   TERMINATED         0.011        0.276                0.198        1            1802.27   0.667916 │
# │ fitness_dd251339   TERMINATED         0.019        0.275                0.212        1            1805.75   0.66467  │
# │ fitness_17514125   TERMINATED         0.012        0.278                0.203        1            1813.37   0.666667 │
# │ fitness_149f6e93   TERMINATED         0.034        0.348                0.206        1            1894.78   0.668201 │
# │ fitness_b020d2bb   TERMINATED         0.02         0.276                0.289        1            1805.14   0.668927 │
# │ fitness_49941fd1   TERMINATED         0.019        0.348                0.198        1            1786.26   0.663696 │
# │ fitness_d35473ee   TERMINATED         0.02         0.346                0.199        1            1797.85   0.655358 │
# │ fitness_041a4550   TERMINATED         0.012        0.273                0.199        1            1802.1    0.670688 │
# │ fitness_8126fc49   TERMINATED         0.012        0.276                0.199        1            3131.15   0.658158 │
# │ fitness_45e89360   TERMINATED         0.134        0.347                0.171        1            1806.88   0.660524 │
# │ fitness_df070c5d   TERMINATED         0.045        0.393                0.171        1            1811.74   0.655979 │
# │ fitness_1a5e2172   TERMINATED         0.011        0.352                0.174        1            1812.87   0.660156 │
# │ fitness_8a882849   TERMINATED         0.024        0.349                0.204        1            1897.43   0.660495 │
# │ fitness_1443f691   TERMINATED         0.046        0.387                0.294        1            1795.9    0.662676 │
# │ fitness_de03b57a   TERMINATED         0.011        0.36                 0.172        1            1795.52   0.664496 │
# │ fitness_fafe1b73   TERMINATED         0.129        0.765                0.289        1            1808.85   0.651992 │
# │ fitness_c74a51b5   TERMINATED         0.012        0.404                0.286        1            1805.86   0.661788 │
# │ fitness_bba45b2b   TERMINATED         0.014        0.278                0.292        1            1812.52   0.660302 │
# │ fitness_877554ce   TERMINATED         0.013        0.275                0.297        1            1809.77   0.669443 │
# │ fitness_4823b73a   TERMINATED         0.012        0.279                0.293        1            1893.38   0.663752 │
# │ fitness_54ee6fa5   TERMINATED         0.012        0.791                0.247        1            1791.8    0.659802 │
# │ fitness_f8bd6d21   TERMINATED         0.034        0.267                0.247        1            3113.8    0.664186 │
# │ fitness_b07bb794   TERMINATED         0.035        0.764                0.27         1            1798.15   0.660772 │
# │ fitness_1b87d954   TERMINATED         0.036        0.266                0.155        1            1807.45   0.667625 │
# │ fitness_1f09daf9   TERMINATED         0.034        0.268                0.24         1            1803.44   0.669229 │
# │ fitness_9ecb84cb   TERMINATED         0.025        0.236                0.231        1            1813.94   0.667139 │
# │ fitness_bf56d810   TERMINATED         0.001        0.272                0.273        1            1814.75   0.666457 │
# │ fitness_52afe334   TERMINATED         0.003        0.268                0.246        1            1897.06   0.666995 │
# │ fitness_9875f850   TERMINATED         0.003        0.372                0.267        1            1793.02   0.665644 │
# │ fitness_aaa82125   TERMINATED         0.004        0.264                0.235        1            1795.41   0.667035 │
# │ fitness_aeffdda9   TERMINATED         0.004        0.702                0.369        1            1803.44   0.652834 │
# │ fitness_c9fe930f   TERMINATED         0.024        0.683                0.233        1            1809.64   0.648035 │
# │ fitness_9117e7db   TERMINATED         0.001        0.312                0.31         1            1816.17   0.659084 │
# │ fitness_6e0f346f   TERMINATED         0.008        0.333                0.244        1            1813.42   0.66203  │
# │ fitness_3730a436   TERMINATED         0.007        0.315                0.309        1            3130.97   0.660054 │
# │ fitness_286fa6ad   TERMINATED         0.019        0.31                 0.305        1            1895.53   0.664275 │
# │ fitness_c49eab83   TERMINATED         0.018        0.312                0.31         1            1793.66   0.658645 │
# │ fitness_61f8dc7e   TERMINATED         0.019        0.317                0.239        1            1791.41   0.661764 │
# │ fitness_91f9cfb8   TERMINATED         0.019        0.314                0.309        1            1805.67   0.661757 │
# │ fitness_55adfe94   TERMINATED         0.008        0.307                0.305        1            1811.32   0.666526 │
# │ fitness_be54e393   TERMINATED         0.018        0.333                0.254        1            1818.44   0.662044 │
# │ fitness_57b859bb   TERMINATED         0.043        0.24                 0.199        1            1811.56   0.660165 │
# │ fitness_f66ae1b5   TERMINATED         0.037        0.287                0.261        1            1893.53   0.667388 │
# │ fitness_2576182d   TERMINATED         0.038        0.242                0.14         1            1794.49   0.663506 │
# │ fitness_a0ebdff1   TERMINATED         0.042        0.242                0.207        1            1794.46   0.672072 │
# │ fitness_4c121278   TERMINATED         0.041        0.291                0.156        1            1804.32   0.666783 │
# │ fitness_f04b513f   TERMINATED         0.037        0.25                 0.151        1            3114.65   0.671217 │
# │ fitness_79111bdb   TERMINATED         0.038        0.287                0.139        1            1800.17   0.663423 │
# │ fitness_fa851e15   TERMINATED         0.038        0.241                0.146        1            1812.48   0.672137 │
# │ fitness_aa48acbe   TERMINATED         0.038        0.286                0.141        1            1814.82   0.662645 │
# │ fitness_e72075a8   TERMINATED         0.032        0.285                0.185        1            1793.27   0.664848 │
# │ fitness_a919ccab   TERMINATED         0.033        0.253                0.148        1            1899.15   0.664107 │
# │ fitness_de7a0793   TERMINATED         0.032        0.214                0.151        1            1801.94   0.665093 │
# │ fitness_96736b2b   TERMINATED         0.032        0.258                0.224        1            1809.64   0.662516 │
# │ fitness_b7262a02   TERMINATED         0.032        0.252                0.207        1            1808.46   0.663838 │
# │ fitness_cc061ab1   TERMINATED         0.031        0.214                0.209        1            1811.63   0.657263 │
# │ fitness_3a785b88   TERMINATED         0.046        0.212                0.209        1            1810.31   0.666175 │
# │ fitness_7685130c   TERMINATED         0.049        0.22                 0.114        1            1793.3    0.669144 │
# │ fitness_77193c32   TERMINATED         0.046        0.211                0.207        1            1892.91   0.669443 │
# │ fitness_3fcbe74e   TERMINATED         0.047        0.227                0.112        1            1793.27   0.657689 │
# │ fitness_7a3e4dde   TERMINATED         0.025        0.214                0.209        1            1804.25   0.653193 │
# │ fitness_ee535bce   TERMINATED         0.045        0.224                0.208        1            3123.32   0.661722 │
# │ fitness_d15ab063   TERMINATED         0.045        0.227                0.21         1            1803.01   0.666054 │
# │ fitness_164fba30   TERMINATED         0.044        0.226                0.227        1            1813.07   0.664934 │
# │ fitness_d6be1b65   TERMINATED         0.05         0.237                0.113        1            1810.14   0.650197 │
# │ fitness_4e0b3e02   TERMINATED         0.05         0.232                0.193        1            1790.81   0.663899 │
# │ fitness_93708121   TERMINATED         0.026        0.232                0.124        1            1796.08   0.65108  │
# │ fitness_3983fcb3   TERMINATED         0.048        0.175                0.127        1            1895.97   0.661256 │
# │ fitness_abd20572   TERMINATED         0.052        0.2                  0.121        1            1799.79   0.660139 │
# │ fitness_f0208b5d   TERMINATED         0.055        0.174                0.12         1            1804      0.66855  │
# │ fitness_ac0ca4e6   TERMINATED         0.015        0.179                0.104        1            1806.45   0.666069 │
# │ fitness_7d8973b3   TERMINATED         0.053        0.245                0.093        1            1812.94   0.655187 │
# │ fitness_f540987f   TERMINATED         0.055        0.195                0.122        1            1789.94   0.660007 │
# │ fitness_09374c3a   TERMINATED         0.025        0.177                0.183        1            3120.88   0.663482 │
# │ fitness_8090e820   TERMINATED         0.041        0.186                0.181        1            1795.41   0.664339 │
# │ fitness_d73b4156   TERMINATED         0.041        0.18                 0.181        1            1807.94   0.662924 │
# │ fitness_59c5a52a   TERMINATED         0.015        0.259                0.18         1            1894.67   0.667353 │
# │ fitness_cafa51d6   TERMINATED         0.057        0.18                 0.097        1            1812.92   0.65116  │
# │ fitness_5ea1e9f2   TERMINATED         0.041        0.192                0.18         1            1814.4    0.656778 │
# │ fitness_7458d3d6   TERMINATED         0.056        0.191                0.188        1            1811.18   0.666995 │
# │ fitness_2fdc4fe9   TERMINATED         0.042        0.169                0.183        1            1794.14   0.66379  │
# │ fitness_e4b0bdcf   TERMINATED         0.041        0.265                0.228        1            1791.3    0.668619 │
# │ fitness_cedd146e   TERMINATED         0.023        0.269                0.231        1            1803.39   0.663266 │
# │ fitness_42b75a9c   TERMINATED         0.023        0.25                 0.222        1            1896.09   0.664919 │
# │ fitness_7076158a   TERMINATED         0.023        0.267                0.226        1            1806.41   0.657485 │
# │ fitness_c450b013   TERMINATED         0.023        0.275                0.226        1            3119.72   0.673076 │
# │ fitness_c7f2cee4   TERMINATED         0.023        0.269                0.23         1            1807.91   0.664546 │
# │ fitness_9f5ca13e   TERMINATED         0.036        0.27                 0.232        1            1803.54   0.664676 │
# │ fitness_58d42282   TERMINATED         0.023        0.269                0.226        1            1791.65   0.664269 │
# │ fitness_9b3e70cf   TERMINATED         0.022        0.266                0.237        1            1795.61   0.663812 │
# │ fitness_2e511e14   TERMINATED         0.036        0.27                 0.277        1            1811.37   0.65776  │
# │ fitness_67eb1e9a   TERMINATED         0.038        0.275                0.238        1            1896.4    0.671414 │
# │ fitness_997a8c74   TERMINATED         0.037        0.278                0.252        1            1811.44   0.66432  │
# │ fitness_44ea289e   TERMINATED         0.037        0.285                0.243        1            1809.81   0.660775 │
# │ fitness_d2056215   TERMINATED         0.038        0.294                0.605        1            1818.57   0.647643 │
# │ fitness_18105ddc   TERMINATED         0.035        0.295                0.198        1            1794.97   0.667928 │
# │ fitness_91ae0f15   TERMINATED         0.037        0.294                0.25         1            1795.91   0.662713 │
# │ fitness_17427ca8   TERMINATED         0.03         0.138                0.251        1            1806.94   0.661319 │
# │ fitness_77ba7629   TERMINATED         0.04         0.245                0.645        1            1898.7    0.646291 │
# │ fitness_f75a3c52   TERMINATED         0.038        0.253                0.249        1            3116.46   0.663009 │
# │ fitness_1a55098c   TERMINATED         0.048        0.365                0.59         1            1801.58   0.638355 │
# │ fitness_b2df740d   TERMINATED         0.048        0.24                 0.259        1            1812.7    0.65883  │
# │ fitness_ba8919f5   TERMINATED         0.029        0.25                 0.268        1            1814.07   0.66949  │
# │ fitness_60e82047   TERMINATED         0.049        0.242                0.264        1            1796.11   0.661512 │
# │ fitness_aad3b5dd   TERMINATED         0.044        0.365                0.264        1            1798.07   0.667682 │
# │ fitness_26cc448c   TERMINATED         0.048        0.245                0.266        1            1807.76   0.668014 │
# │ fitness_a1944681   TERMINATED         0.045        0.373                0.135        1            1896.28   0.658024 │
# │ fitness_bed7ab74   TERMINATED         0.044        0.243                0.215        1            1807.29   0.663752 │
# │ fitness_62a63442   TERMINATED         0.044        0.523                0.268        1            1810.97   0.66133  │
# │ fitness_fda56df9   TERMINATED         0.027        0.247                0.27         1            1811.75   0.665385 │
# │ fitness_82e9ace1   TERMINATED         0.027        0.251                0.276        1            1796      0.667558 │
# │ fitness_cd55ba68   TERMINATED         0.028        0.214                0.277        1            1797.28   0.656803 │
# │ fitness_3af34d8b   TERMINATED         0.031        0.496                0.131        1            1812.72   0.656007 │
# │ fitness_1fa7c905   TERMINATED         0.027        0.206                0.28         1            3105.68   0.660251 │
# │ fitness_60cfea99   TERMINATED         0.028        0.205                0.28         1            1895.56   0.665268 │
# │ fitness_3f2ae922   TERMINATED         0.029        0.629                0.283        1            1800.7    0.653563 │
# │ fitness_05bd8f69   TERMINATED         0.03         0.209                0.077        1            1818.32   0.656454 │
# │ fitness_bd5e01cf   TERMINATED         0.028        0.22                 0.285        1            1812.29   0.659907 │
# │ fitness_59cae7fb   TERMINATED         0.032        0.209                0.241        1            1796.26   0.66135  │
# │ fitness_12236412   TERMINATED         0.034        0.415                0.2          1            1794.17   0.657779 │
# │ fitness_9ab2c62d   TERMINATED         0.015        0.2                  0.284        1            1806.06   0.662207 │
# │ fitness_14b94a96   TERMINATED         0.033        0.207                0.163        1            1893.74   0.66314  │
# │ fitness_79b03e9b   TERMINATED         0.033        0.334                0.162        1            1807.98   0.666236 │
# │ fitness_90c5d6ef   TERMINATED         0.034        0.338                0.111        1            1813.83   0.654289 │
# │ fitness_080d1007   TERMINATED         0.034        0.286                0.159        1            1807.54   0.66338  │
# │ fitness_e4bd71ee   TERMINATED         0.089        0.39                 0.162        1            1791.11   0.653912 │
# │ fitness_0bb8d0bb   TERMINATED         0.075        0.338                0.494        1            3135.73   0.657887 │
# │ fitness_987d67e1   TERMINATED         0.018        0.337                0.167        1            1790.15   0.657309 │
# │ fitness_34e675c9   TERMINATED         0.156        0.338                0.166        1            1803.04   0.663641 │
# │ fitness_5d064002   TERMINATED         0.063        0.33                 0.107        1            1900.98   0.656042 │
# │ fitness_c848f99a   TERMINATED         0.017        0.388                0.216        1            1811.37   0.667299 │
# │ fitness_0ee41351   TERMINATED         0.018        0.283                0.22         1            1810.34   0.667232 │
# │ fitness_dc54d4fc   TERMINATED         0.019        0.257                0.221        1            1805.32   0.665048 │
# │ fitness_d57c606f   TERMINATED         0.019        0.257                0.226        1            1793.37   0.665664 │
# │ fitness_5c4d559c   TERMINATED         0.02         0.259                0.217        1            1789.07   0.666014 │
# │ fitness_11d7ff1d   TERMINATED         0.04         0.258                0.22         1            1803.25   0.667669 │
# │ fitness_76272c50   TERMINATED         0.041        0.257                0.22         1            1898.07   0.667542 │
# │ fitness_aa044a8e   TERMINATED         0.021        0.256                0.218        1            1802.7    0.661962 │
# │ fitness_8d907add   TERMINATED         0.041        0.254                0.221        1            3115.85   0.666784 │
# │ fitness_7bc6e399   TERMINATED         0.041        0.282                0.243        1            1808.01   0.664183 │
# │ fitness_958cb506   TERMINATED         0.041        0.233                0.239        1            1806.89   0.670113 │
# │ fitness_9b01da88   TERMINATED         0.042        0.232                0.197        1            1798.5    0.65656  │
# │ fitness_d9b81bf4   TERMINATED         0.041        0.302                0.197        1            1799.3    0.665167 │
# │ fitness_94d4a394   TERMINATED         0.009        0.282                0.2          1            1807.26   0.655453 │
# │ fitness_1e3c7c79   TERMINATED         0.041        0.306                0.198        1            1893.31   0.666763 │
# │ fitness_f2d1d1f1   TERMINATED         0.053        0.225                0.242        1            1807.43   0.66861  │
# │ fitness_0540f363   TERMINATED         0.052        0.231                0.241        1            1816.65   0.664687 │
# │ fitness_0484ef47   TERMINATED         0.053        0.231                0.24         1            1807.42   0.664626 │
# │ fitness_ede05a09   TERMINATED         0.049        0.22                 0.255        1            1792.25   0.655068 │
# │ fitness_89a6620d   TERMINATED         0.054        0.227                0.254        1            1801.15   0.65352  │
# │ fitness_5f0f427c   TERMINATED         0.052        0.224                0.252        1            1803.32   0.664077 │
# │ fitness_eb3ad95f   TERMINATED         0.015        0.227                0.252        1            3122.88   0.667046 │
# │ fitness_c0269a1a   TERMINATED         0.052        0.228                0.236        1            1806.11   0.669377 │
# │ fitness_6ddcb141   TERMINATED         0.052        0.235                0.238        1            1894.2    0.661331 │
# │ fitness_8deaffff   TERMINATED         0.06         0.221                0.236        1            1810.05   0.663413 │
# │ fitness_bc392f76   TERMINATED         0.06         0.226                0.147        1            1814.6    0.657137 │
# │ fitness_e9a31f66   TERMINATED         0.051        0.239                0.252        1            1795.97   0.654247 │
# │ fitness_171230ac   TERMINATED         0.046        0.237                0.138        1            1798.3    0.65616  │
# │ fitness_396ff192   TERMINATED         0.06         0.161                0.144        1            1807.97   0.664198 │
# │ fitness_86895638   TERMINATED         0.058        0.235                0.235        1            1806.69   0.662084 │
# │ fitness_eef45eb9   TERMINATED         0.057        0.239                0.142        1            1895.65   0.651869 │
# │ fitness_72944492   TERMINATED         0.045        0.243                0.129        1            1804.44   0.658623 │
# │ fitness_3b1d5021   TERMINATED         0.197        0.271                0.295        1            1813.22   0.661455 │
# │ fitness_c41e8676   TERMINATED         0.056        0.273                0.137        1            1792.43   0.65507  │
# │ fitness_12d47a40   TERMINATED         0.066        0.268                0.238        1            1790.74   0.660843 │
# │ fitness_fa008743   TERMINATED         0.055        0.271                0.235        1            3126.02   0.659226 │
# │ fitness_da791399   TERMINATED         0.055        0.269                0.236        1            1799.09   0.661691 │
# │ fitness_61f8cb03   TERMINATED         0.045        0.272                0.124        1            1805.66   0.661817 │
# │ fitness_6895d42a   TERMINATED         0.047        0.271                0.124        1            1895.47   0.665508 │
# │ fitness_d7c93d34   TERMINATED         0.047        0.267                0.296        1            1813.46   0.662237 │
# │ fitness_bcebc7f5   TERMINATED         0.049        0.272                0.266        1            1814.25   0.658059 │
# │ fitness_30b03156   TERMINATED         0.047        0.266                0.093        1            1793.24   0.65623  │
# │ fitness_e7412c55   TERMINATED         0.048        0.301                0.331        1            1800.08   0.6678   │
# │ fitness_440fcceb   TERMINATED         0.048        0.299                0.264        1            1801.9    0.662363 │
# │ fitness_25d464f6   TERMINATED         0.048        0.3                  0.096        1            1802.29   0.65551  │
# │ fitness_396d4005   TERMINATED         0.013        0.297                0.27         1            1805.46   0.664236 │
# │ fitness_a717856a   TERMINATED         0.009        0.305                0.089        1            1808.44   0.660729 │
# │ fitness_fac375e9   TERMINATED         0.011        0.298                0.263        1            1892.79   0.663275 │
# │ fitness_2df66f87   TERMINATED         0.012        0.298                0.268        1            3127.62   0.670864 │
# │ fitness_5b1aaae1   TERMINATED         0.012        0.299                0.265        1            1796.23   0.660461 │
# │ fitness_21e45116   TERMINATED         0.01         0.2                  0.265        1            1794.34   0.674511 │
# │ fitness_58c9b841   TERMINATED         0.013        0.197                0.208        1            1802.57   0.669621 │
# │ fitness_32da5a5d   TERMINATED         0.013        0.199                0.005        1            1807.01   0.649284 │
# │ fitness_f57640a3   TERMINATED         0.025        0.2                  0.206        1            1807.26   0.660613 │
# │ fitness_8e691e63   TERMINATED         0.024        0.209                0.228        1            1817.07   0.656031 │
# │ fitness_ce329274   TERMINATED         0.025        0.196                0.211        1            1790.4    0.671353 │
# │ fitness_51bbf4b7   TERMINATED         0.037        0.203                0.206        1            1892.17   0.661752 │
# │ fitness_34a0fe0f   TERMINATED         0.186        0.196                0.209        1            1797.14   0.6564   │
# │ fitness_3970b6d7   TERMINATED         0.009        0.194                0.206        1            1803.46   0.673995 │
# │ fitness_653fad95   TERMINATED         0.009        0.204                0.209        1            3125.15   0.665959 │
# │ fitness_1a744a55   TERMINATED         0.007        0.194                0.25         1            1805.74   0.658022 │
# │ fitness_f7b9f227   TERMINATED         0.007        0.21                 0.283        1            1809.69   0.663857 │
# │ fitness_511cc5b3   TERMINATED         0.005        0.45                 0.288        1            1810.29   0.661993 │
# │ fitness_cb8fa35f   TERMINATED         0.179        0.197                0.209        1            1796.21   0.658941 │
# │ fitness_503c0f18   TERMINATED         0.006        0.183                0.207        1            1898.06   0.668305 │
# │ fitness_4fb8241a   TERMINATED         0.006        0.216                0.063        1            1793.44   0.652434 │
# │ fitness_90a6b910   TERMINATED         0.006        0.192                0.191        1            1808.16   0.657096 │
# │ fitness_bbcd2d21   TERMINATED         0.007        0.184                0.192        1            1803.96   0.658029 │
# │ fitness_817abb93   TERMINATED         0.004        0.171                0.194        1            1813.76   0.669501 │
# │ fitness_ceb9573f   TERMINATED         0.016        0.186                0.191        1            1805.33   0.663194 │
# │ fitness_13e0083c   TERMINATED         0.015        0.18                 0.19         1            1786.92   0.668358 │
# │ fitness_cd42132a   TERMINATED         0.015        0.163                0.197        1            1892.08   0.659234 │
# │ fitness_4b05284c   TERMINATED         0.015        0.173                0.224        1            1790.02   0.65645  │
# │ fitness_f7384bca   TERMINATED         0.014        0.176                0.186        1            1807.08   0.663448 │
# │ fitness_c24bfc0c   TERMINATED         0.015        0.163                0.181        1            3124.46   0.652251 │
# │ fitness_ed84b4e6   TERMINATED         0.015        0.25                 0.227        1            1810.75   0.667025 │
# │ fitness_157db5d9   TERMINATED         0.001        0.163                0.182        1            1815.56   0.658446 │
# │ fitness_7b0c6744   TERMINATED         0.012        0.161                0.182        1            1816.32   0.660026 │
# │ fitness_8dd38a7f   TERMINATED         0.011        0.17                 0.179        1            1802.33   0.667353 │
# │ fitness_372a09b8   TERMINATED         0.011        0.138                0.175        1            1799.42   0.659065 │
# │ fitness_585e52c6   TERMINATED         0.003        0.135                0.176        1            1893.54   0.663535 │
# │ fitness_33700cc6   TERMINATED         0.002        0.247                0.319        1            1802.37   0.66332  │
# │ fitness_0f8d1f8c   TERMINATED         0.01         0.216                0.177        1            1808.09   0.665462 │
# │ fitness_91e345f9   TERMINATED         0.01         0.14                 0.178        1            1812.62   0.658682 │
# │ fitness_e5cabfdb   TERMINATED         0.01         0.221                0.318        1            1817.02   0.662146 │
# │ fitness_3fa6556d   TERMINATED         0.021        0.216                0.307        1            1788.12   0.665604 │
# │ fitness_e1493ea6   TERMINATED         0.003        0.215                0.322        1            1794.7    0.66341  │
# │ fitness_9538a1e9   TERMINATED         0.02         0.224                0.212        1            3122.94   0.663434 │
# │ fitness_d5e88c61   TERMINATED         0.021        0.242                0.293        1            1895.55   0.658236 │
# │ fitness_367c6de0   TERMINATED         0.02         0.216                0.209        1            1806.1    0.661879 │
# │ fitness_bc9b3100   TERMINATED         0.019        0.223                0.214        1            1812.34   0.662974 │
# │ fitness_346f91be   TERMINATED         0.02         0.213                0.293        1            1815.67   0.662231 │
# │ fitness_e5f66dd0   TERMINATED         0.02         0.216                0.3          1            1811.57   0.66407  │
# │ fitness_1125f813   TERMINATED         0.021        0.048                0.211        1            1791.55   0.6448   │
# │ fitness_ae8fee2c   TERMINATED         0.02         0.248                0.212        1            1795.87   0.670917 │
# │ fitness_47bcea5a   TERMINATED         0.019        0.253                0.217        1            1897.35   0.665938 │
# │ fitness_7509ceb6   TERMINATED         0.026        0.245                0.223        1            1805.44   0.667376 │
# │ fitness_06d2d336   TERMINATED         0.027        0.247                0.274        1            1803.79   0.662185 │
# │ fitness_c5d41417   TERMINATED         0.027        0.193                0.273        1            3123.38   0.662936 │
# │ fitness_3b44042b   TERMINATED         0.026        0.248                0.276        1            1813.96   0.660426 │
# │ fitness_121928fd   TERMINATED         0.029        0.249                0.248        1            1792.75   0.666046 │
# │ fitness_c1180202   TERMINATED         0.027        0.192                0.247        1            1804.03   0.663062 │
# │ fitness_152b836b   TERMINATED         0.026        0.243                0.198        1            1798.26   0.665865 │
# │ fitness_82058b5a   TERMINATED         0.026        0.252                0.205        1            1803.61   0.654624 │
# │ fitness_0e237599   TERMINATED         0.028        0.241                0.197        1            1896.12   0.668043 │
# │ fitness_10000c4c   TERMINATED         0.024        0.281                0.026        1            1804.95   0.648683 │
# │ fitness_38b94aa4   TERMINATED         0.03         0.193                0.225        1            1788.45   0.658016 │
# │ fitness_45b0c9dc   TERMINATED         0.031        0.572                0.201        1            1810      0.65644  │
# │ fitness_13e6c61e   TERMINATED         0.032        0.283                0.2          1            1807.98   0.66492  │
# │ fitness_a6ef965a   TERMINATED         0.032        0.279                0.198        1            1796.86   0.661694 │
# │ fitness_1720c280   TERMINATED         0.032        0.281                0.233        1            1800.16   0.668212 │
# │ fitness_c5254f00   TERMINATED         0.033        0.284                0.229        1            1891.15   0.651874 │
# │ fitness_6aa9d47c   TERMINATED         0.032        0.477                0.228        1            3127.61   0.660944 │
# │ fitness_333068b9   TERMINATED         0.031        0.284                0.225        1            1811.54   0.663847 │
# │ fitness_d269e460   TERMINATED         0.034        0.284                0.232        1            1791.87   0.656325 │
# │ fitness_d5eed448   TERMINATED         0.008        0.231                0.227        1            1801.24   0.668062 │
# │ fitness_c59f6a3b   TERMINATED         0.036        0.234                0.232        1            1810.44   0.664287 │
# │ fitness_8c4b7584   TERMINATED         0.036        0.229                0.236        1            1795.26   0.665096 │
# │ fitness_3a78add8   TERMINATED         0.016        0.234                0.403        1            1813.13   0.653515 │
# │ fitness_8cc6da0b   TERMINATED         0.007        0.481                0.215        1            1897.53   0.666245 │
# │ fitness_97c03c40   TERMINATED         0.006        0.232                0.219        1            1801.74   0.662259 │
# │ fitness_237448b2   TERMINATED         0.015        0.228                0.258        1            1786.8    0.664261 │
# │ fitness_23d3430a   TERMINATED         0.118        0.478                0.261        1            1806.81   0.652431 │
# │ fitness_ed955f4a   TERMINATED         0.016        0.234                0.26         1            1813.3    0.663405 │
# │ fitness_bc7f78c7   TERMINATED         0.121        0.259                0.211        1            1794.43   0.660453 │
# │ fitness_43be1260   TERMINATED         0.016        0.258                0.213        1            1803.58   0.664453 │
# │ fitness_40213c82   TERMINATED         0.016        0.261                0.257        1            3121.73   0.659674 │
# │ fitness_37578e65   TERMINATED         0.013        0.26                 0.257        1            1892.56   0.664268 │
# │ fitness_ff8608ae   TERMINATED         0.014        0.26                 0.155        1            1813.15   0.663384 │
# │ fitness_f8d72160   TERMINATED         0.14         0.262                0.164        1            1798.9    0.655566 │
# │ fitness_06001d5c   TERMINATED         0.013        0.258                0.156        1            1817.99   0.662375 │
# │ fitness_0b2c03c3   TERMINATED         0.023        0.26                 0.192        1            1818.91   0.667731 │
# │ fitness_e04544ff   TERMINATED         0.11         0.203                0.155        1            1804.86   0.669846 │
# │ fitness_84328d55   TERMINATED         0.011        0.206                0.245        1            1809.22   0.665259 │
# │ fitness_970b8447   TERMINATED         0.038        0.199                0.166        1            1895.6    0.665514 │
# │ fitness_b7addb79   TERMINATED         0.023        0.194                0.166        1            1803.59   0.66164  │
# │ fitness_5813c0ac   TERMINATED         0.09         0.207                0.195        1            1795.3    0.662956 │
# │ fitness_03cfccc1   TERMINATED         0.023        0.207                0.19         1            1807.1    0.669125 │
# │ fitness_f9859202   TERMINATED         0.009        0.202                0.246        1            1810.33   0.661802 │
# │ fitness_3b448b2b   TERMINATED         0.115        0.207                0.151        1            1794.03   0.660847 │
# │ fitness_eefee673   TERMINATED         0.091        0.203                0.168        1            3117.89   0.658212 │
# │ fitness_3d0a79f1   TERMINATED         0.04         0.184                0.174        1            1804.9    0.660578 │
# │ fitness_bb21b5f0   TERMINATED         0.038        0.186                0.186        1            1893.71   0.667476 │
# │ fitness_c24a86c5   TERMINATED         0.094        0.182                0.206        1            1808.92   0.660694 │
# │ fitness_36553d69   TERMINATED         0.114        0.174                0.189        1            1787.3    0.661384 │
# │ fitness_1f2c53d6   TERMINATED         0.103        0.149                0.209        1            1805.48   0.661858 │
# │ fitness_dc91307b   TERMINATED         0.001        0.184                0.206        1            1806.83   0.669401 │
# │ fitness_614825db   TERMINATED         0.111        0.184                0.172        1            1795.62   0.662819 │
# │ fitness_f723a44d   TERMINATED         0.111        0.148                0.2          1            1804.32   0.663763 │
# │ fitness_7b819e04   TERMINATED         0.11         0.168                0.205        1            1890.79   0.66462  │
# │ fitness_bf894a50   TERMINATED         0.006        0.168                0.214        1            1802.77   0.652183 │
# │ fitness_28e44f69   TERMINATED         0.114        0.177                0.211        1            3120.06   0.664989 │
# │ fitness_d0643041   TERMINATED         0.001        0.155                0.215        1            1789.02   0.665221 │
# │ fitness_4ec4e2ed   TERMINATED         0.009        0.223                0.216        1            1812.8    0.663236 │
# │ fitness_d504a80a   TERMINATED         0.002        0.171                0.218        1            1809.51   0.662029 │
# │ fitness_3d2665a4   TERMINATED         0.006        0.154                0.217        1            1794.81   0.664207 │
# │ fitness_cc4ded44   TERMINATED         0.008        0.174                0.217        1            1803.16   0.666278 │
# │ fitness_32484bc7   TERMINATED         0.002        0.168                0.217        1            1894.42   0.665555 │
# │ fitness_9f552298   TERMINATED         0.002        0.431                0.219        1            1799.24   0.662328 │
# │ fitness_a98887c8   TERMINATED         0.097        0.212                0.239        1            1792.06   0.656507 │
# │ fitness_0ba2d0b1   TERMINATED         0.007        0.218                0.242        1            1811.02   0.667729 │
# │ fitness_c2409b98   TERMINATED         0.099        0.19                 0.241        1            1811.66   0.657036 │
# │ fitness_f5e481dd   TERMINATED         0.001        0.215                0.241        1            1789.03   0.660138 │
# │ fitness_6eb79325   TERMINATED         0.005        0.215                0.243        1            1805.28   0.659231 │
# │ fitness_5814a463   TERMINATED         0.126        0.244                0.189        1            3114.22   0.653765 │
# │ fitness_8168168a   TERMINATED         0.1          0.215                0.49         1            1895.8    0.658251 │
# │ fitness_3bc74d57   TERMINATED         0.006        0.316                0.489        1            1802.73   0.659292 │
# │ fitness_35db1c82   TERMINATED         0.102        0.24                 0.238        1            1793.12   0.654271 │
# │ fitness_d9241385   TERMINATED         0.099        0.242                0.191        1            1807.11   0.661033 │
# │ fitness_799f6ad1   TERMINATED         0.005        0.312                0.195        1            1814.77   0.668758 │
# │ fitness_3b34c54d   TERMINATED         0.105        0.238                0.186        1            1797.89   0.670021 │
# │ fitness_f2f9f2f3   TERMINATED         0.011        0.242                0.2          1            1804.97   0.651863 │
# │ fitness_6550c95a   TERMINATED         0.011        0.241                0.191        1            1891.52   0.664807 │
# │ fitness_5525a1fc   TERMINATED         0.012        0.233                0.191        1            1804.75   0.666598 │
# │ fitness_f87e1f3f   TERMINATED         0.106        0.242                0.192        1            1788.26   0.665461 │
# │ fitness_c527f0af   TERMINATED         0.012        0.286                0.198        1            1800.48   0.660954 │
# │ fitness_fcd5513d   TERMINATED         0.129        0.288                0.549        1            1814.76   0.658592 │
# │ fitness_c2d9305f   TERMINATED         0.106        0.284                0.153        1            1794.06   0.657633 │
# │ fitness_32505e14   TERMINATED         0.111        0.283                0.15         1            3096.75   0.662952 │
# │ fitness_2dd9b81f   TERMINATED         0.109        0.125                0.548        1            1804.4    0.64255  │
# │ fitness_8fe8cedf   TERMINATED         0.104        0.289                0.227        1            1804.02   0.66128  │
# │ fitness_a25dffaa   TERMINATED         0.038        0.286                0.15         1            1896.51   0.660934 │
# │ fitness_b3854940   TERMINATED         0.038        0.293                0.147        1            1792.23   0.663542 │
# │ fitness_7616646e   TERMINATED         0.107        0.272                0.148        1            1811.49   0.662963 │
# │ fitness_54f45c89   TERMINATED         0.04         0.271                0.225        1            1812.32   0.666324 │
# │ fitness_7cd87722   TERMINATED         0.11         0.266                0.277        1            1797.24   0.66009  │
# │ fitness_df1ef9dd   TERMINATED         0.106        0.273                0.271        1            1801.51   0.663343 │
# │ fitness_f3611b5a   TERMINATED         0.038        0.269                0.277        1            1799.96   0.667387 │
# │ fitness_f8be5cb2   TERMINATED         0.035        0.268                0.274        1            1898.16   0.667109 │
# │ fitness_c1d12267   TERMINATED         0.151        0.267                0.272        1            1791.89   0.659684 │
# │ fitness_a7023472   TERMINATED         0.018        0.253                0.269        1            1809.93   0.660751 │
# │ fitness_e0bcde17   TERMINATED         0.084        0.256                0.274        1            3121.41   0.660425 │
# │ fitness_f5b7fc38   TERMINATED         0.018        0.196                0.279        1            1811.21   0.661175 │
# │ fitness_2f6d96e8   TERMINATED         0.035        0.452                0.268        1            1800.08   0.661354 │
# │ fitness_9a6d7f85   TERMINATED         0.018        0.197                0.298        1            1800.91   0.657123 │
# │ fitness_a36ea4eb   TERMINATED         0.016        0.411                0.176        1            1804.45   0.660995 │
# │ fitness_083c72f7   TERMINATED         0.017        0.41                 0.302        1            1785.91   0.661106 │
# │ fitness_de5ed738   TERMINATED         0.019        0.407                0.252        1            1891.37   0.662474 │
# │ fitness_dafde5c4   TERMINATED         0.08         0.416                0.172        1            1812.36   0.661801 │
# │ fitness_fa549841   TERMINATED         0.043        0.231                0.252        1            1812.9    0.663999 │
# │ fitness_1d21e9bc   TERMINATED         0.001        0.415                0.176        1            1798.41   0.663809 │
# │ fitness_73a31a4a   TERMINATED         0.009        0.229                0.252        1            1800.09   0.661279 │
# │ fitness_6f6a3339   TERMINATED         0.044        0.418                0.177        1            3114.33   0.665733 │
# │ fitness_3cd76993   TERMINATED         0.028        0.235                0.251        1            1802.75   0.660405 │
# │ fitness_56fd5002   TERMINATED         0.042        0.521                0.253        1            1791.18   0.649621 │
# │ fitness_bce90627   TERMINATED         0.042        0.531                0.228        1            1895.02   0.658141 │
# │ fitness_a5ac4914   TERMINATED         0.009        0.228                0.228        1            1808.42   0.666565 │
# │ fitness_f55b7eb2   TERMINATED         0.029        0.229                0.205        1            1807.44   0.660209 │
# │ fitness_f4024d30   TERMINATED         0.03         0.226                0.205        1            1797.01   0.66859  │
# │ fitness_4cda1fc3   TERMINATED         0.03         0.188                0.233        1            1800.3    0.665968 │
# │ fitness_68aa0d0e   TERMINATED         0.025        0.19                 0.229        1            1797.15   0.665752 │
# │ fitness_9880e343   TERMINATED         0.168        0.252                0.23         1            1786.42   0.660534 │
# │ fitness_52e9beb8   TERMINATED         0.023        0.308                0.205        1            1809.08   0.656687 │
# │ fitness_e6b32096   TERMINATED         0.03         0.187                0.201        1            1888.38   0.658953 │
# │ fitness_8558f336   TERMINATED         0.023        0.31                 0.232        1            1798.84   0.670052 │
# │ fitness_eebe5dd8   TERMINATED         0.004        0.187                0.229        1            1807.88   0.665317 │
# │ fitness_a9edcec1   TERMINATED         0.024        0.251                0.209        1            1805.71   0.673034 │
# │ fitness_f99a055c   TERMINATED         0.023        0.312                0.205        1            3108.44   0.667557 │
# │ fitness_12da9242   TERMINATED         0.004        0.316                0.204        1            1801.05   0.662716 │
# │ fitness_44ff12fb   TERMINATED         0.005        0.208                0.209        1            1787.33   0.670478 │
# │ fitness_0108f695   TERMINATED         0.006        0.205                0.206        1            1812.99   0.664485 │
# │ fitness_6f219ebd   TERMINATED         0.022        0.321                0.205        1            1793.61   0.666453 │
# │ fitness_b5d34911   TERMINATED         0.023        0.312                0.208        1            1806.75   0.667601 │
# │ fitness_79bd259f   TERMINATED         0.024        0.312                0.207        1            1898.22   0.657463 │
# │ fitness_377a3ac6   TERMINATED         0.023        0.294                0.186        1            1803.26   0.65604  │
# │ fitness_96e99156   TERMINATED         0.024        0.323                0.187        1            1799.77   0.665166 │
# │ fitness_6dfa289e   TERMINATED         0.022        0.311                0.185        1            1789.61   0.661631 │
# │ fitness_618c2830   TERMINATED         0.024        0.323                0.219        1            1807.89   0.671057 │
# │ fitness_07431d42   TERMINATED         0.024        0.314                0.187        1            1791.26   0.66067  │
# │ fitness_67236eed   TERMINATED         0.013        0.299                0.185        1            1812.09   0.666345 │
# │ fitness_1690b9c0   TERMINATED         0.014        0.254                0.159        1            1891.93   0.666532 │
# │ fitness_0bd8cf46   TERMINATED         0.01         0.299                0.221        1            3119.06   0.660492 │
# │ fitness_0b546513   TERMINATED         0.013        0.245                0.22         1            1802.86   0.670218 │
# │ fitness_d156e0b7   TERMINATED         0.014        0.251                0.216        1            1800.86   0.674433 │
# │ fitness_24870f6e   TERMINATED         0.013        0.25                 0.223        1            1793.94   0.660381 │
# │ fitness_33ac81ab   TERMINATED         0.015        0.357                0.221        1            1812.61   0.667226 │
# │ fitness_62d23160   TERMINATED         0.015        0.341                0.22         1            1791.16   0.663451 │
# │ fitness_c2dbda56   TERMINATED         0.014        0.35                 0.22         1            1809.37   0.662382 │
# │ fitness_7b65603f   TERMINATED         0.016        0.332                0.223        1            1893.25   0.666777 │
# │ fitness_380844c9   TERMINATED         0.018        0.244                0.223        1            1805.21   0.664923 │
# │ fitness_af664a24   TERMINATED         0.017        0.364                0.218        1            1803.68   0.659866 │
# │ fitness_d622a965   TERMINATED         0.018        0.334                0.22         1            1791.57   0.657052 │
# │ fitness_67937a98   TERMINATED         0.018        0.33                 0.219        1            3129.95   0.667121 │
# │ fitness_a24c5396   TERMINATED         0.018        0.332                0.218        1            1792.36   0.663644 │
# │ fitness_566959fc   TERMINATED         0.017        0.329                0.221        1            1810.74   0.66637  │
# │ fitness_300ff722   TERMINATED         0.02         0.327                0.216        1            1810.17   0.662123 │
# │ fitness_775b5256   TERMINATED         0.009        0.25                 0.237        1            1895.54   0.66406  │
# │ fitness_4cffb95e   TERMINATED         0.019        0.218                0.236        1            1804.32   0.660598 │
# │ fitness_12ef229f   TERMINATED         0.01         0.716                0.241        1            1799.42   0.65053  │
# │ fitness_9929c2a5   TERMINATED         0.009        0.213                0.238        1            1793.03   0.661437 │
# │ fitness_7aebd31d   TERMINATED         0.009        0.213                0.197        1            1789.06   0.660957 │
# │ fitness_9c8afdf7   TERMINATED         0.009        0.215                0.238        1            1805.63   0.664208 │
# │ fitness_b1af30ba   TERMINATED         0.008        0.213                0.241        1            1806.23   0.66379  │
# │ fitness_fa56f006   TERMINATED         0.009        0.217                0.198        1            1896.87   0.674091 │
# │ fitness_aa9bc4f4   TERMINATED         0.011        0.208                0.197        1            1799.2    0.656721 │
# │ fitness_893e0437   TERMINATED         0.117        0.261                0.198        1            3120.43   0.662765 │
# │ fitness_93de7516   TERMINATED         0.009        0.257                0.166        1            1798.94   0.66169  │
# │ fitness_4fbd22e2   TERMINATED         0.116        0.262                0.166        1            1794.07   0.65875  │
# │ fitness_0433546c   TERMINATED         0.116        0.262                0.198        1            1792.68   0.668303 │
# │ fitness_83504dad   TERMINATED         0.118        0.263                0.163        1            1808.91   0.660547 │
# │ fitness_a8e4a28b   TERMINATED         0.026        0.277                0.171        1            1805.41   0.656597 │
# │ fitness_04bdb0ef   TERMINATED         0.026        0.262                0.166        1            1809.38   0.662305 │
# │ fitness_18f6425c   TERMINATED         0.027        0.237                0.165        1            1894.32   0.654187 │
# │ fitness_fde94f45   TERMINATED         0.118        0.236                0.135        1            1804.89   0.663501 │
# │ fitness_b3a7d6cc   TERMINATED         0.026        0.09                 0.134        1            1787.59   0.664424 │
# │ fitness_1bb16269   TERMINATED         0.027        0.233                0.206        1            1792.27   0.659756 │
# │ fitness_690373dd   TERMINATED         0.025        0.24                 0.212        1            1811.4    0.666401 │
# │ fitness_00e638bf   TERMINATED         0.013        0.233                0.131        1            1804.54   0.662243 │
# │ fitness_c679505b   TERMINATED         0.013        0.237                0.209        1            1800.1    0.663278 │
# │ fitness_e2624594   TERMINATED         0.014        0.236                0.209        1            1891.49   0.661307 │
# │ fitness_435f1a92   TERMINATED         0.013        0.076                0.13         1            3108.55   0.658323 │
# │ fitness_02929f84   TERMINATED         0.021        0.296                0.209        1            1802.91   0.665473 │
# │ fitness_1cbef805   TERMINATED         0.021        0.294                0.212        1            1793.35   0.661568 │
# │ fitness_7e0da54d   TERMINATED         0.021        0.283                0.205        1            1798.32   0.659648 │
# │ fitness_580c8df2   TERMINATED         0.015        0.293                0.21         1            1809.04   0.661513 │
# │ fitness_80c9e78a   TERMINATED         0.022        0.286                0.211        1            1812.02   0.663396 │
# │ fitness_be9f0675   TERMINATED         0.021        0.296                0.23         1            1805.1    0.666651 │
# │ fitness_92de0c5f   TERMINATED         0.021        0.293                0.231        1            1890.31   0.66405  │
# │ fitness_2b9fb6b4   TERMINATED         0.022        0.285                0.237        1            1798.52   0.667127 │
# │ fitness_b1873043   TERMINATED         0.021        0.203                0.182        1            1792.11   0.659524 │
# │ fitness_7f49b203   TERMINATED         0.013        0.198                0.232        1            1787.2    0.66776  │
# │ fitness_6b6bffc3   TERMINATED         0.012        0.203                0.23         1            1808.17   0.664246 │
# │ fitness_e2d3e508   TERMINATED         0.005        0.2                  0.231        1            1807.52   0.655606 │
# │ fitness_3382c73a   TERMINATED         0.019        0.197                0.231        1            3115.09   0.659835 │
# │ fitness_6a4358ff   TERMINATED         0.006        0.201                0.182        1            1806.31   0.669605 │
# │ fitness_41222053   TERMINATED         0.007        0.647                0.187        1            1893.8    0.662978 │
# │ fitness_1ee43ea9   TERMINATED         0.005        0.219                0.189        1            1805.84   0.667782 │
# │ fitness_07642e9b   TERMINATED         0.011        0.248                0.19         1            1789.28   0.663993 │
# │ fitness_189180c6   TERMINATED         0.005        0.253                0.188        1            1792.93   0.668836 │
# │ fitness_761f0720   TERMINATED         0.005        0.223                0.193        1            1804.79   0.664135 │
# │ fitness_ee0f0f9a   TERMINATED         0.032        0.221                0.188        1            1811.13   0.660218 │
# │ fitness_8763a15a   TERMINATED         0.033        0.25                 0.192        1            1794.87   0.660484 │
# │ fitness_9343c335   TERMINATED         0.032        0.223                0.195        1            1894.77   0.664922 │
# │ fitness_59412b67   TERMINATED         0.033        0.225                0.251        1            1789.34   0.663529 │
# │ fitness_e1e1ab35   TERMINATED         0.032        0.224                0.253        1            1801.27   0.662131 │
# │ fitness_23f4b1c4   TERMINATED         0.032        0.224                0.249        1            3127.01   0.665233 │
# │ fitness_298efaaf   TERMINATED         0.031        0.273                0.254        1            1792.39   0.66602  │
# │ fitness_9c794146   TERMINATED         0.102        0.276                0.253        1            1812.62   0.650873 │
# │ fitness_d82cc62d   TERMINATED         0.103        0.378                0.248        1            1812.98   0.658857 │
# │ fitness_995c18ce   TERMINATED         0.017        0.269                0.243        1            1803.02   0.669367 │
# │ fitness_b22650d6   TERMINATED         0.017        0.271                0.247        1            1894.83   0.660859 │
# │ fitness_89570123   TERMINATED         0.017        0.27                 0.247        1            1783.58   0.663117 │
# │ fitness_6f0fbbd7   TERMINATED         0.101        0.273                0.22         1            1804.33   0.664261 │
# │ fitness_492536d6   TERMINATED         0.101        0.377                0.226        1            1789.75   0.659716 │
# │ fitness_27ccb117   TERMINATED         0.017        0.378                0.217        1            1808.08   0.663785 │
# │ fitness_6f65ef08   TERMINATED         0.015        0.259                0.222        1            1807.78   0.65779  │
# │ fitness_823de561   TERMINATED         0.016        0.247                0.352        1            1801.23   0.662952 │
# │ fitness_05a1d95f   TERMINATED         0.017        0.248                0.22         1            3108.57   0.664577 │
# │ fitness_e361aa9c   TERMINATED         0.027        0.251                0.222        1            1890.78   0.666212 │
# │ fitness_f3dc1913   TERMINATED         0.124        0.241                0.219        1            1789.93   0.661616 │
# │ fitness_4ea4a53c   TERMINATED         0.027        0.251                0.217        1            1805.17   0.666746 │
# │ fitness_5f4b4167   TERMINATED         0.027        0.243                0.216        1            1793.32   0.666555 │
# │ fitness_e0b1962e   TERMINATED         0.011        0.244                0.175        1            1817.39   0.664257 │
# │ fitness_b8b4f7e5   TERMINATED         0.011        0.247                0.357        1            1817.03   0.667205 │
# │ fitness_48973d5a   TERMINATED         0.028        0.305                0.147        1            1803.1    0.666575 │
# │ fitness_fb7f5548   TERMINATED         0.012        0.245                0.2          1            1890      0.663168 │
# │ fitness_13c1678b   TERMINATED         0.01         0.307                0.457        1            1790.07   0.665014 │
# │ fitness_1804fd5b   TERMINATED         0.01         0.316                0.175        1            1799.64   0.66151  │
# │ fitness_bf54d725   TERMINATED         0.012        0.302                0.201        1            1785.48   0.664124 │
# │ fitness_c8df299b   TERMINATED         0.009        0.349                0.147        1            1809.41   0.661976 │
# │ fitness_6728eede   TERMINATED         0.108        0.21                 0.15         1            1802.73   0.662248 │
# │ fitness_152c2349   TERMINATED         0.008        0.31                 0.199        1            3110.09   0.659474 │
# │ fitness_72192dea   TERMINATED         0.111        0.306                0.201        1            1806.88   0.658391 │
# │ fitness_8234a08e   TERMINATED         0.112        0.004                0.204        1            1896.37   0.586955 │
# │ fitness_badc24e1   TERMINATED         0.11         0.209                0.198        1            1788.46   0.649764 │
# │ fitness_b1b956eb   TERMINATED         0.023        0.212                0.202        1            1809.09   0.663485 │
# │ fitness_a9573bd8   TERMINATED         0.113        0.353                0.199        1            1795.56   0.663768 │
# │ fitness_918e7ebb   TERMINATED         0.023        0.209                0.201        1            1808.62   0.665979 │
# │ fitness_703db990   TERMINATED         0.023        0.18                 0.199        1            1805.48   0.665044 │
# │ fitness_86294c0e   TERMINATED         0.036        0.224                0.234        1            1803.04   0.665661 │
# │ fitness_9bca50f9   TERMINATED         0.096        0.437                0.233        1            1790.74   0.662861 │
# │ fitness_057df5d8   TERMINATED         0.037        0.187                0.235        1            1888.37   0.660161 │
# │ fitness_b862c578   TERMINATED         0.037        0.178                0.234        1            1805.3    0.663902 │
# │ fitness_52f04ff0   TERMINATED         0.068        0.185                0.235        1            1793.61   0.665172 │
# │ fitness_7396a25b   TERMINATED         0.001        0.188                0.233        1            3116.44   0.667123 │
# │ fitness_2a9bab62   TERMINATED         0.036        0.185                0.18         1            1801.06   0.662722 │
# │ fitness_69ad1a4b   TERMINATED         0.036        0.278                0.232        1            1810.18   0.666612 │
# │ fitness_806549ef   TERMINATED         0.019        0.185                0.235        1            1802.89   0.660858 │
# │ fitness_e557ed10   TERMINATED         0.019        0.285                0.177        1            1790.24   0.667474 │
# │ fitness_6094ca31   TERMINATED         0.135        0.279                0.18         1            1802.91   0.66395  │
# │ fitness_abb679d5   TERMINATED         0.02         0.591                0.262        1            1895.57   0.656975 │
# │ fitness_7bc31a19   TERMINATED         0.019        0.283                0.212        1            1795.05   0.664641 │
# │ fitness_4e2f1202   TERMINATED         0.02         0.279                0.215        1            1807.29   0.659504 │
# │ fitness_adab38a0   TERMINATED         0.001        0.232                0.263        1            1810.64   0.668345 │
# │ fitness_423b103d   TERMINATED         0.001        0.281                0.26         1            1804.94   0.667931 │
# │ fitness_a6e95bfa   TERMINATED         0.001        0.228                0.209        1            1792.69   0.668099 │
# │ fitness_1867b97a   TERMINATED         0.015        0.233                0.219        1            3112.11   0.661924 │
# │ fitness_83803b9f   TERMINATED         0.014        0.228                0.262        1            1807.06   0.664483 │
# │ fitness_03a63064   TERMINATED         0.191        0.224                0.213        1            1896.24   0.660264 │
# │ fitness_a2a6cf94   TERMINATED         0.014        0.232                0.116        1            1790.24   0.662269 │
# │ fitness_f94d5254   TERMINATED         0.014        0.228                0.157        1            1808.38   0.663414 │
# │ fitness_62ca6c9f   TERMINATED         0.013        0.258                0.113        1            1809.42   0.659891 │
# │ fitness_63f0514e   TERMINATED         0.075        0.263                0.161        1            1803.06   0.659757 │
# │ fitness_2b0b0f6d   TERMINATED         0.014        0.26                 0.165        1            1788.06   0.655161 │
# │ fitness_3187daa3   TERMINATED         0.029        0.259                0.111        1            1804.06   0.664844 │
# │ fitness_51e0f6f4   TERMINATED         0.076        0.259                0.16         1            1896.67   0.665905 │
# │ fitness_ba57a290   TERMINATED         0.028        0.26                 0.629        1            1793.93   0.639743 │
# │ fitness_1e65d34a   TERMINATED         0.006        0.259                0.158        1            1811.01   0.662601 │
# │ fitness_d731ae89   TERMINATED         0.026        0.262                0.158        1            1809.1    0.663632 │
# │ fitness_3ca02429   TERMINATED         0.028        0.155                0.214        1            1803.13   0.659819 │
# │ fitness_27d72427   TERMINATED         0.029        0.203                0.174        1            3116.9    0.667481 │
# │ fitness_b96cf72d   TERMINATED         0.007        0.199                0.245        1            1791.4    0.662527 │
# │ fitness_064c582b   TERMINATED         0.007        0.321                0.211        1            1801.15   0.672363 │
# │ fitness_c9fc45f0   TERMINATED         0.008        0.326                0.244        1            1795.24   0.668329 │
# │ fitness_ee8ec86e   TERMINATED         0.006        0.212                0.244        1            1897.26   0.666716 │
# │ fitness_be1dbb96   TERMINATED         0.025        0.324                0.211        1            1812.84   0.665222 │
# │ fitness_afacad4c   TERMINATED         0.006        0.322                0.243        1            1813.53   0.665969 │
# │ fitness_771d4b25   TERMINATED         0.005        0.32                 0.244        1            1799.67   0.666349 │
# │ fitness_b6c515ce   TERMINATED         0.023        0.208                0.188        1            1788.35   0.661795 │
# │ fitness_96551b09   TERMINATED         0.006        0.328                0.184        1            1805.05   0.662094 │
# │ fitness_003e779a   TERMINATED         0.006        0.321                0.179        1            1793.07   0.666396 │
# │ fitness_2b29cdd7   TERMINATED         0.006        0.328                0.18         1            1893.98   0.66379  │
# │ fitness_544c46f2   TERMINATED         0.005        0.344                0.181        1            1807.41   0.660826 │
# │ fitness_5325f88a   TERMINATED         0.008        0.299                0.181        1            1804.77   0.670983 │
# │ fitness_421ac709   TERMINATED         0.007        0.334                0.197        1            3118.96   0.668002 │
# │ fitness_51d5024e   TERMINATED         0.01         0.287                0.19         1            1798.95   0.663847 │
# │ fitness_a30abb1b   TERMINATED         0.009        0.295                0.19         1            1791.69   0.663313 │
# │ fitness_983828ef   TERMINATED         0.011        0.299                0.225        1            1805.93   0.668125 │
# │ fitness_f66ee2d9   TERMINATED         0.01         0.3                  0.141        1            1793.17   0.664096 │
# │ fitness_6c3e4b91   TERMINATED         0.01         0.299                0.224        1            1888.64   0.660093 │
# │ fitness_cc9115ab   TERMINATED         0.009        0.306                0.225        1            1808.34   0.662793 │
# │ fitness_2ab517c5   TERMINATED         0.009        0.296                0.193        1            1808.05   0.660469 │
# │ fitness_64007cf1   TERMINATED         0.011        0.295                0.169        1            1797.45   0.665799 │
# │ fitness_0a8c606e   TERMINATED         0.011        0.298                0.134        1            1925.25   0.662322 │
# │ fitness_30be28e8   TERMINATED         0.01         0.302                0.335        1            2083.03   0.664352 │
# │ fitness_a2eec66e   TERMINATED         0.004        0.305                0.172        1            2195.91   0.66492  │
# │ fitness_ef0eb63a   TERMINATED         0.004        0.308                0.142        1            3719.32   0.661392 │
# │ fitness_6fe3d91b   TERMINATED         0.169        0.29                 0.17         1            2223.64   0.657751 │
# │ fitness_dda73615   TERMINATED         0.001        0.276                0.14         1            2234.67   0.664342 │
# │ fitness_dff83964   TERMINATED         0.001        0.276                0.171        1            2311.18   0.661529 │
# │ fitness_6526bde7   TERMINATED         0.106        0.277                0.288        1            2364.67   0.656707 │
# │ fitness_8e4fca27   TERMINATED         0.003        0.742                0.207        1            2595.89   0.656537 │
# │ fitness_7c7bbede   TERMINATED         0.105        0.278                0.291        1            2617.15   0.664096 │
# │ fitness_0a33133c   TERMINATED         0.001        0.273                0.285        1            2550.87   0.67148  │
# │ fitness_f832741f   TERMINATED         0.041        0.734                0.287        1            2569.73   0.660108 │
# │ fitness_6ac2dac1   TERMINATED         0.017        0.276                0.285        1            2571.43   0.673186 │
# │ fitness_96e1d539   TERMINATED         0.041        0.241                0.288        1            2643.51   0.660987 │
# │ fitness_2f649c14   TERMINATED         0.014        0.244                0.206        1            2473.28   0.670928 │
# │ fitness_c13e1aa9   TERMINATED         0.017        0.244                0.208        1            3515.69   0.665452 │
# │ fitness_ec71189c   TERMINATED         0.015        0.243                0.203        1            2738.27   0.665585 │
# │ fitness_bc92b01e   TERMINATED         0.016        0.345                0.199        1            2130.75   0.667712 │
# │ fitness_f212ba11   TERMINATED         0.017        0.349                0.202        1            3392.7    0.656895 │
# │ fitness_5532dab0   TERMINATED         0.001        0.242                0.208        1            2081.81   0.660849 │
# │ fitness_53ba0804   TERMINATED         0.016        0.273                0.306        1            2121.72   0.658163 │
# │ fitness_9d5dd105   TERMINATED         0.017        0.351                0.272        1            2120.65   0.659165 │
# │ fitness_9964a813   TERMINATED         0.017        0.273                0.269        1            2088.04   0.661572 │
# │ fitness_cfa1677e   TERMINATED         0.017        0.271                0.266        1            2090.17   0.662727 │
# │ fitness_889f992f   TERMINATED         0.02         0.27                 0.304        1            4612.98   0.661348 │
# │ fitness_3654fc14   TERMINATED         0.018        0.269                0.273        1            2071.73   0.666208 │
# │ fitness_3d2d7daf   TERMINATED         0.02         0.27                 0.273        1            3475.6    0.667183 │
# │ fitness_80263d37   TERMINATED         0.019        0.268                0.261        1            2175.23   0.668432 │
# │ fitness_e5a82b20   TERMINATED         0.147        0.266                0.32         1            2163.97   0.662593 │
# │ fitness_d10275a4   TERMINATED         0.023        0.264                0.314        1            2152.42   0.65963  │
# │ fitness_b0283d2a   TERMINATED         0.021        0.264                0.311        1            5465.14   0.659454 │
# │ fitness_006eac0c   TERMINATED         0.021        0.315                0.225        1            2405.04   0.666639 │
# │ fitness_162a704f   TERMINATED         0.021        0.263                0.227        1            2360.73   0.660156 │
# │ fitness_f66f8753   TERMINATED         0.024        0.317                0.225        1            2410.07   0.665232 │
# │ fitness_36e1c1e9   TERMINATED         0.024        0.311                0.224        1            2390.03   0.66484  │
# │ fitness_928b04dc   TERMINATED         0.013        0.249                0.227        1            2343.01   0.661558 │
# │ fitness_28869a2a   TERMINATED         0.013        0.319                0.218        1            3617.9    0.655618 │
# │ fitness_1e2169d0   TERMINATED         0.004        0.253                0.224        1            2345.43   0.663105 │
# │ fitness_83e1cc88   TERMINATED         0.013        0.248                0.219        1            4735.73   0.664687 │
# │ fitness_11f44f11   TERMINATED         0.012        0.289                0.217        1            2336.92   0.660611 │
# │ fitness_d81db61c   TERMINATED         0.014        0.287                0.215        1            2421.54   0.670672 │
# │ fitness_9d840a8d   TERMINATED         0.013        0.252                0.214        1            2422.39   0.663232 │
# │ fitness_f8779ddb   TERMINATED         0.007        0.294                0.211        1            2385.23   0.663892 │
# │ fitness_f206ad24   TERMINATED         0.013        0.291                0.258        1            2382.25   0.664849 │
# │ fitness_ccde1841   TERMINATED         0.013        0.294                0.253        1            6520.67   0.668575 │
# │ fitness_85e6e5bf   TERMINATED         0.005        0.291                0.256        1            2290.49   0.659671 │
# │ fitness_3ae4cdfe   TERMINATED         0.013        0.287                0.257        1            2317.23   0.670221 │
# │ fitness_81177b91   TERMINATED         0.007        0.287                0.252        1            4015.44   0.666478 │
# │ fitness_80619ed8   TERMINATED         0.005        0.283                0.256        1            2265.87   0.671338 │
# │ fitness_f94e9d02   TERMINATED         0.009        0.293                0.25         1            2224.91   0.663188 │
# │ fitness_20b16605   TERMINATED         0.005        0.288                0.243        1            2727.74   0.670525 │
# │ fitness_f7eaaa85   TERMINATED         0.004        0.289                0.246        1            8328.86   0.669769 │
# │ fitness_9b0733a0   TERMINATED         0.008        0.28                 0.244        1            2763.42   0.660654 │
# │ fitness_f2672f58   TERMINATED         0.008        0.281                0.258        1            3090.35   0.66484  │
# │ fitness_501fff92   TERMINATED         0.008        0.285                0.297        1            3042.04   0.673385 │
# │ fitness_edebab52   TERMINATED         0.003        0.283                0.282        1            3025.24   0.662589 │
# │ fitness_063c6b8f   TERMINATED         0.004        0.282                0.265        1            6595.1    0.670451 │
# │ fitness_8b046b51   TERMINATED         0.003        0.283                0.264        1            5424.49   0.662525 │
# │ fitness_8ff0241b   TERMINATED         0.001        0.28                 0.274        1            5175.64   0.665997 │
# │ fitness_12e9d0ad   TERMINATED         0.001        0.287                0.252        1            5070.43   0.663355 │
# │ fitness_ba186a7c   TERMINATED         0.003        0.283                0.286        1            5016.16   0.663118 │
# │ fitness_e1ef5b69   TERMINATED         0.002        0.303                0.302        1            4930.92   0.666876 │
# │ fitness_d724519a   TERMINATED         0.001        0.28                 0.292        1            8677.92   0.660319 │
# │ fitness_b76341e9   TERMINATED         0.002        0.278                0.285        1            2603.57   0.669008 │
# │ fitness_0d813368   TERMINATED         0.001        0.297                0.301        1            5024.35   0.669447 │
# │ fitness_01713d41   TERMINATED         0.001        0.306                0.298        1            5207.05   0.657485 │
# │ fitness_db4300ba   TERMINATED         0.007        0.259                0.282        1            2554.97   0.66332  │
# │ fitness_65143e93   TERMINATED         0.006        0.264                0.301        1            2611.71   0.661876 │
# │ fitness_b975dbc8   TERMINATED         0.007        0.26                 0.289        1            2707.04   0.664214 │
# │ fitness_a76b5d08   TERMINATED         0.007        0.264                0.283        1            3922.6    0.668865 │
# │ fitness_8bb43eac   TERMINATED         0.008        0.256                0.284        1            2006.5    0.667886 │
# │ fitness_2c1adc4b   TERMINATED         0.008        0.304                0.275        1            1906.73   0.666814 │
# │ fitness_378458d4   TERMINATED         0.01         0.302                0.296        1            1981.22   0.665506 │
# │ fitness_fad9482e   TERMINATED         0.008        0.308                0.298        1            2121.7    0.670548 │
# │ fitness_24a68f2e   TERMINATED         0.009        0.304                0.292        1            2008.2    0.668027 │
# │ fitness_4bd102e8   TERMINATED         0.009        0.312                0.333        1            5366.72   0.660014 │
# │ fitness_b0963a96   TERMINATED         0.01         0.319                0.297        1            3361.15   0.660015 │
# │ fitness_1cfd2a83   TERMINATED         0.009        0.305                0.331        1            1951.11   0.666092 │
# │ fitness_5a34651a   TERMINATED         0.011        0.311                0.323        1            4646.24   0.66475  │
# │ fitness_6b7b025c   TERMINATED         0.008        0.309                0.193        1            2006.51   0.659112 │
# │ fitness_22bd1870   TERMINATED         0.01         0.312                0.309        1            2114.1    0.663402 │
# │ fitness_4489075a   TERMINATED         0.009        0.323                0.327        1            4635.02   0.664693 │
# │ fitness_91f10c6f   TERMINATED         0.011        0.315                0.27         1            1977.75   0.664455 │
# │ fitness_c65a3fb9   TERMINATED         0.012        0.318                0.312        1            1908.98   0.663369 │
# │ fitness_41b79dc7   TERMINATED         0.012        0.318                0.342        1            1971.19   0.664936 │
# │ fitness_1e4279cd   TERMINATED         0.012        0.321                0.314        1            2106.87   0.658124 │
# │ fitness_ec48e646   TERMINATED         0.013        0.338                0.309        1            3337.32   0.65748  │
# │ fitness_eaed72e2   TERMINATED         0.015        0.334                0.283        1            1973.2    0.659139 │
# │ fitness_3a6a14cc   TERMINATED         0.016        0.271                0.314        1            1900.52   0.669459 │
# │ fitness_1b137646   TERMINATED         0.015        0.332                0.311        1            1978.57   0.662227 │
# │ fitness_15e8838f   TERMINATED         0.005        0.276                0.283        1            2087.28   0.66417  │
# │ fitness_f900265e   TERMINATED         0.005        0.274                0.281        1            4614.37   0.671318 │
# │ fitness_2b30c78c   TERMINATED         0.016        0.272                0.281        1            5323.22   0.666787 │
# │ fitness_6f5a0ad7   TERMINATED         0.016        0.274                0.283        1            4614.26   0.666382 │
# │ fitness_c7b8152a   TERMINATED         0.004        0.271                0.277        1            1971.59   0.667041 │
# │ fitness_eec63e8d   TERMINATED         0.004        0.27                 0.263        1            1915.58   0.661088 │
# │ fitness_c2f2ebb9   TERMINATED         0.005        0.273                0.274        1            1983.62   0.669625 │
# │ fitness_5df2312e   TERMINATED         0.005        0.279                0.269        1            3343.94   0.657026 │
# │ fitness_f5ad4b0d   TERMINATED         0.005        0.248                0.269        1            2103.2    0.664832 │
# │ fitness_271c99cb   TERMINATED         0.005        0.251                0.271        1            1975.36   0.664396 │
# │ fitness_153ed4a7   TERMINATED         0.017        0.255                0.296        1            1906.46   0.665775 │
# │ fitness_5fbc1843   TERMINATED         0.018        0.252                0.294        1            1974.89   0.671851 │
# │ fitness_96a89cee   TERMINATED         0.015        0.294                0.596        1            2092.85   0.65169  │
# │ fitness_60b3e426   TERMINATED         0.016        0.251                0.282        1            4603.19   0.670544 │
# │ fitness_9be461b4   TERMINATED         0.016        0.257                0.297        1            1967.18   0.667509 │
# │ fitness_0d0df01f   TERMINATED         0.016        0.238                0.282        1            1909.01   0.663075 │
# │ fitness_44bb82be   TERMINATED         0.015        0.239                0.29         1            3346.42   0.658133 │
# │ fitness_c6c0b427   TERMINATED         0.013        0.242                0.289        1            4619.97   0.659991 │
# │ fitness_5e157aaa   TERMINATED         0.019        0.239                0.291        1            1976.59   0.6628   │
# │ fitness_8f31ab90   TERMINATED         0.019        0.24                 0.287        1            5334.31   0.669619 │
# │ fitness_7578ab59   TERMINATED         0.019        0.24                 0.292        1            2111.18   0.668953 │
# │ fitness_08297f7a   TERMINATED         0.02         0.239                0.297        1            1979.2    0.66434  │
# │ fitness_173788ef   TERMINATED         0.018        0.249                0.293        1            1915.87   0.654779 │
# │ fitness_36ff2e9c   TERMINATED         0.021        0.249                0.294        1            1976.03   0.665024 │
# │ fitness_79af128a   TERMINATED         0.087        0.253                0.197        1            2097.4    0.662742 │
# │ fitness_2dfe146e   TERMINATED         0.02         0.257                0.192        1            3326.96   0.666415 │
# │ fitness_1e038de6   TERMINATED         0.012        0.248                0.194        1            1972.92   0.665429 │
# │ fitness_c8b214c1   TERMINATED         0.012        0.255                0.196        1            1906.28   0.666756 │
# │ fitness_221a3bb7   TERMINATED         0.012        0.258                0.193        1            4544.73   0.665025 │
# │ fitness_097176e9   TERMINATED         0.012        0.261                0.196        1            1978.6    0.666219 │
# │ fitness_637652b6   TERMINATED         0.024        0.261                0.195        1            4367.13   0.663119 │
# │ fitness_12956eb7   TERMINATED         0.012        0.226                0.196        1            2100.63   0.661733 │
# │ fitness_dda73728   TERMINATED         0.024        0.261                0.262        1            1961.19   0.671807 │
# │ fitness_ea3429a7   TERMINATED         0.022        0.227                0.322        1            1906.06   0.663086 │
# │ fitness_9738819f   TERMINATED         0.025        0.266                0.265        1            3316.13   0.669081 │
# │ fitness_bca1fc05   TERMINATED         0.025        0.224                0.183        1            1987.28   0.659327 │
# │ fitness_d071ef4f   TERMINATED         0.025        0.223                0.265        1            2400.41   0.658835 │
# │ fitness_9654ba66   TERMINATED         0.008        0.29                 0.263        1            1664.63   0.666031 │
# │ fitness_ab374461   TERMINATED         0.023        0.228                0.269        1            1666.11   0.654611 │
# │ fitness_34451fb0   TERMINATED         0.025        0.228                0.271        1            1680.86   0.663026 │
# ╰──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯


