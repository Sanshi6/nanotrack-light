from __future__ import absolute_import
# import _init_paths
import os
import argparse
import numpy as np

from easydict import EasyDict as edict

import ray
from ray import tune
from ray.tune.schedulers import AsyncHyperBandScheduler
from ray.tune.suggest import HyperOptSearch
from hyperopt import hp

from nanotrack.models.backbone.RepVGG import reparameterize_model
from nanotrack.models.model_builder import ModelBuilder
from nanotrack.tracker.tracker_builder import build_tracker
from nanotrack.utils.model_load import load_pretrain
from toolkit.search.tools import auc_otb, eao_vot, eao_vot_rpn, eao_vot_anchor_free
from nanotrack.core.config import cfg

parser = argparse.ArgumentParser(description='tuning for both SiamFC and SiamRPN (works well on VOT)')
parser.add_argument('--arch', dest='arch', default='SiamFCRes22', help='architecture of model')
parser.add_argument('--resume', default='', type=str, required=True, help='resumed model')
parser.add_argument('--gpu_nums', default=4, type=int, help='gpu numbers')
parser.add_argument('--anchor_nums', default=5, type=int,  help='anchor numbers for rpn')
parser.add_argument('--cls_type', default="thicker", type=str,  help='cls/loss type, thicker or thinner or else you defined')
parser.add_argument('--dataset', default='VOT2015', type=str, help='dataset')

args = parser.parse_args()

print('==> TPE works well with both SiamFC and SiamRPN')
print('==> However TPE is slower than GENE')

# prepare tracker -- rpn waited
info = edict()
info.arch = args.arch
info.dataset = args.dataset
info.epoch_test = False
info.cls_type = args.cls_type

# create model and tracker
cfg.merge_from_file(args.config)
# create model
model = ModelBuilder(cfg)
# load model
model = load_pretrain(model, args.snapshot).cuda().eval()
deploy = 1
if deploy:
    model.backbone = reparameterize_model(model.backbone)
# build tracker
tracker = build_tracker(model, cfg)

model.eval()
model = model.cuda()
print('pretrained model has been loaded')


# fitness function
def fitness(config, reporter):
    # 超参数 init 设置
    penalty_k = config["penalty-k"]
    window_influence = config["window_influence"]
    lr = config["lr"]

    model_config = dict()
    model_config['hp'] = dict()
    model_config['hp']['penalty_k'] = penalty_k
    model_config['hp']['window_influence'] = window_influence
    model_config['hp']['lr'] = lr

    # 评价标准
    eao = eao_vot_anchor_free(tracker)
    print("penalty_k: {0}, scale_lr: {1}, window_influence: {2}, eao: {3}".format(
        penalty_k, lr, window_influence, eao))
    reporter(EAO=eao)



if __name__ == "__main__":
    # the resources you computer have, object_store_memory is shm
    ray.init(num_gpus=args.gpu_nums, num_cpus=args.gpu_nums * 8, redirect_output=True, object_store_memory=30000000000)
    tune.register_trainable("fitness", fitness)

    # define search space for SiamFC or SiamRPN
    params = {
        "penalty-k": hp.quniform('scale_step', 0.145, 0.148, 0.150, 0.152, 0.155),
        "window-influence": hp.quniform('scale_penalty', 0.462, 0.465, 0.468, 0.470, 0.472, 0.475, 0.40),
        "lr": hp.quniform('w_influence', 0.385, 0.390, 0.395, 0.400, 0.405, 0.410, 0.415, 0.420),
    }

    tune_spec = {
        "zp_tune": {
            "run": "fitness",
            "trial_resources": {
                "cpu": 1,  # single task cpu num
                "gpu": 0.5,  # single task gpu num
            },
            "num_samples": 10000,  # sample hyperparameters times
            "local_dir": './TPE_results'
        }
    }

    # this procedures will stop
    stop = {
        "EAO": 0.50,                    # EAO >= 0.50,
        # "timesteps_total": 100,         # iteration times
    }
    tune_spec['zp_tune']['stop'] = stop

    scheduler = AsyncHyperBandScheduler(
        # time_attr="timesteps_total",
        reward_attr="EAO",
        max_t=400,
        grace_period=20
    )
    algo = HyperOptSearch(params, max_concurrent=args.gpu_nums * 2 + 1,
                          reward_attr="EAO")  # max_concurrent: the max running task

    tune.run_experiments(tune_spec, search_alg=algo, scheduler=scheduler)



