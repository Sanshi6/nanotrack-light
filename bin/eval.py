from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import argparse
import numpy as np

import sys

sys.path.append(os.path.abspath('.'))

from glob import glob
from tqdm import tqdm
from multiprocessing import Pool
from toolkit.datasets import OTBDataset, UAVDataset, LaSOTDataset, \
    VOTDataset, NFSDataset, VOTLTDataset, DTB70Dataset, UAVDTDataset, VisDroneDataset
from toolkit.evaluation import OPEBenchmark, AccuracyRobustnessBenchmark, \
    EAOBenchmark, F1Benchmark

import warnings

warnings.filterwarnings('ignore')
from got10k.experiments import ExperimentGOT10k


# args.tracker_path
# args.dataset
# args.tracker_name
# args.num
def eval(args):
    tracker_dir = os.path.join(args.tracker_path, args.dataset)
    trackers = glob(os.path.join(args.tracker_path,
                                 args.dataset,
                                 args.tracker_name + '*').replace("\\", "/"))
    trackers = [x.split('/')[-1].split("\\")[-1] for x in trackers]
    assert len(trackers) > 0
    args.num = min(args.num, len(trackers))
    root = './datasets'
    root = os.path.join(root, args.dataset)
    if 'OTB' in args.dataset:
        dataset = OTBDataset(args.dataset, root)
        dataset.set_tracker(tracker_dir, trackers)
        benchmark = OPEBenchmark(dataset)
        success_ret = {}

        # with Pool(processes=args.num) as pool:
        #     for ret in tqdm(pool.imap_unordered(benchmark.eval_success,trackers), desc='eval success', total=len(trackers), ncols=100):
        #         success_ret.update(ret)

        for tracker in trackers:
            ret = benchmark.eval_success(tracker)
            success_ret.update(ret)

        precision_ret = {}
        # with Pool(processes=args.num) as pool:
        #     for ret in tqdm(pool.imap_unordered(benchmark.eval_precision,
        #         trackers), desc='eval precision', total=len(trackers), ncols=100):
        #         precision_ret.update(ret)

        for tracker in trackers:
            ret = benchmark.eval_precision(tracker)
            precision_ret.update(ret)

        # with Pool(processes=args.num) as pool:
        #     for ret in pool.imap_unordered(benchmark.eval, trackers):
        #         eao_result.update(ret)

        # -> modify

        # for tracker in trackers:
        #     ret = benchmark.eval(tracker)
        #     eao_result.update(ret)

        benchmark.show_result(success_ret, precision_ret,
                              show_video_level=args.show_video_level)

        tracker_auc = {}
        for tracker_name in success_ret.keys():
            auc = np.mean(list(success_ret[tracker_name].values()))
            tracker_auc[tracker_name] = auc
        tracker_auc_ = sorted(tracker_auc.items(),
                              key=lambda x: x[1],
                              reverse=True)[:20]

        return tracker_auc[args.tracker_name]  # for test

        #
        # ar_benchmark.show_result(ar_result, eao_result,
        #         show_video_level=args.show_video_level)
        # return eao_result[args.tracker_name]['all']             # for test


    elif 'DTB70' in args.dataset:
        dataset = DTB70Dataset(args.dataset, root)
        dataset.set_tracker(tracker_dir, trackers)
        benchmark = OPEBenchmark(dataset)
        success_ret = {}
        with Pool(processes=args.num) as pool:
            for ret in tqdm(pool.imap_unordered(benchmark.eval_success, trackers), desc='eval success',
                            total=len(trackers), ncols=100):
                success_ret.update(ret)
        precision_ret = {}
        with Pool(processes=args.num) as pool:
            for ret in tqdm(pool.imap_unordered(benchmark.eval_precision,
                                                trackers), desc='eval precision', total=len(trackers), ncols=100):
                precision_ret.update(ret)
        benchmark.show_result(success_ret, precision_ret,
                              show_video_level=args.show_video_level)

    elif 'UAVDT' in args.dataset:
        dataset = UAVDTDataset(args.dataset, root)
        dataset.set_tracker(tracker_dir, trackers)
        benchmark = OPEBenchmark(dataset)
        success_ret = {}
        with Pool(processes=args.num) as pool:
            for ret in tqdm(pool.imap_unordered(benchmark.eval_success, trackers), desc='eval success',
                            total=len(trackers), ncols=100):
                success_ret.update(ret)
        precision_ret = {}
        with Pool(processes=args.num) as pool:
            for ret in tqdm(pool.imap_unordered(benchmark.eval_precision,
                                                trackers), desc='eval precision', total=len(trackers), ncols=100):
                precision_ret.update(ret)
        benchmark.show_result(success_ret, precision_ret,
                              show_video_level=args.show_video_level)

    elif 'VisDrone' in args.dataset:
        dataset = VisDroneDataset(args.dataset, root)
        dataset.set_tracker(tracker_dir, trackers)
        benchmark = OPEBenchmark(dataset)
        success_ret = {}
        with Pool(processes=args.num) as pool:
            for ret in tqdm(pool.imap_unordered(benchmark.eval_success, trackers), desc='eval success',
                            total=len(trackers), ncols=100):
                success_ret.update(ret)
        precision_ret = {}
        with Pool(processes=args.num) as pool:
            for ret in tqdm(pool.imap_unordered(benchmark.eval_precision,
                                                trackers), desc='eval precision', total=len(trackers), ncols=100):
                precision_ret.update(ret)
        benchmark.show_result(success_ret, precision_ret,
                              show_video_level=args.show_video_level)

    elif 'GOT-10k' in args.dataset:
        root_dir = os.path.abspath('datasets/GOT-10k')
        e = ExperimentGOT10k(root_dir)
        ao, sr, speed = e.report([args.tracker_name])
        ss = 'ao:%.3f --sr:%.3f -speed:%.3f' % (float(ao), float(sr), float(speed))
        print(ss)

    elif 'LaSOT' == args.dataset:
        dataset = LaSOTDataset(args.dataset, root)
        dataset.set_tracker(tracker_dir, trackers)
        benchmark = OPEBenchmark(dataset)
        success_ret = {}
        with Pool(processes=args.num) as pool:
            for ret in tqdm(pool.imap_unordered(benchmark.eval_success,
                                                trackers), desc='eval success', total=len(trackers), ncols=100):
                success_ret.update(ret)
        precision_ret = {}
        with Pool(processes=args.num) as pool:
            for ret in tqdm(pool.imap_unordered(benchmark.eval_precision,
                                                trackers), desc='eval precision', total=len(trackers), ncols=100):
                precision_ret.update(ret)
        norm_precision_ret = {}
        with Pool(processes=args.num) as pool:
            for ret in tqdm(pool.imap_unordered(benchmark.eval_norm_precision,
                                                trackers), desc='eval norm precision', total=len(trackers), ncols=100):
                norm_precision_ret.update(ret)
        benchmark.show_result(success_ret, precision_ret, norm_precision_ret,
                              show_video_level=args.show_video_level)
    elif 'UAV' in args.dataset:  # 注意UAVDT和 UAV123 以及 UAV20L的区别
        dataset = UAVDataset(args.dataset, root)
        dataset.set_tracker(tracker_dir, trackers)
        benchmark = OPEBenchmark(dataset)
        success_ret = {}
        with Pool(processes=args.num) as pool:
            for ret in tqdm(pool.imap_unordered(benchmark.eval_success,
                                                trackers), desc='eval success', total=len(trackers), ncols=100):
                success_ret.update(ret)
        precision_ret = {}
        with Pool(processes=args.num) as pool:
            for ret in tqdm(pool.imap_unordered(benchmark.eval_precision,
                                                trackers), desc='eval precision', total=len(trackers), ncols=100):
                precision_ret.update(ret)
        benchmark.show_result(success_ret, precision_ret,
                              show_video_level=args.show_video_level)
    elif 'NFS' in args.dataset:
        dataset = NFSDataset(args.dataset, root)
        dataset.set_tracker(tracker_dir, trackers)
        benchmark = OPEBenchmark(dataset)
        success_ret = {}
        with Pool(processes=args.num) as pool:
            for ret in tqdm(pool.imap_unordered(benchmark.eval_success,
                                                trackers), desc='eval success', total=len(trackers), ncols=100):
                success_ret.update(ret)
        precision_ret = {}
        with Pool(processes=args.num) as pool:
            for ret in tqdm(pool.imap_unordered(benchmark.eval_precision,
                                                trackers), desc='eval precision', total=len(trackers), ncols=100):
                precision_ret.update(ret)
        benchmark.show_result(success_ret, precision_ret,
                              show_video_level=args.show_video_level)
    elif args.dataset in ['VOT2016', 'VOT2017', 'VOT2018', 'VOT2019']:
        dataset = VOTDataset(args.dataset, root)
        dataset.set_tracker(tracker_dir, trackers)
        ar_benchmark = AccuracyRobustnessBenchmark(dataset)
        ar_result = {}
        for tracker in trackers:
            ret = ar_benchmark.eval(tracker)
            ar_result.update(ret)

        # with Pool(processes=args.num) as pool:
        #     for ret in pool.imap_unordered(ar_benchmark.eval, trackers):
        #         ar_result.update(ret)

        benchmark = EAOBenchmark(dataset)
        eao_result = {}

        for tracker in trackers:
            ret = benchmark.eval(tracker)
            eao_result.update(ret)

        # with Pool(processes=args.num) as pool:
        #     for ret in pool.imap_unordered(benchmark.eval, trackers):
        #         eao_result.update(ret)

        ar_benchmark.show_result(ar_result, eao_result,
                                 show_video_level=args.show_video_level)
        return eao_result[args.tracker_name]['all']  # for test
    elif 'VOT2018-LT' == args.dataset:
        dataset = VOTLTDataset(args.dataset, root)
        dataset.set_tracker(tracker_dir, trackers)
        benchmark = F1Benchmark(dataset)
        f1_result = {}
        with Pool(processes=args.num) as pool:
            for ret in tqdm(pool.imap_unordered(benchmark.eval,
                                                trackers), desc='eval f1', total=len(trackers), ncols=100):
                f1_result.update(ret)
        benchmark.show_result(f1_result,
                              show_video_level=args.show_video_level)
    else:
        print(5)
    print("skip something")


# shell command
# python ./bin/eval.py \
# --tracker_path ./hp_search_result \
# --dataset VOT2018 \
# --num 4 \
# --tracker_name  'checkpoint*' 

if __name__ == '__main__':
    tracker_name = 'nanotrack'
    # tracker_name = 'NanoTrack'

    dataset = 'OTB100'

    parser = argparse.ArgumentParser(description='tracking evaluation')
    parser.add_argument('--tracker_path', '-p', default='E:/SiamProject/NanoTrack/results', type=str,
                        help='tracker Ray_result path')
    parser.add_argument('--dataset', '-d', default=dataset, type=str,
                        help='dataset name')
    parser.add_argument('--num', '-n', default=4, type=int,
                        help='number of thread to eval')
    parser.add_argument('--tracker_name', '-t', default=tracker_name,
                        type=str, help='tracker name')
    parser.add_argument('--show_video_level', '-s', dest='show_video_level',
                        action='store_true')
    parser.set_defaults(show_video_level=False)

    args = parser.parse_args()

    eval(args)
