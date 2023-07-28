import random

from bin.search.eval_otb import eval_auc_tune
from toolkit.datasets import DatasetFactory
from toolkit.search.track_tune import track_tune, run_tracker
import matlab.engine

eng = matlab.engine.start_matlab()  # for test eao in vot-toolkit
eng.cd('bin/search')


def auc_otb(tracker, net, config):
    """
    get AUC for OTB benchmark
    """
    global result_path
    dataset = DatasetFactory.create_dataset(name="VOT2019",
                                            dataset_root="datasets/VOT2019",
                                            load_img=False)
    video_keys = list(dataset.keys()).copy()
    random.shuffle(video_keys)
    for video in video_keys:
        result_path = track_tune(tracker, net, dataset[video], config)

    auc = eval_auc_tune(result_path, config['benchmark'])

    return auc


def eao_vot(tracker, net, config):
    dataset = DatasetFactory.create_dataset(name="VOT2019",
                                            dataset_root="datasets/VOT2019",
                                            load_img=False)
    video_keys = sorted(list(dataset.keys()).copy())
    results = []
    for video in video_keys:
        video_result = track_tune(tracker, net, dataset[video], config)
        results.append(video_result)

    channel = config['benchmark'].split('VOT')[-1]

    eng.cd('bin.search')
    eao = eng.get_eao(results, channel)

    return eao


def eao_vot_rpn(tracker, net, config):
    dataset = DatasetFactory.create_dataset(name="VOT2019",
                                            dataset_root="datasets/VOT2019",
                                            load_img=False)
    video_keys = sorted(list(dataset.keys()).copy())
    results = []
    for video in video_keys:
        video_result = track_tune(tracker, net, dataset[video], config)
        results.append(video_result)

    year = config['benchmark'][-4:]  # need a str, instead of a int
    eng.cd('bin.search')
    eao = eng.get_eao(results, year)

    return eao


def eao_vot_anchor_free(tracker):
    dataset = DatasetFactory.create_dataset(name="VOT2019",
                                            dataset_root="datasets/VOT2019",
                                            load_img=False)
    video_keys = sorted(list(dataset.keys()).copy())

    results = []
    for video in video_keys:
        video_result = run_tracker(tracker, dataset[video], video)
        results.append(video_result)

    year = "2023"  # need a str, instead of a int
    eng.cd('bin.search')
    eao = eng.get_eao(results, year)

    return eao