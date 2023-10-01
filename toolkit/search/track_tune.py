import os
from os.path import join
import cv2
import numpy as np

from got10k.utils.metrics import poly_iou
from nanotrack.utils.bbox import get_axis_aligned_bbox, cxy_wh_2_rect
from toolkit.utils.region import vot_overlap


def run_tracker(tracker, video_name, video):
    frame_counter = 0
    lost_number = 0
    toc = 0
    pred_bboxes = []
    for idx, (img, gt_bbox) in enumerate(video):
        if len(gt_bbox) == 4:
            gt_bbox = [gt_bbox[0], gt_bbox[1],
                       gt_bbox[0], gt_bbox[1] + gt_bbox[3] - 1,
                       gt_bbox[0] + gt_bbox[2] - 1, gt_bbox[1] + gt_bbox[3] - 1,
                       gt_bbox[0] + gt_bbox[2] - 1, gt_bbox[1]]
        tic = cv2.getTickCount()
        if idx == frame_counter:
            cx, cy, w, h = get_axis_aligned_bbox(np.array(gt_bbox))
            gt_bbox_ = [cx - (w - 1) / 2, cy - (h - 1) / 2, w, h]
            tracker.init(img, gt_bbox_)
            pred_bbox = gt_bbox_
            pred_bboxes.append(1)
        elif idx > frame_counter:
            outputs = tracker.track(img)
            pred_bbox = outputs['bbox']
            overlap = vot_overlap(pred_bbox, gt_bbox,
                                  (img.shape[1], img.shape[0]))
            if overlap > 0:

                pred_bboxes.append(pred_bbox)
            else:

                pred_bboxes.append(2)
                frame_counter = idx + 5
                lost_number += 1
        else:
            pred_bboxes.append(0)
        toc += cv2.getTickCount() - tic
    toc /= cv2.getTickFrequency()
    print('Video: {:12s} Time: {:4.1f}s Speed: {:3.1f}fps Lost: {:d}'.format(
        video_name, toc, idx / toc, lost_number))
    return pred_bboxes



def track_tune(tracker, net, video, config):
    arch = config['arch']
    benchmark_name = config['benchmark']
    resume = config['resume']
    hp = config['hp']  # penalty_k, scale_lr, window_influence, adaptive size (for vot2017 or later)

    tracker_path = join('test', (benchmark_name + resume.split('/')[-1].split('.')[0] +
                                 '_small_size_{:.4f}'.format(hp['small_sz']) +
                                 '_big_size_{:.4f}'.format(hp['big_sz']) +
                                 '_penalty_k_{:.4f}'.format(hp['penalty_k']) +
                                 '_w_influence_{:.4f}'.format(hp['window_influence']) +
                                 '_scale_lr_{:.4f}'.format(hp['lr'])).replace('.', '_'))  # no .

    if not os.path.exists(tracker_path):
        os.makedirs(tracker_path)

    if 'VOT' in benchmark_name:
        baseline_path = join(tracker_path, 'baseline')
        video_path = join(baseline_path, video['name'])
        if not os.path.exists(video_path):
            os.makedirs(video_path)
        result_path = join(video_path, video['name'] + '_001.txt')
    else:
        raise ValueError('Only VOT is supported')

    # occ for parallel running
    if not os.path.exists(result_path):
        fin = open(result_path, 'w')
        fin.close()
    else:
        if benchmark_name.startswith('VOT'):
            return 0
        else:
            raise ValueError('Only VOT is supported')


    start_frame, lost_times, toc = 0, 0, 0
    regions = []  # Ray_result and states[1 init / 2 lost / 0 skip]
    image_files, gt = video['image_files'], video['gt']
    for f, image_file in enumerate(image_files):
        im = cv2.imread(image_file)
        if len(im.shape) == 2:
            im = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
        if f == start_frame:  # init
            cx, cy, w, h = get_axis_aligned_bbox(gt[f])
            target_pos = np.array([cx, cy])
            target_sz = np.array([w, h])
            state = tracker.init(im, target_pos, target_sz, net, hp=hp)  # init tracker
            regions.append([float(1)] if 'VOT' in benchmark_name else gt[f])
        elif f > start_frame:  # tracking
            state = tracker.track(state, im)  # track
            location = cxy_wh_2_rect(state['target_pos'], state['target_sz'])
            b_overlap = poly_iou(gt[f], location) if 'VOT' in benchmark_name else 1
            if b_overlap > 0:
                regions.append(location)
            else:
                regions.append([float(2)])
                lost_times += 1
                start_frame = f + 5  # skip 5 frames
        else:  # skip
            regions.append([float(0)])

    # save results for OTB
    if benchmark_name.startswith('VOT'):
        return regions
    else:
        raise ValueError('Only VOT is supported')
