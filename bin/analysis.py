import os
import glob
import numpy as np

def load_results(folder):
    # 加载跟踪结果文件，返回一个包含所有序列的字典
    results = {}
    for result_file in glob.glob(os.path.join(folder, '*.txt')):
        seq_name = os.path.basename(result_file).split('.')[0]
        with open(result_file, 'r') as f:
            results[seq_name] = [line.strip().split(',') for line in f.readlines()]
    return results

def calculate_iou(gt_boxes, pred_boxes):
    # 计算 IoU，并返回平均 IoU
    ious = []
    for gt_box, pred_box in zip(gt_boxes, pred_boxes):
        x1 = max(gt_box[0], pred_box[0])
        y1 = max(gt_box[1], pred_box[1])
        x2 = min(gt_box[2], pred_box[2])
        y2 = min(gt_box[3], pred_box[3])

        intersection = max(x2 - x1 + 1, 0) * max(y2 - y1 + 1, 0)
        union = (gt_box[2] - gt_box[0] + 1) * (gt_box[3] - gt_box[1] + 1) + \
                (pred_box[2] - pred_box[0] + 1) * (pred_box[3] - pred_box[1] + 1) - intersection
        iou = intersection / union
        ious.append(iou)

    return np.mean(ious)

hp_search_folder = r"E:\SiamProject\SiamTrackers-master\NanoTrack\hp_search_result"
best_iou = 0
best_hp_combination = None

# 遍历 hp_search_result 文件夹中的子文件夹
for hp_combination_folder in os.listdir(hp_search_folder):
    hp_combination_path = os.path.join(hp_search_folder, hp_combination_folder)

    if not os.path.isdir(hp_combination_path):
        continue

    # 加载此超参数组合的跟踪结果
    results = load_results(hp_combination_path)

    # 初始化性能指标列表
    iou_list = []

    # 遍历序列，计算性能指标
    for seq_name, pred_boxes in results.items():
        gt_boxes = load_ground_truth(seq_name)  # 您需要实现一个加载真实边界框的函数
        iou = calculate_iou(gt_boxes, pred_boxes)
        iou_list.append(iou)

    # 计算此超参数组合的平均性能指标
    mean_iou = np.mean(iou_list)

    # 更新最佳超参数组合
    if mean_iou > best_iou:
        best_iou = mean_iou
        best_hp_combination = hp_combination_folder

print("Best hyperparameter combination: {}".format(best_hp_combination))
print("Best average IoU: {:.4f}".format(best_iou))



























