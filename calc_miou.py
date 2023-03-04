import argparse
import os
import glob
import json
import numpy as np


def intersect_and_union(pred_label, label, num_classes, ignore_index):
    mask = (label != ignore_index)
    pred_label = pred_label[mask]
    label = label[mask]

    intersect = pred_label[pred_label == label]
    area_intersect, _ = np.histogram(intersect, bins=np.arange(num_classes + 1))
    area_pred_label, _ = np.histogram(pred_label, bins=np.arange(num_classes + 1))
    area_label, _ = np.histogram(label, bins=np.arange(num_classes + 1))
    area_union = area_pred_label + area_label - area_intersect

    return area_intersect, area_union, area_pred_label, area_label


def mean_iou(results, gt_seg_maps, num_classes, ignore_index=0):
    total_area_intersect = np.zeros((num_classes,), dtype=np.float32)
    total_area_union = np.zeros((num_classes,), dtype=np.float32)
    total_area_pred_label = np.zeros((num_classes,), dtype=np.float32)
    total_area_label = np.zeros((num_classes,), dtype=np.float32)
    for i in range(results.shape[0]):
        area_intersect, area_union, area_pred_label, area_label = \
            intersect_and_union(results[i], gt_seg_maps[i], num_classes, ignore_index=-1)
        total_area_intersect += area_intersect
        total_area_union += area_union
        total_area_pred_label += area_pred_label
        total_area_label += area_label
    acc = total_area_intersect / total_area_label
    iou = total_area_intersect / total_area_union

    mask = np.ones(iou.shape, dtype=bool)
    mask[ignore_index] = False
    return np.mean(iou[mask]), acc, iou


def calc_mIoU(json_dict_list, classes, ignore_label):
    pred_list = []
    gt_list = []
    for json_dict in json_dict_list:
        pred_list.append(json_dict['labels']['array'])
        gt_list.append(json_dict['answer']['array'])
    miou, acc, iou = mean_iou(np.vstack(pred_list), np.vstack(gt_list), len(classes), classes.index(ignore_label))

    print(f'mIoU:{miou:.4f}')
    for index, label in enumerate(classes):
        print(f'{label}: {iou[index]:.4f}')


def main(input_json_dir_path, input_classes_path, ignore_label):
    with open(input_classes_path, 'r') as f:
        classes = f.readlines()
    classes = tuple([label.strip() for label in classes])

    json_path_list = glob.glob(os.path.join(input_json_dir_path, '*.json'))
    json_dict_list = []
    for json_path in json_path_list:
        with open(json_path, 'r') as f:
            json_dict = json.load(f)
            json_dict_list.append(json_dict)
    calc_mIoU(json_dict_list, classes, ignore_label)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='inference')
    parser.add_argument('--input_json_dir_path', type=str, default='~/.vaik-segmentation-pb-experiment/test_images_out')
    parser.add_argument('--input_classes_path', type=str,
                        default=os.path.join(os.path.dirname(__file__), 'test_images/classes.txt'))
    parser.add_argument('--ignore_label', type=str, default='background')
    args = parser.parse_args()

    args.input_json_dir_path = os.path.expanduser(args.input_json_dir_path)
    args.input_classes_path = os.path.expanduser(args.input_classes_path)

    main(**args.__dict__)
