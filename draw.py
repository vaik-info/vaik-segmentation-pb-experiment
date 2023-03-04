import argparse
import os
import glob
import json
import numpy as np
from PIL import Image
import colorsys


def get_classes_color(classes):
    colors = [[0, 0, 0]]
    for classes_index in range(1, len(classes)):
        rgb = colorsys.hsv_to_rgb(classes_index / len(classes), 1, 1)
        colors.append([int(rgb[0] * 255), int(rgb[1] * 255), int(rgb[2] * 255)])
    return colors


def get_image(json_dict_elem, colors):
    vis_image = np.zeros(json_dict_elem['shape'] + [3, ], dtype=np.uint8)
    org_image = np.reshape(np.asarray(json_dict_elem['array']), vis_image.shape[:-1])
    for class_index in range(len(colors)):
        array_index = org_image == class_index
        vis_image[array_index] = colors[class_index]
    return Image.fromarray(vis_image)

def main(input_json_dir_path, input_classes_path, output_dir_path):
    os.makedirs(output_dir_path, exist_ok=True)
    with open(input_classes_path, 'r') as f:
        classes = f.readlines()
    classes = tuple([label.strip() for label in classes])

    colors = get_classes_color(classes)
    for class_label, color in zip(classes, colors):
        canvas = np.zeros((16, 16, 3), dtype=np.uint8)
        canvas[:, :, :] = np.asarray(color)
        Image.fromarray(canvas).save(
            os.path.join(output_dir_path, f'{classes.index(class_label):04d}_{class_label}.png'), quality=100,
            subsampling=0)

    json_path_list = glob.glob(os.path.join(input_json_dir_path, '*.json'))
    json_dict_list = []
    for json_path in json_path_list:
        with open(json_path, 'r') as f:
            json_dict = json.load(f)
            json_dict_list.append(json_dict)
    for json_path, json_dict in zip(json_path_list, json_dict_list):
        gt_image = get_image(json_dict['answer'], colors)
        output_gt_image_path = os.path.join(output_dir_path, f'draw_{os.path.basename(json_path)}_gt.png')
        gt_image.save(output_gt_image_path, quality=100, subsampling=0)
        pred_image = get_image(json_dict['labels'], colors)
        output_pred_image_path = os.path.join(output_dir_path, f'draw_{os.path.basename(json_path)}_pred.png')
        pred_image.save(output_pred_image_path, quality=100, subsampling=0)




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='inference')
    parser.add_argument('--input_json_dir_path', type=str, default='~/.vaik-segmentation-pb-experiment/test_images_out')
    parser.add_argument('--input_classes_path', type=str,  default=os.path.join(os.path.dirname(__file__), 'test_images/classes.txt'))
    parser.add_argument('--output_dir_path', type=str,  default='~/.vaik-segmentation-pb-experiment/test_images_out_draw')
    args = parser.parse_args()

    args.input_json_dir_path = os.path.expanduser(args.input_json_dir_path)
    args.input_classes_path = os.path.expanduser(args.input_classes_path)
    args.output_dir_path = os.path.expanduser(args.output_dir_path)
    main(**args.__dict__)