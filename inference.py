import argparse
import os
import glob
import json
import tqdm
import numpy as np
from PIL import Image
from tqdm import tqdm
from vaik_segmentation_pb_inference.pb_model import PbModel


def main(input_saved_model_dir_path, input_classes_path, input_image_dir_path, answer_image_dir_path, output_dir_path):
    os.makedirs(output_dir_path, exist_ok=True)
    with open(input_classes_path, 'r') as f:
        classes = f.readlines()
    classes = tuple([label.strip() for label in classes])

    model = PbModel(input_saved_model_dir_path, classes)

    types = ('*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG')
    image_path_list = []
    for file in types:
        image_path_list.extend(glob.glob(os.path.join(input_image_dir_path, f'{file}'), recursive=True))
    image_list = []
    for image_path in tqdm(image_path_list, 'read images'):
        image = np.asarray(Image.open(image_path).convert('RGB'))
        image_list.append(image)
    import time
    start = time.time()
    output, raw_pred = model.inference(image_list)
    end = time.time()

    for image_path, output_elem in zip(image_path_list, output):
        answer_image = np.asarray(
            Image.open(os.path.join(answer_image_dir_path, os.path.basename(image_path).replace('raw', 'seg'))).convert(
                'L'))
        output_json_path = os.path.join(output_dir_path, os.path.splitext(os.path.basename(image_path))[0] + '.json')
        output_elem['answer'] = {'array': answer_image.flatten().tolist(), 'shape': answer_image.shape}
        output_elem['labels'] = {'array': output_elem['labels'].flatten().tolist(),
                                 'shape': output_elem['labels'].shape}
        with open(output_json_path, 'w') as f:
            json.dump(output_elem, f, ensure_ascii=False, indent=4, sort_keys=True, separators=(',', ': '))
    print(f'{len(image_list) / (end - start)}[images/sec]')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='inference')
    parser.add_argument('--input_saved_model_dir_path', type=str,
                        default='~/output_model/2023-03-04-19-07-35/step-5000_batch-8_epoch-10_loss_0.0010_one_hot_mean_io_u_0.8857_val_loss_0.0016_val_one_hot_mean_io_u_0.8710')
    parser.add_argument('--input_classes_path', type=str,
                        default=os.path.join(os.path.dirname(__file__), 'test_images/classes.txt'))
    parser.add_argument('--input_image_dir_path', type=str,
                        default=os.path.join(os.path.dirname(__file__), 'test_images/raw'))
    parser.add_argument('--answer_image_dir_path', type=str,
                        default=os.path.join(os.path.dirname(__file__), 'test_images/seg'))
    parser.add_argument('--output_dir_path', type=str, default='~/.vaik-segmentation-pb-experiment/test_images_out')
    args = parser.parse_args()

    args.input_saved_model_dir_path = os.path.expanduser(args.input_saved_model_dir_path)
    args.input_classes_path = os.path.expanduser(args.input_classes_path)
    args.input_image_dir_path = os.path.expanduser(args.input_image_dir_path)
    args.answer_image_dir_path = os.path.expanduser(args.answer_image_dir_path)
    args.output_dir_path = os.path.expanduser(args.output_dir_path)

    main(**args.__dict__)
