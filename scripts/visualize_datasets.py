import os

import argparse
import cv2
import numpy as np
from tqdm import tqdm

from modules.factory import train_dataset, valid_dataset, test_dataset
from modules.utils import create_folder


ORIGINAL_DATA_TRAIN = "/workdir/data/images/train"
ORIGINAL_DATA_TEST = "/workdir/data/images/test"


def add_bounding_box(img, x1, y1, x2, y2):
    """Add bounding box to an image

    Args:
        img: image array
    """
    height, width = img.shape[:2]
    x1 = max(int(x1 * width), 0)
    y1 = max(int(y1 * height), 0)
    x2 = min(int(x2 * width), width - 1)
    y2 = min(int(y2 * height), height - 1)

    # add box to image
    img = cv2.rectangle(
        cv2.cvtColor(img.copy(), cv2.COLOR_GRAY2BGR),
        (x1, y1),  # coordinates of the starting point
        (x2, y2),  # coordinates of the ending point
        color=(0, 0, 255),
        thickness=5,
    )
    return img


def save_dataset(dataset, output_folder, is_test=False):
    create_folder(output_folder)
    for i in tqdm(range(len(dataset))):
        image, row = dataset.__getitem__(i, True)
        image = add_bounding_box(image, row['x1'], row['y1'], row['x2'], row['y2'])
        video_id = row.get('video_id').split('.')[0]
        time = row.get('time')

        file_id = f"{video_id}_{time}.png"
        if is_test:
            original_image = cv2.imread(os.path.join(ORIGINAL_DATA_TEST, file_id))
        else:
            original_image = cv2.imread(os.path.join(ORIGINAL_DATA_TRAIN, file_id))
        double_image = np.hstack((original_image, image))  # stack 2 images

        output_filename = os.path.join(output_folder, file_id)
        cv2.imwrite(output_filename, double_image)


def run(args):
    output_train_folder = os.path.join(args.output, "train_dataset_vis")
    output_valid_folder = os.path.join(args.output, "valid_dataset_vis")
    output_test_folder = os.path.join(args.output, "test_dataset_vis")

    save_dataset(train_dataset, output_train_folder)
    save_dataset(valid_dataset, output_valid_folder)
    save_dataset(test_dataset, output_test_folder, is_test=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--output', '-o',
                        required=True,
                        help='path where to save frames')

    arguments = parser.parse_args()
    print(arguments)
    run(arguments)
