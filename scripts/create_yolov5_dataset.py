# type: ignore
"""
Prepare images and annotations for YoloV5.

NOTE:
- if image in black list copy it to yolov5 TEST(!) folder
- if image has valid bbox annotations copy it to TRAIN(!) folder
- copy images and labels to VALID folder manually without script
"""

import argparse
import json
import os
from shutil import copyfile

import numpy as np
import pandas as pd
from tqdm import tqdm


DEST_TRAIN_IMAGES_PATH = "/workdir/src/yolov5/datasets/deep_chimpact/images/train"
DEST_TRAIN_LABELS_PATH = "/workdir/src/yolov5/datasets/deep_chimpact/labels/train"

DEST_TEST_IMAGES_PATH = "/workdir/src/yolov5/datasets/deep_chimpact/images/test"


def main(args) -> None:
    images_source = args.images
    df_meta = pd.read_csv(args.meta)

    with open(args.black, 'r') as f:
        black_list = json.load(f)

    for _, row in tqdm(df_meta.iterrows()):
        video_id = row.get('video_id').split('.')[0]
        time = row.get('time')
        image_name = f"{video_id}_{time}.png"
        src_path = os.path.join(images_source, image_name)

        if not os.path.exists(src_path):
            continue

        if image_name in black_list or args.test:
            dst_path = os.path.join(DEST_TEST_IMAGES_PATH, image_name)
        else:
            x1, y1, x2, y2 = row.get('x1'), row.get('y1'), row.get('x2'), row.get('y2')
            annot = f"0 {str((x1 + x2) / 2.0)} {str((y1 + y2) / 2.0)} {str(x2 - x1)} {str(y2 - y1)}"
            dst_path = os.path.join(DEST_TRAIN_IMAGES_PATH, image_name)

            dst_lbl_path = os.path.join(DEST_TRAIN_LABELS_PATH, image_name.replace(".png", ".txt"))
            with open(dst_lbl_path, "w") as f:
                f.write(annot)

        copyfile(src_path, dst_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--images',
                        required=True,
                        help='path to folder with images')

    parser.add_argument('--meta',
                        required=True,
                        help='path to csv-file with meta')

    parser.add_argument('--black',
                        required=True,
                        help='path to black list')

    parser.add_argument('-test',
                        action='store_true',
                        help='save all to test folder')

    arguments = parser.parse_args()
    print(arguments)
    main(arguments)
