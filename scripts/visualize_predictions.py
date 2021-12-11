import os

import argparse
import cv2
import pandas as pd
from tqdm import tqdm

from modules.utils import create_folder


ORIGINAL_DATA_TEST = "/workdir/data/images/test"


def run(args):
    df = pd.read_csv(args.file_csv)
    create_folder(args.output)
    for _, row in tqdm(df.iterrows()):
        video_id = row['video_id'].split('.')[0]
        fname = f"{video_id}_{str(row['time'])}.png"
        try:
            image = cv2.imread(os.path.join(ORIGINAL_DATA_TEST, fname))
            image = cv2.putText(
                img=image,
                text=str(row['distance']),
                org=(100, 100),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=2,
                color=(0, 0, 255),
                thickness=2,
                lineType=cv2.LINE_AA
            )
            image = cv2.resize(image, (640, 360))
            cv2.imwrite(os.path.join(args.output, fname), image)
        except Exception as exc:
            print(exc, fname)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--output', '-o',
                        required=True,
                        help='path where to save frames')

    parser.add_argument('--file_csv', '-f',
                        required=True,
                        help="csv file with predictions")

    arguments = parser.parse_args()
    print(arguments)
    run(arguments)
