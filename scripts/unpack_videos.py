import argparse
import os

import cv2
import pandas as pd
from tqdm import tqdm

from modules.utils import create_folder


def load_video(filepath):
    cap = cv2.VideoCapture(filepath)
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = list()
    for _ in range(num_frames):
        _, frame = cap.read()
        if frame is not None:
            frames.append(frame)
    return frames


def run(args):
    df = pd.read_csv(args.meta)
    create_folder(args.output)

    for video_id in tqdm(list(df.video_id.unique())):
        df_video = df.loc[df.video_id == video_id]
        filename = os.path.join(args.videos, video_id)

        try:
            frames_list = load_video(filename)
            for _, row in df_video.iterrows():
                idx = row.get('time')
                frame = frames_list[idx]

                video_name = video_id.split('.')[0]
                target_filename = os.path.join(args.output, f"{video_name}_{str(idx)}.png")
                cv2.imwrite(target_filename, frame)
        except Exception as exc:
            print(exc, video_id)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--videos',
                        required=True,
                        help='path to folder with videos')

    parser.add_argument('--meta',
                        required=True,
                        help='path to csv-file with meta')

    parser.add_argument('--output', '-o',
                        required=True,
                        help='path where to save frames')

    arguments = parser.parse_args()
    print(arguments)
    run(arguments)
