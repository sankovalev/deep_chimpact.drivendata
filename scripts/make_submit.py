# type: ignore
"""Make final submit with valid format and heuristics"""

import argparse
from distutils.command import config
import os
import sys

import numpy as np
import pandas as pd
from tqdm import tqdm

from modules.utils import create_folder


sys.path.append("/workdir/src/")

PREDS_PATH = "/workdir/data/predictions/"
SUBMISSION_EXAMPLE = "/workdir/data/submission_format.csv"
TEST_META_PATH = "/workdir/data/meta/test_metadata_v4.csv"
VALUE_FOR_FILLNA = 7.5  # train median
TRAIN_MEDIANS = dict(moyen_bafing=12.0, tai=6.5)
MIN_POSSIBLE_DISTANCE = 1.0
MAX_POSSIBLE_DISTANCE = 25.0


def run(args):
    preds_file = os.path.join(PREDS_PATH, f"raw_{args.name}.csv")
    preds_df = pd.read_csv(preds_file)
    test_df = pd.read_csv(TEST_META_PATH)
    example_df = pd.read_csv(SUBMISSION_EXAMPLE)

    final_records = []

    for _, row in tqdm(example_df.iterrows(), total=len(example_df)):
        video_id, time = row.get('video_id'), row.get('time')

        # 1. Try to find this prediction:
        cur_df = preds_df.loc[(preds_df.video_id == video_id) & (preds_df.time == time)]
        if len(cur_df) == 1:
            final_records.append(dict(**cur_df.iloc[0]))
            continue

        # 2. Try to find this video and nearest times:
        cur_df = preds_df.loc[
            (preds_df.video_id == video_id) & \
            (time - args.timew < preds_df.time) & \
            (preds_df.time < (time + args.timew))
        ]
        if len(cur_df) > 0:
            final_records.append({
                "video_id": video_id,
                "time": time,
                "distance": cur_df.distance.mean()
            })
            continue

        # 3. Try to find this video:
        cur_df = preds_df.loc[preds_df.video_id == video_id]
        if len(cur_df) > 0:
            final_records.append({
                "video_id": video_id,
                "time": time,
                "distance": cur_df.distance.median()
            })
            continue

        # 4. Set median by train
        row_test = test_df.loc[(test_df.video_id == video_id) & (test_df.time == time)].iloc[0]
        final_records.append({
            "video_id": video_id,
            "time": time,
            "distance": TRAIN_MEDIANS[row_test.get('park')]
        })

    assert len(final_records) == len(example_df), f"{len(final_records)}; {len(example_df)}"

    final_df = pd.DataFrame.from_records(final_records)

    # Replace too small and too large distances:
    final_df.loc[final_df.distance < MIN_POSSIBLE_DISTANCE, 'distance'] = MIN_POSSIBLE_DISTANCE
    final_df.loc[final_df.distance > MAX_POSSIBLE_DISTANCE, 'distance'] = MAX_POSSIBLE_DISTANCE

    # Fill Nan values
    df_na = final_df.loc[final_df.distance.isna()]
    print(f"There are {len(df_na)} NaN distances in final dataframe")
    final_df.distance = final_df.groupby("video_id").transform(lambda x: x.fillna(x.median()))['distance']
    final_df.fillna(VALUE_FOR_FILLNA, inplace=True)
    assert final_df.distance.isna().sum() == 0

    final_df.distance = np.round(final_df.distance * 2) / 2

    filename = os.path.join(PREDS_PATH, f"{args.name}.csv")
    final_df.to_csv(filename, index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--name',
                        type=str,
                        required=True,
                        help="name of experiment")

    parser.add_argument('--timew',
                        type=int,
                        default=10,
                        help="time window")

    arguments = parser.parse_args()
    print(arguments)
    run(arguments)
