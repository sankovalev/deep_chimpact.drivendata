# type: ignore
"""Make submit as a combination of several submits."""

import argparse
import os

import numpy as np
import pandas as pd


PREDS_PATH = "/workdir/data/predictions/"
SUBMISSION_EXAMPLE = "/workdir/data/submission_format.csv"


def run(args):
    example_df = pd.read_csv(SUBMISSION_EXAMPLE)
    distances = []
    for sub_name in args.subs:
        file_path = os.path.join(PREDS_PATH, f"{sub_name}.csv")
        df = pd.read_csv(file_path)

        assert df.video_id.equals(example_df.video_id)
        assert df.time.equals(example_df.time)

        distances.append(df.distance.values)

    array = np.array(distances)
    res_df = example_df.copy()

    if args.coefs != []:
        weights = [float(coef) for coef in args.coefs]
        print("Average with weights")
        res_df['distance'] = np.average(array, axis=0, weights=weights)
    else:
        res_df['distance'] = array.max(axis=0)

    file_path = os.path.join(PREDS_PATH, f"{args.name}.csv")
    res_df.to_csv(file_path, index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--subs', '-s',
                        nargs='+',
                        required=True,
                        help="names of sumbits")

    parser.add_argument('--name',
                        type=str,
                        required=True,
                        help="name of new submit")

    parser.add_argument(
        '--coefs',
        nargs='+',
        default=[],
        help="merge these files with original meta"
    )

    arguments = parser.parse_args()
    print(arguments)
    run(arguments)
