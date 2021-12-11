# type: ignore
"""Make 1 file from folds."""

import argparse
import os

import pandas as pd


DATA_PATH = "/workdir/data/level2/"


def run(args):
    folds = []
    for filename in args.folds:
        df = pd.read_csv(filename)
        df.rename(columns={'distance': 'nn_preds'}, inplace=True)
        folds.append(df)
    df_folds = pd.concat(folds)
    file_path = os.path.join(DATA_PATH, f"{args.name}.csv")
    df_folds.to_csv(file_path, index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--folds', '-f',
        nargs='+',
        required=True,
        help="merge these folds into one file"
    )

    parser.add_argument(
        '--name',
        type=str,
        required=True,
        help="name of new train dataset"
    )

    arguments = parser.parse_args()
    print(arguments)
    run(arguments)
