# type: ignore
"""Make level 2 dataset for training."""

import argparse
import os

import numpy as np
import pandas as pd


DATA_PATH = "/workdir/data/level2/"


def read_meta_and_create_features(fname):
    df = pd.read_csv(fname)
    df.dropna(inplace=True)
    df.drop(columns=['x1', 'y1', 'x2', 'y2', 'probability', 'park', 'site_id'], inplace=True)
    return df

def merge_them_all(meta_df, files):
    for i, filename in enumerate(files):
        df = pd.read_csv(filename)
        meta_df = pd.merge(meta_df, df,  how='left', on=['video_id','time'])
        meta_df.rename(columns={'nn_preds': f'nn_preds_{i}'}, inplace=True)
        meta_df.rename(columns={'distance': f'nn_preds_{i}'}, inplace=True)
    return meta_df

def create_train(args):
    meta_df = read_meta_and_create_features(args.meta_train)
    meta_df = merge_them_all(meta_df, args.train_files)
    meta_df.dropna(inplace=True)
    file_path = os.path.join(DATA_PATH, f"train_{args.name}.csv")
    meta_df.to_csv(file_path, index=False)

def create_test(args):
    meta_df = read_meta_and_create_features(args.meta_test)
    meta_df = merge_them_all(meta_df, args.test_files)
    meta_df.dropna(inplace=True)
    file_path = os.path.join(DATA_PATH, f"test_{args.name}.csv")
    meta_df.to_csv(file_path, index=False)

def run(args):
    create_train(args)
    create_test(args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--meta_train",
        required=True,
        help="csv with train meta",
    )

    parser.add_argument(
        '--meta_test',
        required=True,
        help="csv with test meta"
    )

    parser.add_argument(
        '--train_files',
        nargs='+',
        required=True,
        help="merge these files with original meta"
    )

    parser.add_argument(
        '--test_files',
        nargs='+',
        required=True,
        help="merge these raw submits with original meta"
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
