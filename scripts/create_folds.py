import os

import argparse
from sklearn.model_selection import GroupKFold
import pandas as pd

from modules.utils import create_folder


meta_columns = ["video_id", "time", "x1", "y1", "x2", "y2", "probability", "park", "site_id"]
target_column = "distance"
group_column = "video_id"


def run(args):
    df_labels = pd.read_csv(args.labels_file)
    df_meta = pd.read_csv(args.meta_file)
    assert list(df_labels.video_id) == list(df_meta.video_id)
    df = df_meta.copy()
    df[target_column] = df_labels.distance.copy()
    assert df.shape == (15229, 10)

    train_output_folder = os.path.join(args.output_folder, "train")
    valid_output_folder = os.path.join(args.output_folder, "valid")
    create_folder(train_output_folder)
    create_folder(valid_output_folder)

    cv = GroupKFold(n_splits=args.folds)
    X = df[meta_columns]
    y = df[target_column]
    groups = df[group_column]

    for idx, (train_idxs, valid_idxs) in enumerate(cv.split(X, y, groups=groups)):
        df_train = df.iloc[train_idxs]
        df_valid = df.iloc[valid_idxs]

        train_output_path = os.path.join(train_output_folder, f"fold{idx}.csv")
        valid_output_path = os.path.join(valid_output_folder, f"fold{idx}.csv")

        df_train.to_csv(train_output_path, index=False)
        df_valid.to_csv(valid_output_path, index=False)

        print(f"Succesfully created fold{idx}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-l", "--labels_file",
        required=True,
        help="csv with train labels"
    )

    parser.add_argument(
        "-m", "--meta_file",
        required=True,
        help="csv with meta",
    )

    parser.add_argument(
        "-o",  "--output_folder",
        required=True,
        help="where to save files",
    )

    parser.add_argument(
        "--folds",
        default=5,
        type=int,
        help="number of folds",
    )

    arguments = parser.parse_args()
    print(arguments)
    run(arguments)
