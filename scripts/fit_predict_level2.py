# type: ignore
"""Fit predict on level 2."""

import argparse
import os

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score, cross_val_predict, GroupKFold
from sklearn.metrics import mean_absolute_error
import pandas as pd


not_actual_columns = ['video_id', 'time']
group_column = "video_id"

PREDS_PATH = "/workdir/data/predictions/"


def run(args):
    train_df = pd.read_csv(args.train_file)
    test_df = pd.read_csv(args.test_file)

    assert list(train_df.columns) == list(test_df.columns)

    train_labels = pd.read_csv(args.train_labels)
    train_df = pd.merge(train_df, train_labels,  how='left', on=['video_id','time'])

    cv = GroupKFold(n_splits=7)
    groups = train_df[group_column]

    y_train = train_df.distance.values
    train_df.drop(columns=['distance'], inplace=True)
    train_df.drop(columns=not_actual_columns, inplace=True)
    X_train = train_df.values

    for col in train_df.columns:
        print("Apriori MAE:", mean_absolute_error(y_train, train_df[col]))

    reg = LinearRegression()
    scores = cross_val_score(
        estimator=reg,
        X=X_train,
        y=y_train,
        groups=groups,
        cv=cv,
        scoring='neg_mean_absolute_error'
    )
    print(f"Cross val scores: {np.abs(np.mean(scores))}")

    sub_df = test_df[['video_id', 'time']]
    test_df.drop(columns=not_actual_columns, inplace=True)
    X_test = test_df.values
    reg.fit(
        X=X_train,
        y=y_train,
    )
    print(f"Coefs: {reg.coef_}")
    predictions = reg.predict(
        X=X_test,
    )
    sub_df['distance'] = pd.Series(predictions).copy()
    file_path = os.path.join(PREDS_PATH, f"raw_{args.name}.csv")
    sub_df.to_csv(file_path, index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--train_file",
        required=True,
        help="csv with train info",
    )

    parser.add_argument(
        "--train_labels",
        default="/workdir/data/meta/train_labels.csv",
        help="file with train distances"
    )

    parser.add_argument(
        "--test_file",
        required=True,
        help="csv with test info",
    )

    parser.add_argument(
        '--name',
        type=str,
        required=True,
        help="name of new raw submission"
    )

    arguments = parser.parse_args()
    print(arguments)
    run(arguments)
