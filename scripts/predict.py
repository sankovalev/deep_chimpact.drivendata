# type: ignore
"""Make ensemble prediction on test part"""

import argparse
import os
import sys

import pandas as pd
from tqdm import tqdm

from modules.factory import get_test_loader
from modules.utils import create_folder
from modules.argus_models import ChimpactEnsembleModel


sys.path.append("/workdir/src/")

PREDS_PATH = "/workdir/data/predictions/"

create_folder(PREDS_PATH)


def run(args):
    test_loader = get_test_loader()
    ensemble = ChimpactEnsembleModel(
        argus_checkpoints=args.models,
        device="cuda:0",
        use_tta=args.tta
    )
    results = []
    for batch, video_ids, times in tqdm(test_loader):
        preds = ensemble.predict(batch)
        for i, (video_id, time) in enumerate(zip(video_ids, times)):
            results.append(
                {
                    "video_id": video_id,
                    "time": int(time),
                    "distance": float(preds[i])
                }
            )

    file_path = os.path.join(PREDS_PATH, f"raw_{args.name}.csv")
    res_df = pd.DataFrame.from_records(results)
    res_df.to_csv(file_path, index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--models', '-m',
                        nargs='+',
                        required=True,
                        help="paths to saved models")

    parser.add_argument('--name',
                        type=str,
                        required=True,
                        help="name of experiment")

    parser.add_argument('-tta',
                        action='store_true')

    arguments = parser.parse_args()
    print(arguments)
    run(arguments)
