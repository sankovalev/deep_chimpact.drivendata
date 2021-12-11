# type: ignore
"""Train model with parameters from config"""

import argparse
import os
import sys

from argus import load_model
import torch
import torch.distributed as dist
from timm.utils.model_ema import ModelEmaV2 as ModelEma

from modules.factory import get_loaders, get_model, get_callbacks, get_metrics, get_nn_params


sys.path.append("/workdir/src/")


def run(args):
    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        dist.init_process_group(backend='nccl', init_method='env://')

    train_loader, valid_loader = get_loaders(args.distributed, args.local_rank)
    callbacks, val_callbacks = get_callbacks(args.distributed, args.local_rank)
    metrics = get_metrics()
    if args.model is not None:
        model_params = get_nn_params()
        model = load_model(
            file_path=args.model,
            optimizer=model_params['optimizer'],
            loss=model_params['loss'],
            device=model_params['device']
        )
    else:
        model = get_model(args.distributed, args.local_rank)

    if args.ema:
        model.ema_model = ModelEma(
            model=model.get_nn_module(),
            decay=0.995,
            device=model_params['device']
        )

    model.fit(
        train_loader=train_loader,
        val_loader=valid_loader,
        num_epochs=args.num_epochs,
        metrics=metrics,
        metrics_on_train=True,
        callbacks=callbacks,
        val_callbacks=val_callbacks,
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--num_epochs',
                        type=int,
                        default=500)

    parser.add_argument('--model', '-m',
                        type=str,
                        default=None,
                        help="path to saved model to continue from checkpoint")

    parser.add_argument('-ema',
                        action='store_true')

    parser.add_argument("--local_rank", default=0, type=int)

    arguments = parser.parse_args()
    arguments.distributed = False
    if 'WORLD_SIZE' in os.environ:
        arguments.distributed = int(os.environ['WORLD_SIZE']) > 1

    run(arguments)
