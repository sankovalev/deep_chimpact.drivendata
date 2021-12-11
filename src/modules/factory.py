# type: ignore

import os

from argus.callbacks import MonitorCheckpoint, EarlyStopping, ReduceLROnPlateau, \
                            LoggingToCSV, on_epoch_complete, CyclicLR
from torch.utils.data import DataLoader, DistributedSampler
import torch.distributed as dist
from torch.nn import SyncBatchNorm
from torch.nn.parallel import DistributedDataParallel

import config
from modules.argus_models import ChimpactModel
from modules.callbacks import TqdmCallback
from modules.datasets import TrainAndValidDataset, TestDataset, ValidForStackingDataset
from modules.metrics import MetricMAE, MetricMSE
from modules.transforms import get_transforms


def get_callbacks(distributed, local_rank):
    exp_name = f"{config.experiment['name']}-fold{config.experiment['fold']}"
    callbacks = []
    if local_rank == 0:
        callbacks += [
            MonitorCheckpoint(
                dir_path=os.path.join(
                    config.experiment['path'],
                    'saved_models',
                    exp_name),
                max_saves=1, optimizer_state=True, monitor='val_loss', better='min'
            ),
            LoggingToCSV(
                file_path=os.path.join(
                    config.experiment['path'],
                    'logs',
                    f"{exp_name}.csv")
            ),
            TqdmCallback()
        ]
    callbacks += [
        EarlyStopping(monitor='val_loss', better='min', patience=25),
    ]
    if config.cyclic_lr:
        callbacks += [
            CyclicLR(
                base_lr=1e-7,
                max_lr=config.model_params['optimizer']['lr'],
                mode="triangular2",
                cycle_momentum=False
            )
        ]
    else:
        callbacks += [
            ReduceLROnPlateau(monitor='val_loss', better='min', patience=10, verbose=True)
        ]
    if distributed:
        @on_epoch_complete
        def schedule_sampler(state):
            state.data_loader.sampler.set_epoch(state.epoch + 1)
        callbacks += [schedule_sampler]

    val_callbacks = []
    return callbacks, val_callbacks


def get_metrics():
    return [
        MetricMAE(),
        MetricMSE()
    ]


def get_nn_params():
    return config.model_params


def get_model(distributed, local_rank):
    world_size = 1
    world_batch_size = config.batch_size
    if distributed:
        world_size = dist.get_world_size()
        world_batch_size = config.batch_size * world_size

    model_params = config.model_params
    model_params['optimizer']['lr'] = (
        model_params['optimizer']['lr'] * (world_batch_size / config.batch_size))
    model = ChimpactModel(model_params)
    if distributed:
        model.nn_module = SyncBatchNorm.convert_sync_batchnorm(model.nn_module)
        model.nn_module = DistributedDataParallel(
            model.nn_module.to(local_rank),
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=config.find_unused_parameters
        )
        if local_rank:
            model.logger.disabled = True
    return model


train_dataset = TrainAndValidDataset(
    fold_idx=config.experiment['fold'],
    transforms=get_transforms(True, config.height, config.width),
    debug=config.debug,
    valid=False,
)

valid_dataset = TrainAndValidDataset(
    fold_idx=config.experiment['fold'],
    transforms=get_transforms(False, config.height, config.width),
    debug=config.debug,
    valid=True,
)

test_dataset = TestDataset(
    transforms=get_transforms(False, config.height, config.height)
)


def get_loaders(distributed, local_rank, **kwargs):
    train_sampler = None
    valid_sampler = None
    if distributed:
        train_sampler = DistributedSampler(
            train_dataset,
            num_replicas=dist.get_world_size(),
            rank=local_rank,
            shuffle=True
        )
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=kwargs.get('bs', config.batch_size),
        drop_last=True,
        sampler=train_sampler,
        shuffle=train_sampler is None,
        num_workers=config.num_workers,
        pin_memory=True,
    )
    valid_loader = DataLoader(
        dataset=valid_dataset,
        batch_size=kwargs.get('bs', 1),
        sampler=valid_sampler,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True
    )
    return train_loader, valid_loader


def get_test_loader():
    return DataLoader(
        dataset=test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True
    )


def get_valid_loader_by_fold(fold):
    dataset = ValidForStackingDataset(
        transforms=get_transforms(False, config.height, config.width),
        fold_idx=fold,
    )
    return DataLoader(
        dataset=dataset,
        batch_size=1,
        sampler=None,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True
    )
