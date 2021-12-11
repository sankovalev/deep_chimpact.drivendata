# type: ignore

from torch.nn import L1Loss


experiment = dict(
    name="n09_e08",
    fold=4,
    path="/workdir/data/experiments/",
)

height, width = 224, 224

batch_size = 32  # per one GPU node (only during training)

num_workers = 8  # per one process

cyclic_lr = False  # use CyclicLR instead of ReduceLROnPlateau

model_params = {
    'nn_module': (
        'timm_model_with_meta', dict(
            model_name='tf_efficientnetv2_m_in21ft1k',
            pretrained=True,
            in_chans=4,
            num_classes=1,
        )
    ),
    'optimizer': {
        'lr': 0.0003
    },
    'loss': {
        'first': L1Loss(),
        'second': L1Loss(),
        'first_weight': 1.0,
        'second_weight': 0.0
    },
    'device': 'cuda:0',
    'amp': True,
    'iter_size': 2
}

# pass this parameter to DDP, use False by default
find_unused_parameters = False

debug = False
