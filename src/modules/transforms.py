# type: ignore

import albumentations as albu
from albumentations.pytorch.transforms import ToTensorV2


def get_transforms(is_train, height, width):
    resize_and_padding_augs = [
        albu.Resize(height=height, width=width),
        albu.PadIfNeeded(min_height=None, min_width=None,
                         pad_height_divisor=32, pad_width_divisor=32),
    ]
    specific_augs = [
        albu.OneOf([
            albu.Flip(p=0.7),
            albu.Transpose(),
            albu.RandomRotate90(p=0.3),
            albu.NoOp()
        ]),
        albu.ElasticTransform(sigma=20, alpha_affine=20, p=0.5),
        albu.ShiftScaleRotate(p=0.25),
        albu.CoarseDropout(max_height=5, max_width=5),
    ]
    final_augs = [
        ToTensorV2(transpose_mask=True)
    ]

    if is_train:
        return albu.Compose(resize_and_padding_augs + specific_augs + final_augs)
    return albu.Compose(resize_and_padding_augs + final_augs)
