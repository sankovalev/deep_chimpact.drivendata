
import os

import torch
import ttach


def mean(input_x, dim):
    """Mean."""
    return torch.mean(input_x, dim=dim)


def gmean(input_x, dim):
    """Geometric mean."""
    log_x = torch.log(input_x)
    return torch.exp(torch.mean(log_x, dim=dim))


def hmean(input_x, dim):
    """Harmonic mean."""
    size = input_x.shape[dim]
    return size / torch.sum((1.0 / input_x), dim=dim)


def create_folder(folder, rm_existing=False):
    exists = os.path.exists(folder)
    if rm_existing and exists:
        os.rmdir(folder)
    os.makedirs(folder, exist_ok=True)


class RegressionTTAWrapper(ttach.ClassificationTTAWrapper):

    def forward(self, data: dict, *args):
        merger = ttach.base.Merger(type=self.merge_mode, n=len(self.transforms))
        tensor = data['tensor'].clone()

        for transformer in self.transforms:
            data['tensor'] = transformer.augment_image(tensor)
            augmented_output = self.model(data, *args)
            if self.output_key is not None:
                augmented_output = augmented_output[self.output_key]
            deaugmented_output = transformer.deaugment_label(augmented_output)
            merger.append(deaugmented_output)

        result = merger.result
        if self.output_key is not None:
            result = {self.output_key: result}

        return result
