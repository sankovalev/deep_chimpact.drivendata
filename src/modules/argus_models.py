# type: ignore

from argus import Model, load_model
from argus.utils import deep_chunk, deep_to, deep_detach
from pytorch_toolbelt import losses as L
import timm
import torch
import ttach

from modules.utils import gmean, RegressionTTAWrapper
from modules.models import ModelWithMeta


class ChimpactModel(Model):
    nn_module = {
        'timm_model': timm.create_model,
        'timm_model_with_meta': ModelWithMeta
    }
    optimizer = torch.optim.AdamW
    loss = L.JointLoss
    prediction_transform = torch.nn.Identity
    ema_model = None

    def __init__(self, params):
        super().__init__(params)
        self.iter_size = params.get('iter_size', 1)
        self.amp = params.get('amp', False)
        self.grad_scaler = torch.cuda.amp.GradScaler(enabled=self.amp)

    def train_step(self, batch, state) -> dict:
        self.train()
        self.optimizer.zero_grad()

        # Gradient accumulation
        for chunk_batch in deep_chunk(batch, self.iter_size):
            input_batch, target = deep_to(chunk_batch, self.device, non_blocking=True)
            with torch.cuda.amp.autocast(enabled=self.amp):
                prediction = self.nn_module(input_batch)
                loss = self.loss(prediction.float(), target.float())
                loss = loss / self.iter_size

            self.grad_scaler.scale(loss).backward()

        self.grad_scaler.step(self.optimizer)
        self.grad_scaler.update()

        if self.ema_model is not None:
            self.ema_model.update(self.nn_module)

        prediction = deep_detach(prediction)
        target = deep_detach(target)
        prediction = self.prediction_transform(prediction)
        return {
            'prediction': prediction,
            'target': target,
            'loss': loss.item()
        }

    def val_step(self, batch, state) -> dict:
        self.eval()
        with torch.no_grad():
            input_batch, target = deep_to(batch, device=self.device, non_blocking=True)

            if self.ema_model is not None:
                prediction = self.ema_model.module(input_batch)
            else:
                prediction = self.nn_module(input_batch)

            loss = self.loss(prediction, target)
            prediction = self.prediction_transform(prediction)
            return {
                'prediction': prediction,
                'target': target,
                'loss': loss.item()
            }


class ChimpactEnsembleModel(Model):
    argus_models = []

    def __init__(self, argus_checkpoints, device='cuda', avg_func=gmean, use_tta=False) -> None:
        self.device = device
        for chkpt in argus_checkpoints:
            model = load_model(chkpt, device=self.device)
            if use_tta:
                model.nn_module = RegressionTTAWrapper(
                    model=model.get_nn_module(),
                    transforms=ttach.aliases.d4_transform(),
                    merge_mode='gmean'  # TODO: try tsharpen
                )
            self.argus_models.append(model)
        self.avg_func = avg_func
        self.logger = self.build_logger()

    def train_ready(self) -> bool:
        return True

    def predict_ready(self) -> bool:
        return True

    def eval(self):
        """Set the nn_module into eval mode."""
        for model in self.argus_models:
            model.eval()

    def val_step(self, batch, state) -> dict:
        """Perform a single validation step for ensemble.
        """
        self._check_predict_ready()
        self.eval()
        with torch.no_grad():
            input, target = deep_to(batch, device=self.device, non_blocking=True)

            ensemble_losses = list()
            ensemble_predictions = list()
            for model in self.argus_models:
                prediction = model.nn_module(input)
                ensemble_losses.append(model.loss(prediction, target))
                ensemble_predictions.append(model.prediction_transform(prediction))

            pred = torch.stack(ensemble_predictions)
            loss = torch.stack(ensemble_losses)

            ens_prediction = self.avg_func(pred, dim=0)
            ens_loss = torch.mean(loss)
            return {
                'prediction': ens_prediction,
                'target': target,
                'loss': ens_loss.item()
            }

    def predict(self, input):
        """Make a prediction with the given input."""
        self._check_predict_ready()
        self.eval()
        with torch.no_grad():
            input = deep_to(input, self.device)

            ensemble_predictions = list()
            for model in self.argus_models:
                prediction = model.nn_module(input)
                ensemble_predictions.append(model.prediction_transform(prediction))

            pred = torch.stack(ensemble_predictions)
            return self.avg_func(pred, dim=0)
