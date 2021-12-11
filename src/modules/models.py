import timm
import torch


class ModelWithMeta(torch.nn.Module):

    def __init__(self, **kwargs) -> None:
        super().__init__()
        self.backbone = timm.create_model(**kwargs)  # make regression, predict one number
        self.fc = torch.nn.Linear(
            in_features=3,
            out_features=1,
            bias=False
        )  # add aux features and predict one number again

    def forward(self, data):
        pred = self.backbone(data['tensor'])  # Nx1
        x = torch.cat((pred, data['width'], data['height']), dim=1).float()
        x = self.fc(x)
        return x
