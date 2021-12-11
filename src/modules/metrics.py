# type: ignore

from argus.metrics import Metric
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error


class RegressionMetric(Metric):
    name: str = ""
    better: str = ""
    metric: callable
    y_gt_list = []
    y_pr_list = []

    def reset(self):
        self.y_gt_list = []
        self.y_pr_list = []

    def update(self, step_output: dict):
        y_pr = step_output['prediction'].float()
        y_gt = step_output['target'].float()
        self.y_pr_list.extend(y_pr.cpu().tolist())
        self.y_gt_list.extend(y_gt.cpu().tolist())

    def compute(self):
        score = self.metric(
            y_true=self.y_gt_list,
            y_pred=self.y_pr_list
        )
        return torch.as_tensor(score)


class MetricMAE(RegressionMetric):
    name = 'mae'
    better = 'min'

    def __init__(self) -> None:
        super().__init__()
        self.metric = mean_absolute_error


class MetricMSE(RegressionMetric):
    name = 'mse'
    better = 'min'

    def __init__(self) -> None:
        super().__init__()
        self.metric = mean_squared_error
