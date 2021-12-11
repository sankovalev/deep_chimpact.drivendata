from argus.callbacks import Callback
from tqdm import tqdm


class TqdmCallback(Callback):
    iterator = None

    def __init__(self, disable=False) -> None:
        super().__init__()
        self.disable = disable

    def _format_logs(self, logs):
        str_logs = ['{} - {:.4}'.format(k, v) for k, v in logs.items()]
        s = ', '.join(str_logs)
        return s

    def epoch_start(self, state):
        self.iterator = tqdm(
            iterable=state.data_loader,
            desc=state.phase,
            disable=self.disable,
            leave=False
        )

    def iteration_start(self, state):
        self.iterator.update()

    def iteration_complete(self, state):
        self.iterator.set_postfix_str(self._format_logs(state.metrics))

    def epoch_complete(self, state):
        self.iterator.close()
        self.iterator = None
