import torch
import torch.nn as nn


class AbsWeighting:
    r"""An abstract class for weighting strategies.
    """
    def __init__(self, trainer, train_batch, writer):
        self.device = trainer.device
        self.writer = writer
        self.task_num = trainer.task_num
        self.epochs = trainer.epochs
        self.train_batch = train_batch