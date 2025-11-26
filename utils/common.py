import models
import torch
import os
import numpy as np
from torch.utils.data.sampler import Sampler
import itertools
import random


def generate_model(option, ema=False):
    model = getattr(models, option.model)(option.num_class)

    model.cuda()
    if ema:
        for param in model.parameters():
            param.detach_()
    return model


class TwoStreamBatchSampler(Sampler):

    def __init__(self, total_count, primary_count, primary_batch_size, secondary_batch_size, shuffle=False):
        super().__init__(data_source=None)

        self.indices = list(range(total_count))
        if shuffle:
            random.shuffle(self.indices)

        self.primary_indices = self.indices[:primary_count]
        self.secondary_indices = self.indices[primary_count:]

        self.primary_batch_size = primary_batch_size
        self.secondary_batch_size = secondary_batch_size

        assert len(self.primary_indices) >= self.primary_batch_size > 0
        assert len(self.secondary_indices) >= self.secondary_batch_size > 0


    def __iter__(self):
        primary_iter = iterate_once(self.primary_indices)
        secondary_iter = iterate_eternally(self.secondary_indices)
        return (
            primary_batch + secondary_batch
            for (primary_batch, secondary_batch) in zip(
            grouper(primary_iter, self.primary_batch_size),
            grouper(secondary_iter, self.secondary_batch_size))
        )

    def __len__(self):
        return len(self.primary_indices) // self.primary_batch_size


def iterate_once(iterable):
    return np.random.permutation(iterable)


def iterate_eternally(indices):
    def infinite_shuffles():
        while True:
            yield np.random.permutation(indices)

    return itertools.chain.from_iterable(infinite_shuffles())


def grouper(iterable, n):
    """
    Collect data into fixed-length chunks or blocks
    """
    # grouper('ABCDEFG', 3) --> ABC DEF"
    args = [iter(iterable)] * n
    return zip(*args)


def set_seed(inc, base_seed=2023):

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = True

    seed = base_seed + inc
    random.seed(seed)
    np.random.seed(seed + 1)
    torch.manual_seed(seed + 2)
    torch.cuda.manual_seed_all(seed + 3)
    os.environ['PYTHONHASHSEED'] = str(seed + 4)

