import numpy as np

from .bsds300 import BSDS300Dataset
from .gas import GasDataset
from .hepmass import HEPMASSDataset
from .miniboone import MiniBooNEDataset
from .plane import TwoSpiralsDataset, GaussianGridDataset, CheckerboardDataset
from .power import PowerDataset

from torch.utils import data


uci_datasets = {
    'power': PowerDataset,
    'gas': GasDataset,
    'hepmass': HEPMASSDataset,
    'miniboone': MiniBooNEDataset,
    'bsds300': BSDS300Dataset
}


def load_uci_dataset(name, split, frac=None):
    return uci_datasets[name](split, frac)


def get_uci_dataset_range(dataset_name):
    train_dataset = load_uci_dataset(dataset_name, split='train')
    val_dataset = load_uci_dataset(dataset_name, split='val')
    test_dataset = load_uci_dataset(dataset_name, split='test')
    train_min, train_max = np.min(train_dataset.data, axis=0), np.max(train_dataset.data, axis=0)
    val_min, val_max = np.min(val_dataset.data, axis=0), np.max(val_dataset.data, axis=0)
    test_min, test_max = np.min(test_dataset.data, axis=0), np.max(test_dataset.data, axis=0)
    min_ = np.minimum(train_min, np.minimum(val_min, test_min))
    max_ = np.maximum(train_max, np.maximum(val_max, test_max))
    return np.array((min_, max_))


plane_datasets = {
    'spirals': TwoSpiralsDataset,
    'diamond': GaussianGridDataset,
    'checkerboard': CheckerboardDataset
}


def load_plane_dataset(name, n):
    return plane_datasets[name](n)


class InfiniteLoader:
    """A data loader that can load a dataset repeatedly."""

    def __init__(self, dataset, batch_size=1, shuffle=True,
                 drop_last=True, num_epochs=None):
        """Constructor.
        Args:
            dataset: A `Dataset` object to be loaded.
            batch_size: int, the size of each batch.
            shuffle: bool, whether to shuffle the dataset after each epoch.
            drop_last: bool, whether to drop last batch if its size is less than
                `batch_size`.
            num_epochs: int or None, number of epochs to iterate over the dataset.
                If None, defaults to infinity.
        """
        self.loader = data.DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=drop_last
        )
        self.finite_iterable = iter(self.loader)
        self.counter = 0
        self.num_epochs = float('inf') if num_epochs is None else num_epochs

    def __next__(self):
        try:
            return next(self.finite_iterable)
        except StopIteration:
            self.counter += 1
            if self.counter >= self.num_epochs:
                raise StopIteration
            self.finite_iterable = iter(self.loader)
            return next(self.finite_iterable)

    def __iter__(self):
        return self


