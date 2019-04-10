import numpy as np

from matplotlib import pyplot as plt
from torch.utils import data

from utils.uciutils import load_bsds300


class BSDS300Dataset(data.Dataset):
    def __init__(self, split='train', frac=None):
        splits = dict(zip(
            ('train', 'val', 'test'),
            load_bsds300()
        ))
        self.data = np.array(splits[split]).astype(np.float32)
        self.n, self.dim = self.data.shape
        if frac is not None:
            self.n = int(frac * self.n)

    def __getitem__(self, item):
        return self.data[item]

    def __len__(self):
        return self.n


def test():
    dataset = BSDS300Dataset(split='train')
    print(type(dataset.data))
    print(dataset.data.shape)
    print(dataset.data.min(), dataset.data.max())
    fig, axs = plt.subplots(8, 8, figsize=(10, 10), sharex=True, sharey=True)
    axs = axs.reshape(-1)
    for i, dimension in enumerate(dataset.data.T):
        axs[i].hist(dimension, bins=100)
    # plt.hist(dataset.data.reshape(-1), bins=250)
    plt.tight_layout()
    plt.show()
    print(len(dataset))
    loader = data.DataLoader(dataset, batch_size=128, drop_last=True)
    print(len(loader))


if __name__ == '__main__':
    test()
