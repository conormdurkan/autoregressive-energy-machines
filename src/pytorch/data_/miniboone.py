import numpy as np
import os

from matplotlib import pyplot as plt
from torch.utils.data import Dataset

from utils import io
from utils.uciutils import preprocess_and_save_miniboone


class MiniBooNEDataset(Dataset):
    def __init__(self, split='train', frac=None):
        path = os.path.join(io.get_data_root(), 'data', 'miniboone/{}.npy'.format(split))
        try:
            self.data = np.load(path).astype(np.float32)
        except FileNotFoundError:
            print('Preprocessing and saving Miniboone...')
            preprocess_and_save_miniboone()
            print('Done!')
            self.data = np.load(path).astype(np.float32)
        self.n, self.dim = self.data.shape
        if frac is not None:
            self.n = int(frac * self.n)

    def __getitem__(self, item):
        return self.data[item]

    def __len__(self):
        return self.n


def test():
    dataset = MiniBooNEDataset(split='train')
    print(type(dataset.data))
    print(dataset.data.shape)
    print(dataset.data.min(), dataset.data.max())
    plt.hist(dataset.data.reshape(-1), bins=250)
    plt.show()


def main():
    test()


if __name__ == '__main__':
    main()
