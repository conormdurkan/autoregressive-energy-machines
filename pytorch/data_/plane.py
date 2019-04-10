import numpy as np

from matplotlib import cm, pyplot as plt
from torch.utils.data import Dataset

from utils import plane as sharedplane


class PlaneDataset(Dataset):
    def __init__(self, n):
        self.n = n
        self.data = None
        self.create_data()

    def __getitem__(self, item):
        return self.data[item]

    def __len__(self):
        return self.n

    def reset(self):
        self.create_data()

    def create_data(self):
        raise NotImplementedError


class GaussianGridDataset(PlaneDataset):
    def __init__(self, n, width=15):
        self.width = width
        super().__init__(n)

    def create_data(self, rotate=True):
        self.data = sharedplane.create_gaussian_grid_data(self.n, self.width, rotate)


class TwoSpiralsDataset(PlaneDataset):
    def __init__(self, n):
        super().__init__(n)

    def create_data(self):
        self.data = sharedplane.create_two_spirals_data(self.n)


class TestGridDataset(PlaneDataset):
    def __init__(self, n_points_per_axis, bounds):
        self.bounds = bounds
        self.n_points_per_axis = n_points_per_axis
        self.X = None
        self.Y = None
        super().__init__(n_points_per_axis ** 2)

    def create_data(self):
        x = np.linspace(self.bounds[0, 0], self.bounds[0, 1], self.n_points_per_axis)
        y = np.linspace(self.bounds[1, 0], self.bounds[1, 1], self.n_points_per_axis)
        self.X, self.Y = np.meshgrid(x, y)
        self.data = np.vstack([self.X.flatten(), self.Y.flatten()]).T.astype(np.float32)


class CheckerboardDataset(PlaneDataset):
    def __init__(self, n):
        super().__init__(n)

    def create_data(self):
        x1 = np.random.rand(self.n) * 8 - 4
        x2_ = np.random.rand(self.n) + np.random.randint(-2, 2, self.n) * 2
        x2 = x2_ + (np.floor(x1) % 2)
        self.data = (np.concatenate([x1[:, None], x2[:, None]], 1)).astype(np.float32)


class FaceDataset(PlaneDataset):
    def __init__(self, n, face='einstein'):
        self.face = face
        self.image = None
        super().__init__(n)

    def create_data(self):
        self.data = sharedplane.create_einstein_data(self.n, self.face)


def test():
    n = int(1e6)
    dataset = GaussianGridDataset(n)
    samples = dataset.data

    fig, ax = plt.subplots(1, 1, figsize=(5, 5))

    # ax.hist2d(samples[:, 0], samples[:, 1],
    #               range=[[0, 1], [0, 1]], bins=512, cmap=cm.viridis)
    ax.hist2d(samples[:, 0], samples[:, 1], range=[[-4, 4], [-4, 4]], bins=512,
              cmap=cm.viridis)

    ax.set_xticks([])
    ax.set_yticks([])

    plt.show()
    # path = os.path.join(utils.get_output_root(), 'plane-test.png')
    # plt.savefig(path, rasterized=True)


if __name__ == '__main__':
    test()
