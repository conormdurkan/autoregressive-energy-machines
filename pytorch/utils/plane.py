import numpy as np
import os

from skimage import color, io as imageio, transform

from utils import io


def create_gaussian_grid_data(n, width, rotate=True):
    bound = -2.5
    means = np.array([
        (x + 1e-3 * np.random.rand(), y + 1e-3 * np.random.rand())
        for x in np.linspace(-bound, bound, width)
        for y in np.linspace(-bound, bound, width)
    ])

    covariance_factor = 0.06 * np.eye(2)

    index = np.random.choice(range(width ** 2), size=n, replace=True)
    noise = np.random.randn(n, 2)
    data = means[index] + noise @ covariance_factor
    if rotate:
        rotation_matrix = np.array([
            [1 / np.sqrt(2), -1 / np.sqrt(2)],
            [1 / np.sqrt(2), 1 / np.sqrt(2)]
        ])
        data = data @ rotation_matrix
    data = data.astype(np.float32)
    return data


def create_two_spirals_data(n):
    """
    Modified from https://github.com/rtqichen/ffjord/blob/master/lib/toy_data.py
    """
    m = np.sqrt(np.random.rand(n // 2, 1)) * 900 * (2 * np.pi) / 360
    a = 0.7
    d1x = -a * np.cos(m) * m + np.random.rand(n // 2, 1) * a / 2
    d1y = a * np.sin(m) * m + np.random.rand(n // 2, 1) * a / 2
    x = np.vstack((np.hstack((d1x, d1y)), np.hstack((-d1x, -d1y)))) / 3
    x += np.random.randn(*x.shape) * 0.09
    data = x.astype(np.float32)
    return data


def create_checkerboard_data(n):
    """
    Modified from https://github.com/rtqichen/ffjord/blob/master/lib/toy_data.py
    """
    x1 = np.random.rand(n) * 8 - 4
    x2_ = np.random.rand(n) + np.random.randint(-2, 2, n) * 2
    x2 = x2_ + (np.floor(x1) % 2)
    data = (np.concatenate([x1[:, None], x2[:, None]], 1)).astype(np.float32)
    return data


def create_einstein_data(n, face='einstein'):
    root = io.get_image_root()
    path = os.path.join(root, face + '.jpg')
    image = imageio.imread(path)
    image = color.rgb2gray(image)
    image = transform.resize(image, (512, 512))

    grid = np.array([
        (x, y) for x in range(image.shape[0]) for y in range(image.shape[1])
    ])

    rotation_matrix = np.array([
        [0, -1],
        [1, 0]
    ])
    p = image.reshape(-1) / sum(image.reshape(-1))
    ix = np.random.choice(range(len(grid)), size=n, replace=True, p=p)
    points = grid[ix].astype(np.float32)
    points += np.random.rand(n, 2)  # dequantize
    points /= (image.shape[0])  # scale to [0, 1]

    data = (points @ rotation_matrix).astype(np.float32)
    data[:, 1] += 1
    return data
