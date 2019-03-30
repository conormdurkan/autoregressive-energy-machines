""" Data loading utilities"""
import os
from collections import Counter

import h5py
import numpy as np
import tensorflow as tf

import imageio
import pandas as pd
import os
import sys

from .data_generators_2D import datasets2D


def UCI(dataset, batch_size=64):
    data_train, data_val, data_test = load_UCI(dataset)
    Dataset = tf.data.Dataset.from_tensor_slices(
        tf.constant(np.random.permutation(data_train), tf.float32)
    )
    Dataset = Dataset.shuffle(buffer_size=10000)
    Dataset = Dataset.batch(batch_size)
    Dataset = Dataset.repeat()
    iterator = Dataset.make_one_shot_iterator()
    x_batch = iterator.get_next()
    return x_batch, data_val, data_test


def Datasets2D(dataset, batch_size=64, n_ex=100000):
    data_raw = datasets2D(dataset, n_ex)
    Dataset = tf.data.Dataset.from_tensor_slices(
        tf.constant(np.random.permutation(data_raw), tf.float32)
    )
    Dataset = Dataset.shuffle(buffer_size=10000)
    Dataset = Dataset.batch(batch_size)
    Dataset = Dataset.repeat()
    iterator = Dataset.make_one_shot_iterator()
    x_batch = iterator.get_next()
    return x_batch, data_raw


def _load_mnist(n_val_ex=400):
    """Load MNIST data (no labels)."""
    (x_train, _), (x_test, _) = tf.keras.datasets.mnist.load_data()
    x_val = x_train[-n_val_ex:]
    x_train = x_train[:-n_val_ex]
    return x_train, x_val, x_test


def _load_statically_binarized_mnist():
    """Load Hugo Larochelle's statically binarized MNIST."""
    ims, labels = np.split(
        imageio.imread("https://i.imgur.com/j0SOfRW.png ")[..., :3].ravel(), [-70000]
    )
    ims = np.unpackbits(ims).reshape((-1, 28, 28))
    ims, labels = [np.split(y, [50000, 60000]) for y in (ims, labels)]
    train_ims = ims[0][..., None]
    val_ims = ims[1][..., None]
    test_ims = ims[1][..., None]
    return train_ims, val_ims, test_ims


def BinarizedMNIST(batch_size=64, n_val_ex=400, binarization="dynamic"):
    """Binarized MNIST data loader."""
    if binarization == "static":
        x_train, x_val, x_test = _load_statically_binarized_mnist()
    elif binarization == "dynamic":
        x_train, x_val, x_test = _load_mnist(n_val_ex=n_val_ex)
    Dataset = tf.data.Dataset.from_tensor_slices(
        tf.constant(np.random.permutation(x_train)[..., None], tf.float32)
    )
    Dataset = Dataset.shuffle(buffer_size=10000)
    Dataset = Dataset.batch(batch_size)
    Dataset = Dataset.repeat()
    iterator = Dataset.make_one_shot_iterator()
    if binarization == "dynamic":
        x_batch_uint8 = iterator.get_next()
        x_batch = tf.cast(x_batch_uint8, tf.float32) / 255.0
        x_batch = tf.cast(x_batch > tf.random_uniform(tf.shape(x_batch)), tf.float32)
    elif binarization == "static":
        x_batch = iterator.get_next()
        x_batch_uint8 = None
    return x_batch_uint8, x_batch, x_val[..., None], x_test[..., None]


def load_cifar10(batch_size=64, val_prop=0.05):
    """Load CIFAR10."""
    train, test = tf.keras.datasets.cifar10.load_data()
    x_train = train[0]
    x_test = test[0]
    n_train = x_train.shape[0]
    X_val = x_train[int((1 - val_prop) * n_train) :]
    x_train = x_train[: int((1 - val_prop) * n_train)]
    return x_train, X_val, x_test


def load_UCI(dataset, data_root="data/UCI"):
    """Load UCI and BSDS300 datasets."""

    if dataset == "power":
        data_loader = load_power
    elif dataset == "gas":
        data_loader = load_gas
    elif dataset == "hepmass":
        data_loader = load_hepmass
    elif dataset == "miniboone":
        data_loader = load_miniboone
    elif dataset == "bsds300":
        data_loader = load_bsds300
    return data_loader(data_root=data_root)


def load_power(data_root="data/UCI"):
    """Load UCI dataset: Power."""

    def load_data():
        file = os.path.join(data_root, "power/data.npy")
        return np.load(file)

    def load_data_split_with_noise():
        rng = np.random.RandomState(42)

        data = load_data()
        rng.shuffle(data)
        N = data.shape[0]

        data = np.delete(data, 3, axis=1)
        data = np.delete(data, 1, axis=1)
        ############################
        # Add noise
        ############################
        # global_intensity_noise = 0.1*rng.rand(N, 1)
        voltage_noise = 0.01 * rng.rand(N, 1)
        # grp_noise = 0.001*rng.rand(N, 1)
        gap_noise = 0.001 * rng.rand(N, 1)
        sm_noise = rng.rand(N, 3)
        time_noise = np.zeros((N, 1))
        noise = np.hstack((gap_noise, voltage_noise, sm_noise, time_noise))
        data = data + noise

        N_test = int(0.1 * data.shape[0])
        data_test = data[-N_test:]
        data = data[0:-N_test]
        N_validate = int(0.1 * data.shape[0])
        data_validate = data[-N_validate:]
        data_train = data[0:-N_validate]

        return data_train, data_validate, data_test

    def load_data_normalised():
        data_train, data_validate, data_test = load_data_split_with_noise()
        data = np.vstack((data_train, data_validate))
        mu = data.mean(axis=0)
        s = data.std(axis=0)
        data_train = (data_train - mu) / s
        data_validate = (data_validate - mu) / s
        data_test = (data_test - mu) / s

        return data_train, data_validate, data_test

    return load_data_normalised()


def load_gas(data_root="data/UCI"):
    """Load UCI dataset: Gas."""

    def load_data(file):
        data = pd.read_pickle(file)
        data.drop("Meth", axis=1, inplace=True)
        data.drop("Eth", axis=1, inplace=True)
        data.drop("Time", axis=1, inplace=True)
        return data

    def get_correlation_numbers(data):
        C = data.corr()
        A = C > 0.98
        B = A.sum(axis=1)
        return B

    def load_data_and_clean(file):
        data = load_data(file)
        B = get_correlation_numbers(data)

        while np.any(B > 1):
            col_to_remove = np.where(B > 1)[0][0]
            col_name = data.columns[col_to_remove]
            data.drop(col_name, axis=1, inplace=True)
            B = get_correlation_numbers(data)
        data = (data - data.mean()) / data.std()

        return data

    def load_data_and_clean_and_split(file):
        data = load_data_and_clean(file)
        N_test = int(0.1 * data.shape[0])
        data_test = data[-N_test:]
        data_train = data[0:-N_test]
        N_validate = int(0.1 * data_train.shape[0])
        data_validate = data_train[-N_validate:]
        data_train = data_train[0:-N_validate]

        return np.array(data_train), np.array(data_validate), np.array(data_test)

    file = os.path.join(data_root, "gas/ethylene_CO.pickle")
    return load_data_and_clean_and_split(file)


def load_hepmass(data_root="data/UCI"):
    """Load UCI dataset: Hepmass."""

    def load_data(path):

        data_train = pd.read_csv(
            filepath_or_buffer=os.path.join(path, "1000_train.csv"), index_col=False
        )
        data_test = pd.read_csv(
            filepath_or_buffer=os.path.join(path, "1000_test.csv"), index_col=False
        )

        return data_train, data_test

    def load_data_no_discrete(path):
        data_train, data_test = load_data(path)

        # Gets rid of any background noise examples i.e. class label 0.
        data_train = data_train[data_train[data_train.columns[0]] == 1]
        data_train = data_train.drop(data_train.columns[0], axis=1)
        data_test = data_test[data_test[data_test.columns[0]] == 1]
        data_test = data_test.drop(data_test.columns[0], axis=1)
        # Because the data set is messed up!
        data_test = data_test.drop(data_test.columns[-1], axis=1)

        return data_train, data_test

    def load_data_no_discrete_normalised(path):

        data_train, data_test = load_data_no_discrete(path)
        mu = data_train.mean()
        s = data_train.std()
        data_train = (data_train - mu) / s
        data_test = (data_test - mu) / s

        return data_train, data_test

    def load_data_no_discrete_normalised_as_array(path):

        data_train, data_test = load_data_no_discrete_normalised(path)
        data_train, data_test = data_train.values, data_test.values

        i = 0
        # Remove any features that have too many re-occurring real values.
        features_to_remove = []
        for feature in data_train.T:
            c = Counter(feature)
            max_count = np.array([v for k, v in sorted(c.items())])[0]
            if max_count > 5:
                features_to_remove.append(i)
            i += 1
        data_train = data_train[
            :,
            np.array(
                [i for i in range(data_train.shape[1]) if i not in features_to_remove]
            ),
        ]
        data_test = data_test[
            :,
            np.array(
                [i for i in range(data_test.shape[1]) if i not in features_to_remove]
            ),
        ]

        N = data_train.shape[0]
        N_validate = int(N * 0.1)
        data_validate = data_train[-N_validate:]
        data_train = data_train[0:-N_validate]

        return data_train, data_validate, data_test

    path = os.path.join(data_root, "hepmass")
    return load_data_no_discrete_normalised_as_array(path)


def load_miniboone(data_root="data/UCI"):
    """Load UCI dataset: Miniboone."""

    def load_data(root_path):
        data = np.load(root_path)
        N_test = int(0.1 * data.shape[0])
        data_test = data[-N_test:]
        data = data[0:-N_test]
        N_validate = int(0.1 * data.shape[0])
        data_validate = data[-N_validate:]
        data_train = data[0:-N_validate]

        return data_train, data_validate, data_test

    def load_data_normalised(root_path):
        data_train, data_validate, data_test = load_data(root_path)
        data = np.vstack((data_train, data_validate))
        mu = data.mean(axis=0)
        s = data.std(axis=0)
        data_train = (data_train - mu) / s
        data_validate = (data_validate - mu) / s
        data_test = (data_test - mu) / s

        return data_train, data_validate, data_test

    path = os.path.join(data_root, "miniboone/data.npy")
    return load_data_normalised(path)


def load_bsds300(data_root="data/UCI"):
    """Load BSDS300."""
    path = os.path.join(data_root, "BSDS300/BSDS300.hdf5")
    file = h5py.File(path, "r")

    return np.array(file["train"]), np.array(file["validation"]), np.array(file["test"])
