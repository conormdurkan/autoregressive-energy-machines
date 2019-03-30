""" Data loading utilities"""
import os
import h5py
import numpy as np
import tensorflow as tf
import imageio
import pandas as pd
import os
import sys
import shutil
import tarfile

from tqdm import tqdm
from urllib import request
from collections import Counter

from .data_generators_2D import gen_2D_data


def UCI(dataset, batch_size=64):
    data_train, data_val, data_test = load_UCI_np(dataset)
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
    data_raw = gen_2D_data(dataset, n_ex)
    Dataset = tf.data.Dataset.from_tensor_slices(
        tf.constant(np.random.permutation(data_raw), tf.float32)
    )
    Dataset = Dataset.shuffle(buffer_size=10000)
    Dataset = Dataset.batch(batch_size)
    Dataset = Dataset.repeat()
    iterator = Dataset.make_one_shot_iterator()
    x_batch = iterator.get_next()
    return x_batch, data_raw


def load_UCI_np(dataset, data_root="data"):
    """Load UCI and BSDS300 datasets."""
    data_path = os.path.join(data_root, "processed", dataset)
    if not os.path.exists(data_path):
        download_preprocess_data(data_root=data_root)
    train_data = np.load(os.path.join(data_path, "train.npy"))
    val_data = np.load(os.path.join(data_path, "train.npy"))
    test_data = np.load(os.path.join(data_path, "train.npy"))
    return train_data, val_data, test_data


def preprocess_gas(data_root="data"):
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

        return data.values

    def load_data_and_clean_and_split(file):
        data = load_data_and_clean(file)
        N_test = int(0.1 * data.shape[0])
        data_test = data[-N_test:]
        data_train = data[0:-N_test]
        N_validate = int(0.1 * data_train.shape[0])
        data_validate = data_train[-N_validate:]
        data_train = data_train[0:-N_validate]

        return data_train, data_validate, data_test

    file = os.path.join(data_root, "raw/gas/ethylene_CO.pickle")
    return load_data_and_clean_and_split(file)


def preprocess_power(data_root="data"):
    def load_data_split_with_noise(data):
        rng = np.random.RandomState(42)
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
        # noise = np.hstack((gap_noise, grp_noise, voltage_noise, global_intensity_noise, sm_noise, time_noise))
        # noise = np.hstack((gap_noise, grp_noise, voltage_noise, sm_noise, time_noise))
        noise = np.hstack((gap_noise, voltage_noise, sm_noise, time_noise))
        data = data + noise

        N_test = int(0.1 * data.shape[0])
        data_test = data[-N_test:]
        data = data[0:-N_test]
        N_validate = int(0.1 * data.shape[0])
        data_validate = data[-N_validate:]
        data_train = data[0:-N_validate]

        return data_train, data_validate, data_test

    def load_data_normalised(data):
        data_train, data_validate, data_test = load_data_split_with_noise(data)
        data = np.vstack((data_train, data_validate))
        mu = data.mean(axis=0)
        s = data.std(axis=0)
        data_train = (data_train - mu) / s
        data_validate = (data_validate - mu) / s
        data_test = (data_test - mu) / s

        return data_train, data_validate, data_test

    file = os.path.join(data_root, "raw/power/data.npy")
    data = np.load(file)
    return load_data_normalised(data)


def preprocess_hepmass(data_root="data"):
    def load_data(path):
        data_train = pd.read_csv(
            filepath_or_buffer=os.path.join(path, "1000_train.csv"), index_col=False
        )
        data_test = pd.read_csv(
            filepath_or_buffer=os.path.join(path, "1000_test.csv"), index_col=False
        )
        return data_train, data_test

    def load_data_no_discrete(path):
        """
        Loads the positive class examples from the first 10 percent of the dataset.
        """
        data_train, data_test = load_data(path)

        # Gets rid of any background noise examples i.e. class label 0.
        data_train = data_train[data_train[data_train.columns[0]] == 1]
        data_train = data_train.drop(data_train.columns[0], axis=1)
        data_test = data_test[data_test[data_test.columns[0]] == 1]
        data_test = data_test.drop(data_test.columns[0], axis=1)
        # Because the data_ set is messed up!
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

    path = os.path.join(data_root, "raw/hepmass")
    return load_data_no_discrete_normalised_as_array(path)


def preprocess_miniboone(data_root="data"):
    def load_data(root_path):
        # NOTE: To remember how the pre-processing was done.
        # data_ = pd.read_csv(root_path, names=[str(x) for x in range(50)], delim_whitespace=True)
        # print data_.head()
        # data_ = data_.as_matrix()
        # # Remove some random outliers
        # indices = (data_[:, 0] < -100)
        # data_ = data_[~indices]
        #
        # i = 0
        # # Remove any features that have too many re-occuring real values.
        # features_to_remove = []
        # for feature in data_.T:
        #     c = Counter(feature)
        #     max_count = np.array([v for k, v in sorted(c.iteritems())])[0]
        #     if max_count > 5:
        #         features_to_remove.append(i)
        #     i += 1
        # data_ = data_[:, np.array([i for i in range(data_.shape[1]) if i not in features_to_remove])]
        # np.save("~/data_/miniboone/data_.npy", data_)
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

    path = os.path.join(data_root, "raw/miniboone/data.npy")
    return load_data_normalised(path)


def preprocess_bsds300(data_root="data"):
    path = os.path.join(data_root, "raw/BSDS300/BSDS300.hdf5")
    file = h5py.File(path, "r")
    return file["train"], file["validation"], file["test"]


def preprocess_UCI_data(data_root="data"):
    preprocess_dict = {
        "gas": preprocess_gas,
        "power": preprocess_power,
        "hepmass": preprocess_hepmass,
        "miniboone": preprocess_miniboone,
        "bsds300": preprocess_bsds300,
    }
    for dataset, preprocess_fn in preprocess_dict.items():
        train, val, test = preprocess_fn(data_root)
        splits = (("train", train), ("val", val), ("test", test))
        output_dir = os.path.join(data_root, "processed", dataset)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        for split in splits:
            name, data = split
            file = os.path.join(output_dir, "{}.npy".format(name))
            np.save(file, data)


def download_and_extract(data_root="data"):

    if not os.path.exists(data_root):
        os.makedirs(data_root)
    filename = os.path.join(data_root, "data.tar.gz")

    def reporthook(tbar):
        """
        tqdm report hook for data download.
        From https://github.com/tqdm/tqdm/blob/master/examples/tqdm_wget.py
        """
        last_block = [0]

        def update_to(block=1, block_size=1, tbar_size=None):
            if tbar_size is not None:
                tbar.total = tbar_size
            tbar.update((block - last_block[0]) * block_size)
            last_block[0] = block

        return update_to

    with tqdm(unit="B", unit_scale=True, unit_divisor=1024, miniters=1) as tbar:
        tbar.set_description("Downloading datasets...")
        request.urlretrieve(
            url="https://zenodo.org/record/1161203/files/data.tar.gz?download=1",
            filename=filename,
            reporthook=reporthook(tbar),
        )
        tbar.set_description("Finished downloading.")

    print("\nExtracting datasets...")
    with tarfile.open(filename, "r:gz") as file:
        print(filename)
        file.extractall(path=data_root)
    print("Finished extraction.\n")

    print("Removing zipped data...")
    os.remove(filename)
    os.rename(os.path.join(data_root, "data"), os.path.join(data_root, "raw"))
    print("Zipped data removed.\n")

    response = input("Do you wish to delete CIFAR-10? [Y/n]")
    if response in ["y", ""]:
        shutil.rmtree(os.path.join(data_root, "data/cifar10"))
        print("CIFAR-10 removed.\n")
    elif response == "n":
        print("CIFAR-10 not removed.\n")
    else:
        print("Response not understood. CIFAR-10 not removed.\n")


def download_data(data_root="data"):
    query = (
        "> UCI, BSDS300, and MNIST data not found.\n"
        "> The zipped download is 817MB in size, and 1.6GB once unzipped.\n"
        "> The download includes CIFAR-10, although it is not used in the paper.\n"
        "> After extraction, this script will delete the zipped download.\n"
        "> You will also be able to specify whether to delete CIFAR-10.\n"
        "> Do you wish to download the data? [Y/n]"
    )
    response = input(query)
    if response in ["y", ""]:
        download_and_extract(data_root)
    elif response == "n":
        sys.exit()
    else:
        print("Response not understood.")
        sys.exit()


def download_preprocess_data(data_root="data"):
    download_data(data_root=data_root)
    print("Pre-processing data...")
    preprocess_UCI_data(data_root=data_root)
    print("Data processed.")