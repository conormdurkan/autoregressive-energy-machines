import os
import shutil
import sys
import tarfile
import pandas as pd
import numpy as np

from tqdm import tqdm
from urllib import request


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


def preprocess_UCI_data(data_root="data"):
    preprocess_dict = {"gas": preprocess_gas}
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

    filename = os.path.join(data_root, "raw.tar.gz")

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
        file.extractall(path=filename)
    print("Finished extraction.\n")

    print("Removing zipped data...")
    os.remove(filename)
    print("Zipped data removed.\n")

    response = input("Do you wish to delete CIFAR-10? [Y/n]")
    if response in ["y", ""]:
        shutil.rmtree(os.path.join(data_root, "cifar10"))
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

