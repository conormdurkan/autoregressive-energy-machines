import os
import shutil
import sys
import tarfile

from tqdm import tqdm
from urllib import request

from shared import io


def download_and_extract(data_dir):

    filename = os.path.join(data_dir, 'data.tar.gz')

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

    with tqdm(unit='B', unit_scale=True, unit_divisor=1024, miniters=1) as tbar:
        tbar.set_description('Downloading datasets...')
        request.urlretrieve(
            url='https://zenodo.org/record/1161203/files/data.tar.gz?download=1',
            filename=filename,
            reporthook=reporthook(tbar)
        )
        tbar.set_description('Finished downloading.')

    print('\nExtracting datasets...')
    with tarfile.open(filename, 'r:gz') as file:
        file.extractall(path=data_dir)
    print('Finished extraction.\n')

    print('Removing zipped data...')
    os.remove(filename)
    print('Zipped data removed.\n')

    response = input('Do you wish to delete CIFAR-10? [Y/n]')
    if response in ['y', '']:
        shutil.rmtree(os.path.join(data_dir, 'data/cifar10'))
        print('CIFAR-10 removed.\n')
    elif response == 'n':
        print('CIFAR-10 not removed.\n')
    else:
        print('Response not understood. CIFAR-10 not removed.\n')


def download_data():
    data_root = io.get_data_root()
    if not os.path.isdir(os.path.join(data_root, 'data')):
        query = (
            "> UCI, BSDS300, and MNIST data not found.\n"
            "> The zipped download is 817MB in size, and 1.6GB once unzipped.\n"
            "> The download includes CIFAR-10, although it is not used in the paper.\n"
            "> After extraction, this script will delete the zipped download.\n"
            "> You will also be able to specify whether to delete CIFAR-10.\n"
            "> Do you wish to download the data? [Y/n]"
        )
        response = input(query)
        if response in ['y', '']:
            download_and_extract(data_root)
        elif response == 'n':
            sys.exit()
        else:
            print('Response not understood.')
            sys.exit()


