import h5py
import os

from shared import io
from shared.datautils.download import download_data


def load_bsds300():
    path = os.path.join(io.get_data_root(), 'data', 'BSDS300/BSDS300.hdf5')
    try:
        file = h5py.File(path, 'r')
    except FileNotFoundError:
        download_data()
        file = h5py.File(path, 'r')

    return file['train'], file['validation'], file['test']
