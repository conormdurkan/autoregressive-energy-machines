import numpy as np
import os

from shared import io
from shared.datautils.download import download_data


def load_miniboone():
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

        try:
            data = np.load(root_path)
        except FileNotFoundError:
            download_data()
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

    path = os.path.join(io.get_data_root(), 'data', 'miniboone/data.npy')
    return load_data_normalised(path)


def preprocess_and_save_miniboone():
    train, val, test = load_miniboone()
    splits = (
        ('train', train),
        ('val', val),
        ('test', test)
    )
    for split in splits:
        name, data = split
        file = os.path.join(io.get_data_root(), 'data', 'miniboone/{}.npy'.format(name))
        np.save(file, data)
