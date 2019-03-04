import torch

from torch.nn import functional as F


def tile(x, n):
    assert isinstance(n, int) and n > 0, 'Argument \'n\' must be an integer.'
    x_ = x.reshape(-1)
    x_ = x_.repeat(n)
    x_ = x_.reshape(n, -1)
    x_ = x_.transpose(1, 0)
    x_ = x_.reshape(-1)
    return x_


def tensor2numpy(x):
    return x.detach().cpu().numpy()


def parse_activation(activation):
    activations = {
        'relu': F.relu,
        'tanh': torch.tanh,
        'sigmoid': F.sigmoid,
        'softplus': F.softplus
    }
    return activations[activation]


def get_n_parameters(model):
    total_n_parameters = 0
    for parameter in model.parameters():
        total_n_parameters += torch.numel(parameter)
    return total_n_parameters


def test():
    a = torch.arange(3)
    print(tile(a, 2))


if __name__ == '__main__':
    test()