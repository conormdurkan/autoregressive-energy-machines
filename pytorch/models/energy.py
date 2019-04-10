from torch import nn
from torch.nn import functional as F, init


class ResidualBlock(nn.Module):
    def __init__(self, features, activation, zero_initialization=True,
                 dropout_probability=0.1, use_batch_norm=True):
        super().__init__()
        self.activation = activation

        self.use_batch_norm = use_batch_norm
        if use_batch_norm:
            self.batch_norm_layers = nn.ModuleList([
                nn.BatchNorm1d(features, eps=1e-3)
                for _ in range(2)
            ])
        self.layers = nn.ModuleList([
            nn.Linear(features, features)
            for _ in range(2)
        ])
        self.dropout = nn.Dropout(p=dropout_probability)
        if zero_initialization:
            init.zeros_(self.layers[-1].weight)
            init.zeros_(self.layers[-1].bias)

    def forward(self, inputs):
        temps = inputs
        if self.use_batch_norm:
            temps = self.batch_norm_layers[0](temps)
        temps = self.activation(temps)
        temps = self.layers[0](temps)
        if self.use_batch_norm:
            temps = self.batch_norm_layers[1](temps)
        temps = self.activation(temps)
        temps = self.dropout(temps)
        temps = self.layers[1](temps)
        return temps + inputs


class EnergyNet(nn.Module):
    def __init__(self, input_dim, n_hidden_layers=2, hidden_dim=32,
                 energy_upper_bound=None,
                 activation=F.relu):
        super().__init__()
        self.energy_upper_bound = energy_upper_bound
        self.activation = activation
        self.initial_layer = nn.Linear(input_dim, hidden_dim)
        self.hidden_layers = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim) for _ in range(n_hidden_layers)
        ])
        self.final_layer = nn.Linear(hidden_dim, 1)

    def forward(self, inputs):
        temps = self.initial_layer(inputs)
        temps = self.activation(temps)

        for layer in self.hidden_layers:
            temps = layer(temps)
            temps = self.activation(temps)

        outputs = self.final_layer(temps)
        if self.energy_upper_bound is not None:
            outputs = -F.softplus(outputs) + self.energy_upper_bound
        return outputs


class ResidualEnergyNet(nn.Module):
    def __init__(self, input_dim, n_residual_blocks=2, hidden_dim=32,
                 energy_upper_bound=None,
                 activation=F.relu, use_batch_norm=False, dropout_probability=None):
        super().__init__()
        self.activation = activation
        self.energy_upper_bound = energy_upper_bound
        self.initial_layer = nn.Linear(input_dim, hidden_dim)
        self.blocks = nn.ModuleList([
            ResidualBlock(
                features=hidden_dim,
                activation=activation,
                use_batch_norm=use_batch_norm,
                dropout_probability=0 if dropout_probability is None else dropout_probability
            )
            for _ in range(n_residual_blocks)
        ])
        self.final_layer = nn.Linear(hidden_dim, 1)

    def forward(self, inputs):
        temps = self.initial_layer(inputs)
        del inputs  # free GPU memory
        for block in self.blocks:
            temps = block(temps)
        temps = self.activation(temps)
        outputs = self.final_layer(temps)
        if self.energy_upper_bound is not None:
            outputs = -F.softplus(outputs) + self.energy_upper_bound
        return outputs
