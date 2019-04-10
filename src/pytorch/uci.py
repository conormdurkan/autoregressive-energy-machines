import argparse
import json
import numpy as np
import os
import torch

import data_
import models
import utils

from tensorboardX import SummaryWriter
from torch import optim
from torch.utils import data
from tqdm import tqdm

from utils import io

parser = argparse.ArgumentParser()

# CUDA
parser.add_argument('--use_gpu', default=True, help='Whether to use GPU.')

# data
parser.add_argument('--dataset_name', type=str, default='hepmass', help='Dataset to use.')
parser.add_argument('--train_batch_size', type=int, default=512,
                    help='Size of batch used for training.')
parser.add_argument('--val_frac', type=float, default=0.1,
                    help='Fraction of validation set to use.')
parser.add_argument('--val_batch_size', type=int, default=512,
                    help='Size of batch used for validation.')
parser.add_argument('--num_workers', type=int, default=0,
                    help='Number of workers used in data loaders.')

# MADE
parser.add_argument('--n_residual_blocks_made', default=4,
                    help='Number of residual blocks in MADE.')
parser.add_argument('--hidden_dim_made', default=512,
                    help='Dimensionality of hidden layers in MADE.')
parser.add_argument('--activation_made', default='relu',
                    help='Activation function for MADE.')
parser.add_argument('--use_batch_norm_made', default=False,
                    help='Whether to use batch norm in MADE.')
parser.add_argument('--dropout_probability_made', default=0.2,
                    help='Dropout probability for MADE.')

# energy net
parser.add_argument('--context_dim', default=64,
                    help='Dimensionality of context vector.')
parser.add_argument('--n_residual_blocks_energy_net', default=4,
                    help='Number of residual blocks in energy net.')
parser.add_argument('--hidden_dim_energy_net', default=128,
                    help='Dimensionality of hidden layers in energy net.')
parser.add_argument('--energy_upper_bound', default=0,
                    help='Max value for output of energy net.')
parser.add_argument('--activation_energy_net', default='relu',
                    help='Activation function for energy net.')
parser.add_argument('--use_batch_norm_energy_net', default=False,
                    help='Whether to use batch norm in energy net.')
parser.add_argument('--dropout_probability_energy_net', default=0.2,
                    help='Dropout probability for energy net.')
parser.add_argument('--scale_activation', default='softplus',
                    help='Activation to use for scales in proposal mixture components.')
parser.add_argument('--apply_context_activation', default=False,
                    help='Whether to apply activation to context vector.')

# proposal
parser.add_argument('--n_mixture_components', default=20,
                    help='Number of proposal mixture components (per dimension).')
parser.add_argument('--proposal_component', default='gaussian',
                    help='Type of location-scale family distribution '
                         'to use in proposal mixture.')
parser.add_argument('--n_proposal_samples_per_input', default=20,
                    help='Number of proposal samples used to estimate '
                         'normalizing constant during training.')
parser.add_argument('--n_proposal_samples_per_input_validation', default=20,
                    help='Number of proposal samples used to estimate '
                         'normalizing constant during validation.')
parser.add_argument('--mixture_component_min_scale', default=1e-3,
                    help='Minimum scale for proposal mixture components.')

# optimization
parser.add_argument('--learning_rate', default=5e-4,
                    help='Learning rate for Adam.')
parser.add_argument('--n_total_steps', default=400000,
                    help='Number of total training steps.')
parser.add_argument('--alpha_warm_up_steps', default=5000,
                    help='Number of warm-up steps for AEM density.')
parser.add_argument('--hard_alpha_warm_up', default=True,
                    help='Whether to use a hard warm up for alpha')

# logging and checkpoints
parser.add_argument('--monitor_interval', default=100,
                    help='Interval in steps at which to report training stats.')
parser.add_argument('--save_interval', default=10000,
                    help='Interval in steps at which to save model.')

# reproducibility
parser.add_argument('--seed', default=1638128,
                    help='Random seed for PyTorch and NumPy.')

args = parser.parse_args()

torch.manual_seed(args.seed)
np.random.seed(args.seed)

if args.use_gpu and torch.cuda.is_available():
    device = torch.device('cuda')
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
    device = torch.device('cpu')

# create datasets
# training set
train_dataset = data_.load_uci_dataset(args.dataset_name, split='train')
train_loader = data_.InfiniteLoader(
    dataset=train_dataset,
    batch_size=args.train_batch_size,
    shuffle=True,
    drop_last=True,
    num_epochs=None
)

# validation set
val_dataset = data_.load_uci_dataset(args.dataset_name, split='val', frac=args.val_frac)
val_loader = data.DataLoader(
    dataset=val_dataset,
    batch_size=args.val_batch_size,
    shuffle=True,
    drop_last=True,
    num_workers=args.num_workers
)

# define parameters for MADE and energy net
dim = train_dataset.dim  # D
output_dim_multiplier = args.context_dim + 3 * args.n_mixture_components  # K + 3M

# Create MADE
made = models.ResidualMADE(
    input_dim=dim,
    n_residual_blocks=args.n_residual_blocks_made,
    hidden_dim=args.hidden_dim_made,
    output_dim_multiplier=output_dim_multiplier,
    conditional=False,
    activation=utils.parse_activation(args.activation_made),
    use_batch_norm=args.use_batch_norm_made,
    dropout_probability=args.dropout_probability_made
).to(device)

# Create energy net
energy_net = models.ResidualEnergyNet(
    input_dim=(args.context_dim + 1),
    n_residual_blocks=args.n_residual_blocks_energy_net,
    hidden_dim=args.hidden_dim_energy_net,
    energy_upper_bound=args.energy_upper_bound,
    activation=utils.parse_activation(args.activation_energy_net),
    use_batch_norm=args.use_batch_norm_energy_net,
    dropout_probability=args.dropout_probability_energy_net
).to(device)

# create AEM
aem = models.AEM(
    autoregressive_net=made,
    energy_net=energy_net,
    context_dim=args.context_dim,
    n_proposal_mixture_components=args.n_mixture_components,
    proposal_component_family=args.proposal_component,
    n_proposal_samples_per_input=args.n_proposal_samples_per_input,
    mixture_component_min_scale=args.mixture_component_min_scale,
    apply_context_activation=args.apply_context_activation
).to(device)

# optimizer and scheduler
parameters = list(made.parameters()) + list(energy_net.parameters())
optimizer = optim.Adam(parameters, lr=args.learning_rate)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, args.n_total_steps)

# create Summary Writer
timestamp = io.get_timestamp()
log_dir = os.path.join(io.get_log_root(), args.dataset_name, timestamp)
writer = SummaryWriter(log_dir=log_dir)
filename = os.path.join(log_dir, 'config.json')
with open(filename, 'w') as file:
    json.dump(vars(args), file)

# Training loop
tbar = tqdm(range(args.n_total_steps))
alpha = 0
for step in tbar:
    aem.train()
    scheduler.step()
    optimizer.zero_grad()

    # training step
    batch = next(train_loader).to(device)
    log_density, log_proposal_density, _, log_normalizer = aem(batch)
    mean_log_density = torch.mean(log_density)
    mean_log_proposal_density = torch.mean(log_proposal_density)
    mean_log_normalizer = torch.mean(log_normalizer)

    if args.alpha_warm_up_steps is not None:
        if args.hard_alpha_warm_up:
            alpha = float(step > args.alpha_warm_up_steps)
        else:
            alpha = torch.Tensor([min(step / args.alpha_warm_up_steps, 1)])
        loss = - (alpha * mean_log_density + mean_log_proposal_density)
    else:
        loss = - (mean_log_density + mean_log_proposal_density)
    loss.backward()
    optimizer.step()

    if (step + 1) % args.monitor_interval == 0:
        loss = loss.detach()
        aem.eval()
        aem.set_n_proposal_samples_per_input_validation(args.n_proposal_samples_per_input_validation)
        running_val_log_density = 0
        running_val_log_proposal_density = 0
        for val_batch in val_loader:
            log_density_val, log_proposal_density_val, _, _ = aem(val_batch.to(device).detach())
            mean_log_density_val = torch.mean(log_density_val).detach()
            mean_log_proposal_density_val = torch.mean(log_proposal_density_val).detach()
            running_val_log_density += mean_log_density_val
            running_val_log_proposal_density += mean_log_proposal_density_val
        running_val_log_density /= len(val_loader)
        running_val_log_proposal_density /= len(val_loader)

        s = 'Loss: {:.4f}, ' \
            'train: {:.4f}, ' \
            'val: {:.4f}, ' \
            'train prop: {:.4f}, ' \
            'val prop: {:.4f}'.format(
            loss.item(),
            mean_log_density.item(),
            running_val_log_density.item(),
            mean_log_proposal_density.item(),
            running_val_log_proposal_density.item()
        )
        tbar.set_description(s)

        # write summaries
        summaries = {
            'loss': loss.detach(),
            'log-prob-aem': mean_log_density.detach(),
            'log-prob-proposal': mean_log_proposal_density.detach(),
            'log-prob-aem-val': torch.Tensor([running_val_log_density]),
            'log-prob-proposal-val': torch.Tensor([running_val_log_proposal_density]),
            'log-normalizer': mean_log_normalizer.detach(),
            'learning-rate': torch.Tensor(scheduler.get_lr()),
        }
        for summary, value in summaries.items():
            writer.add_scalar(tag=summary, scalar_value=value, global_step=step)

        if (step + 1) % args.save_interval == 0:
            path = os.path.join(io.get_checkpoint_root(), 'pytorch', '{}.t'.format(args.dataset_name))
            if not os.path.exists(path):
                os.makedirs(io.get_checkpoint_root())
            torch.save(aem.state_dict(), path)

path = os.path.join(io.get_checkpoint_root(), 'pytorch', '{}-{}.t'.format(args.dataset_name, timestamp))
torch.save(aem.state_dict(), path)
