import torch

import models

from torch import distributions, nn
from torch.nn import functional as F

from probability import distributions_


class AEM(nn.Module):
    def __init__(self, autoregressive_net, energy_net, context_dim,
                 n_proposal_mixture_components, proposal_component_family,
                 n_proposal_samples_per_input, mixture_component_min_scale=None,
                 scale_activation=F.softplus, apply_context_activation=False):
        super().__init__()

        self.autoregressive_net = autoregressive_net
        self.dim = autoregressive_net.input_dim

        self.energy_net = energy_net

        self.proposal = None
        if proposal_component_family == 'gaussian':
            self.Component = distributions_.Normal_
        elif proposal_component_family == 'cauchy':
            self.Component = distributions.Cauchy
        elif proposal_component_family == 'laplace':
            self.Component = distributions.Laplace
        elif proposal_component_family == 'uniform':
            self.Component = None

        self.context_dim = context_dim  # K
        self.n_proposal_mixture_components = n_proposal_mixture_components  # M
        self.made_output_dim_multiplier = context_dim + 3 * n_proposal_mixture_components
        self.n_proposal_samples_per_input = n_proposal_samples_per_input  # S
        self.n_proposal_samples_per_input_validation = int(1e2)
        self.mixture_component_min_scale = 0 if mixture_component_min_scale is None else mixture_component_min_scale
        self.scale_activation = scale_activation
        self.apply_context_activation = apply_context_activation

    def forward(self, inputs, conditional_inputs=None):

        # get energy params and proposal params
        if conditional_inputs is not None:
            autoregressive_outputs = self.autoregressive_net(
                inputs,
                conditional_inputs).reshape(-1, self.dim, self.made_output_dim_multiplier)
        else:
            autoregressive_outputs = self.autoregressive_net(
                inputs).reshape(-1, self.dim, self.made_output_dim_multiplier)

        context_params, proposal_params = (
            autoregressive_outputs[..., :self.context_dim],
            autoregressive_outputs[..., self.context_dim:]
        )
        del autoregressive_outputs  # free GPU memory
        if self.apply_context_activation:
            context_params = self.energy_net.activation(context_params)

        # separate out proposal params into coefficients, locs, and scales
        logits = proposal_params[..., :self.n_proposal_mixture_components]  # [B, D, M]
        if logits.shape[0] == 1:
            logits = logits.reshape(self.dim, self.n_proposal_mixture_components)
        locs = proposal_params[...,
               self.n_proposal_mixture_components:(
                       2 * self.n_proposal_mixture_components)]  # [B, D, M]
        scales = self.mixture_component_min_scale + self.scale_activation(
            proposal_params[..., (2 * self.n_proposal_mixture_components):])  # [B, D, M]

        # sample from proposal
        if self.training:
            n_proposal_samples_per_input = self.n_proposal_samples_per_input
        else:
            n_proposal_samples_per_input = self.n_proposal_samples_per_input_validation

        # create proposal
        if self.Component is not None:
            mixture_distribution = distributions.OneHotCategorical(
                logits=logits,
                validate_args=True
            )
            components_distribution = self.Component(loc=locs, scale=scales)
            self.proposal = distributions_.MixtureSameFamily(
                mixture_distribution=mixture_distribution,
                components_distribution=components_distribution
            )
            proposal_samples = self.proposal.sample(
                (n_proposal_samples_per_input,))  # [S, B, D]

        else:
            self.proposal = distributions.Uniform(low=-4, high=4)
            proposal_samples = self.proposal.sample(
                (n_proposal_samples_per_input, inputs.shape[0], inputs.shape[1])
            )
        del logits, locs, scales  # free GPU memory

        proposal_samples = proposal_samples.permute(1, 2, 0)  # [B, D, S]

        # evaluate data and proposal samples under proposal
        log_density_of_proposal_samples_under_proposal = self.proposal.log_prob(
            proposal_samples)  # [B, D, S]
        log_density_of_training_data_under_proposal = self.proposal.log_prob(
            inputs[..., None]).reshape(-1, self.dim)

        # energy net
        inputs_cat_samples = torch.cat(
            (inputs[..., None], proposal_samples.detach()),  # stop gradient
            dim=-1
        )
        inputs_cat_samples = inputs_cat_samples.reshape(-1, 1)
        context_params = context_params[..., None, :]
        context_params_tiled = context_params.repeat(
            1, 1, n_proposal_samples_per_input + 1, 1
        )
        del context_params  # free GPU memory
        context_params_tiled = context_params_tiled.reshape(-1, self.context_dim)

        energy_net_inputs = torch.cat(
            (inputs_cat_samples, context_params_tiled),
            dim=-1
        )
        del inputs_cat_samples, context_params_tiled  #  free GPU memory

        # Inputs to energy net can have very large batch size since we evaluate all importance samples at once.
        # We must split a very large batch to avoid OOM errors
        if energy_net_inputs.shape[0] > 300000 and not self.training:
            batch_size = 300000
            n_batches, leftover = (
                energy_net_inputs.shape[0] // batch_size,
                energy_net_inputs.shape[0] % batch_size
            )
            slices = [slice(batch_size * i, batch_size * (i + 1)) for i in
                      range(n_batches)]
            slices.append(
                slice(batch_size * n_batches, batch_size * n_batches + leftover))
            energy_net_outputs = torch.cat(
                [self.energy_net(energy_net_inputs[slice_]).detach()  # stop gradient
                 for slice_ in slices],
                dim=0
            )
        else:
            energy_net_outputs = self.energy_net(energy_net_inputs)

        del energy_net_inputs  # free GPU memory

        energy_net_outputs = energy_net_outputs.reshape(-1, self.dim,
                                                        1 + n_proposal_samples_per_input)

        # unnormalized log densities given by energy net
        unnormalized_log_densities_training_data = energy_net_outputs[..., 0]
        unnormalized_log_densities_proposal_samples = energy_net_outputs[..., 1:]

        # calculate log normalizer
        log_normalizer = torch.logsumexp(
            unnormalized_log_densities_proposal_samples -
            log_density_of_proposal_samples_under_proposal.detach(),  # stop gradient
            dim=-1
        ) - torch.log(torch.Tensor([n_proposal_samples_per_input]))

        # calculate normalized density
        log_density = torch.sum(
            unnormalized_log_densities_training_data - log_normalizer,
            dim=-1
        )
        log_proposal_density = torch.sum(
            log_density_of_training_data_under_proposal,
            dim=-1
        )

        outputs = (
            log_density,
            log_proposal_density,
            unnormalized_log_densities_training_data,
            log_normalizer
        )

        return outputs

    def context(self, inputs):
        # get energy params and proposal params
        autoregressive_outputs = self.autoregressive_net(inputs).reshape(-1, self.dim,
                                                                         self.made_output_dim_multiplier)
        context_params = autoregressive_outputs[..., :self.context_dim]
        return context_params

    def log_prob(self, inputs):
        log_density, _, _, _ = self.forward(inputs)
        return log_density

    def set_n_proposal_samples_per_input_validation(self,
                                                    n_proposal_samples_per_input=int(
                                                        1e2)):
        assert not self.training, 'Model must be in eval mode.'
        self.n_proposal_samples_per_input_validation = n_proposal_samples_per_input

    def _sample_batch_from_proposal(self, batch_size,
                                    return_log_density_of_samples=False):
        # need to do n_samples passes through autoregressive net
        samples = torch.zeros(batch_size, self.autoregressive_net.input_dim)
        log_density_of_samples = torch.zeros(batch_size,
                                             self.autoregressive_net.input_dim)
        for dim in range(self.autoregressive_net.input_dim):
            # compute autoregressive outputs
            autoregressive_outputs = self.autoregressive_net(samples).reshape(-1,
                                                                              self.dim,
                                                                              self.autoregressive_net.output_dim_multiplier)

            # grab proposal params for dth dimensions
            proposal_params = autoregressive_outputs[..., dim, self.context_dim:]

            # make mixture coefficients, locs, and scales for proposal
            logits = proposal_params[...,
                     :self.n_proposal_mixture_components]  # [B, D, M]
            if logits.shape[0] == 1:
                logits = logits.reshape(self.dim, self.n_proposal_mixture_components)
            locs = proposal_params[...,
                   self.n_proposal_mixture_components:(
                           2 * self.n_proposal_mixture_components)]  # [B, D, M]
            scales = self.mixture_component_min_scale + self.scale_activation(
                proposal_params[...,
                (2 * self.n_proposal_mixture_components):])  # [B, D, M]

            # create proposal
            if self.Component is not None:
                mixture_distribution = distributions.OneHotCategorical(
                    logits=logits,
                    validate_args=True
                )
                components_distribution = self.Component(loc=locs, scale=scales)
                self.proposal = distributions_.MixtureSameFamily(
                    mixture_distribution=mixture_distribution,
                    components_distribution=components_distribution
                )
                proposal_samples = self.proposal.sample((1,))  # [S, B, D]

            else:
                self.proposal = distributions.Uniform(low=-4, high=4)
                proposal_samples = self.proposal.sample(
                    (1, batch_size, 1)
                )
            proposal_samples = proposal_samples.permute(1, 2, 0)  # [B, D, S]
            proposal_log_density = self.proposal.log_prob(proposal_samples)
            log_density_of_samples[:, dim] += proposal_log_density.reshape(-1).detach()
            samples[:, dim] += proposal_samples.reshape(-1).detach()

        if return_log_density_of_samples:
            return samples, torch.sum(log_density_of_samples, dim=-1)
        else:
            return samples

    def sample_from_proposal(self, n_samples, return_log_density_of_samples=False,
                             batch_size=10000):
        if n_samples > batch_size:
            # determine how many batches are needed
            n_batches, leftover = n_samples // batch_size, n_samples % batch_size

            # get batches
            samples = torch.zeros(n_samples, self.autoregressive_net.input_dim)
            log_density_of_samples = torch.zeros(n_samples)
            for n in range(n_batches):
                batch_of_samples, log_density_of_batch_of_samples = self._sample_batch_from_proposal(
                    batch_size, return_log_density_of_samples=True)
                index = slice((batch_size * n), (batch_size * (n + 1)))
                samples[index, :] += batch_of_samples
                log_density_of_samples[index] += log_density_of_batch_of_samples

            if leftover:
                batch_of_samples, log_density_of_batch_of_samples = self._sample_batch_from_proposal(
                    leftover, return_log_density_of_samples=True)
                samples[-leftover:, :] += batch_of_samples
                log_density_of_samples[-leftover:, :] += log_density_of_batch_of_samples

            if return_log_density_of_samples:
                return samples, log_density_of_samples
            else:
                return samples

        else:
            return self._sample_batch_from_proposal(n_samples,
                                                    return_log_density_of_samples)

    def sample(self, batch_size=1000):
        samples = torch.zeros(batch_size, self.autoregressive_net.input_dim)
        for dim in range(self.autoregressive_net.input_dim):
            # compute autoregressive outputs
            autoregressive_outputs = self.autoregressive_net(samples).reshape(-1,
                                                                              self.dim,
                                                                              self.autoregressive_net.output_dim_multiplier)

            context_params, proposal_params = (
                autoregressive_outputs[..., dim, :self.context_dim][:, None, :],
                autoregressive_outputs[..., dim, self.context_dim:][:, None, :]
            )

            # make mixture coefficients, locs, and scales for proposal
            logits = proposal_params[...,
                     :self.n_proposal_mixture_components]  # [B, D, M]
            if logits.shape[0] == 1:
                logits = logits.reshape(self.dim, self.n_proposal_mixture_components)
            locs = proposal_params[...,
                   self.n_proposal_mixture_components:(
                           2 * self.n_proposal_mixture_components)]  # [B, D, M]
            scales = self.mixture_component_min_scale + self.scale_activation(
                proposal_params[...,
                (2 * self.n_proposal_mixture_components):])  # [B, D, M]

            # create proposal
            n_proposal_samples = 100
            if self.Component is not None:
                mixture_distribution = distributions.OneHotCategorical(
                    logits=logits,
                    validate_args=True
                )
                components_distribution = self.Component(loc=locs, scale=scales)
                self.proposal = distributions_.MixtureSameFamily(
                    mixture_distribution=mixture_distribution,
                    components_distribution=components_distribution
                )
                proposal_samples = self.proposal.sample(
                    (n_proposal_samples,))  # [S, B, D]

            else:
                self.proposal = distributions.Uniform(low=-4, high=4)
                proposal_samples = self.proposal.sample(
                    (n_proposal_samples, batch_size, 1)
                )  # [S, B, D]

            # reshape for log prob calculation
            proposal_samples = proposal_samples.permute(1, 2, 0)  # [B, D, S]
            proposal_log_density = self.proposal.log_prob(proposal_samples)
            proposal_log_density = proposal_log_density.reshape(batch_size,
                                                                n_proposal_samples)

            # reshape again for input to energy net
            proposal_samples = proposal_samples.reshape(-1, 1)
            energy_net_inputs = torch.cat(
                (proposal_samples,
                 context_params.repeat(1, n_proposal_samples, 1).reshape(
                     batch_size * n_proposal_samples, -1)),
                dim=-1
            )
            unnormalized_log_density = self.energy_net(energy_net_inputs)
            unnormalized_log_density = unnormalized_log_density.reshape(batch_size,
                                                                        n_proposal_samples)
            logits = unnormalized_log_density - proposal_log_density
            resampling_distribution = distributions.Categorical(logits=logits)
            selected_indices = resampling_distribution.sample((1,)).reshape(-1)
            proposal_samples = proposal_samples.reshape(batch_size, n_proposal_samples)
            selected_points = proposal_samples[range(batch_size), selected_indices]

            samples[:, dim] += selected_points.detach()

        return samples


def main():
    n_mixtures = 3
    input_dim = 2
    n_residual_blocks = 4
    hidden_dim = 5
    context_dim = 6
    output_dim_multiplier = 3 * n_mixtures + context_dim
    autoregressive_net = models.ResidualMADE(
        input_dim=input_dim,
        n_residual_blocks=n_residual_blocks,
        hidden_dim=hidden_dim,
        output_dim_multiplier=output_dim_multiplier
    )
    energy_net = models.ResidualEnergyNet(
        input_dim=(context_dim + 1),
        n_residual_blocks=n_residual_blocks,
        hidden_dim=hidden_dim,
        energy_upper_bound=0
    )
    aem = models.AEM(
        autoregressive_net=autoregressive_net,
        energy_net=energy_net,
        context_dim=context_dim,
        n_proposal_mixture_components=n_mixtures,
        proposal_component_family='gaussian',
        n_proposal_samples_per_input=20,
        mixture_component_min_scale=1e-3
    )
    n_samples = 10
    samples = aem.sample_from_proposal(n_samples)
    print(samples)


if __name__ == '__main__':
    main()
