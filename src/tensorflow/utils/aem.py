""" Autoregressive Energy Machine (AEM)"""
import tensorflow as tf

from .made_utils import ResMADE
from .energy_nets import contextual_res_net


def get_activation(activation_name):
    if activation_name == "tanh":
        activation = tf.nn.tanh
    elif activation_name == "relu":
        activation = tf.nn.relu
    elif activation_name == "lrelu":
        activation = tf.nn.leaky_relu
    elif activation_name == "elu":
        activation = tf.nn.elu
    elif activation_name == "identity":
        activation = tf.identity
    return activation


class AEM:
    """Autoregressive Energy Machine (AEM). TODO""" 

    def __init__(
        self,
        x_batch,
        n_importance_samples=20,
        proposal_comp_scale_min=1e-3,
        n_proposal_mixture_comps=20,
        n_res_blocks_MADE=4,
        n_hidden_units_MADE=512,
        activation_MADE="relu",
        dropout_p_MADE=0.0,
        final_act_MADE=True,
        n_context_units_energy_net=64,
        n_res_blocks_energy_net=4,
        n_hidden_units_energy_net=128,
        activation_energy_net="relu",
        dropout_p_energy_net=0.0,
        final_act_energy_net=True,
    ):
        self.x_batch = x_batch
        self.n_importance_samples = n_importance_samples
        self.proposal_comp_scale_min = proposal_comp_scale_min
        self.n_proposal_mixture_comps = n_proposal_mixture_comps
        self.n_res_blocks_MADE = n_res_blocks_MADE
        self.n_hidden_units_MADE = n_hidden_units_MADE
        self.activation_MADE = activation_MADE
        self.dropout_p_MADE = dropout_p_MADE
        self.final_act_MADE = final_act_MADE
        self.n_context_units_energy_net = n_context_units_energy_net
        self.n_res_blocks_energy_net = n_res_blocks_energy_net
        self.n_hidden_units_energy_net = n_hidden_units_energy_net
        self.activation_energy_net = activation_energy_net
        self.dropout_p_energy_net = dropout_p_energy_net
        self.final_act_energy_net = final_act_energy_net

        self._energy_context = None
        self._proposal_params = None
        self._proposal_log_prob_data = None
        self._proposal_samples = None
        self._proposal_log_prob_samples_proposal = None
        self._unnorm_log_prob_data = None
        self._unnorm_log_prob_samples = None
        self._norm_constants_est = None
        self._log_prob_est_data = None

    @property
    def energy_context(self):
        if self._energy_context is None:
            self._process_inputs_MADE()
        return self._energy_context

    @property
    def proposal_params(self):
        if self._proposal_params is None:
            self._process_inputs_MADE()
        return self._proposal_params

    @property
    def proposal_log_prob_data(self):
        if self._proposal_log_prob_data is None:
            self._sample_eval_proposal_dist()
        return self._proposal_log_prob_data

    @property
    def proposal_samples(self):
        if self._proposal_samples is None:
            self._sample_eval_proposal_dist()
        return self._proposal_samples

    @property
    def proposal_log_prob_samples_proposal(self):
        if self._proposal_log_prob_samples_proposal is None:
            self._sample_eval_proposal_dist()
        return self._proposal_log_prob_samples_proposal

    @property
    def unnorm_log_prob_data(self):
        if self._unnorm_log_prob_data is None:
            self._eval_energy_function()
        return self._unnorm_log_prob_data

    @property
    def unnorm_log_prob_samples(self):
        if self._unnorm_log_prob_samples is None:
            self._eval_energy_function()
        return self._unnorm_log_prob_samples

    @property
    def norm_constants_est(self):
        if self._norm_constants_est is None:
            self._est_norm_constants_and_log_prob()
        return self._norm_constants_est

    @property
    def log_prob_est_data(self):
        if self._log_prob_est_data is None:
            self._est_norm_constants_and_log_prob()
        return self._log_prob_est_data

    def _process_inputs_MADE(self):
        self.made_dropout_p_tf = tf.placeholder_with_default(self.dropout_p_MADE, ())
        made_outputs = ResMADE(
            self.x_batch,
            n_out=3 * self.n_proposal_mixture_comps + self.n_context_units_energy_net,
            n_residual_blocks=self.n_res_blocks_MADE,
            hidden_units=self.n_hidden_units_MADE,
            activation=get_activation(self.activation_MADE),
            dropout_p=self.made_dropout_p_tf,
            final_act=self.final_act_MADE,
        )
        energy_context = made_outputs[..., : self.n_context_units_energy_net]
        proposal_params = made_outputs[..., self.n_context_units_energy_net :]
        self._energy_context = energy_context
        self._proposal_params = proposal_params

    def _sample_eval_proposal_dist(self):
        proposal_logits = self.proposal_params[..., : self.n_proposal_mixture_comps]
        proposal_means = self.proposal_params[
            ..., self.n_proposal_mixture_comps : 2 * self.n_proposal_mixture_comps
        ]
        proposal_scales = (
            tf.nn.softplus(
                self.proposal_params[..., 2 * self.n_proposal_mixture_comps :]
            )
            + self.proposal_comp_scale_min
        )
        tfd = tf.contrib.distributions
        components_dist = tfd.Normal(loc=proposal_means, scale=proposal_scales)
        proposal_dist = tfd.MixtureSameFamily(
            mixture_distribution=tfd.Categorical(logits=proposal_logits),
            components_distribution=components_dist,
        )
        self._proposal_log_prob_data = tf.reduce_sum(
            proposal_dist.log_prob(self.x_batch), axis=-1
        )
        proposal_samples = proposal_dist.sample(self.n_importance_samples)
        proposal_log_prob_samples_proposal = proposal_dist.log_prob(proposal_samples)

        # Stop gradients to prevent backprop wrt proposal samples
        self._proposal_samples = tf.stop_gradient(
            tf.transpose(proposal_samples, [1, 2, 0])
        )
        self._proposal_log_prob_samples_proposal = tf.stop_gradient(
            tf.transpose(proposal_log_prob_samples_proposal, [1, 2, 0])
        )

    def _eval_energy_function(self):
        self.energy_net_dropout_p_tf = tf.placeholder_with_default(
            self.dropout_p_energy_net, ()
        )
        x_batch_cat_samples = tf.concat(
            [self.x_batch[..., None], self.proposal_samples], axis=-1
        )
        x_batch_cat_samples = tf.reshape(x_batch_cat_samples, [-1, 1])
        energy_context = self.energy_context[..., None, :]
        energy_context_tiled = tf.tile(
            energy_context, [1, 1, self.n_importance_samples + 1, 1]
        )
        energy_context_tiled = tf.reshape(
            energy_context_tiled, [-1, self.n_context_units_energy_net]
        )
        energy_net_outputs = contextual_res_net(
            x_batch_cat_samples,
            energy_context_tiled,
            hidden_dim=self.n_hidden_units_energy_net,
            n_res_blocks=self.n_res_blocks_energy_net,
            activation=get_activation(self.activation_energy_net),
            dropout_p=self.energy_net_dropout_p_tf,
        )
        self.data_dim = self.x_batch.get_shape().as_list()[-1]
        energy_net_outputs = tf.reshape(
            energy_net_outputs, [-1, self.data_dim, 1 + self.n_importance_samples]
        )
        unnorm_log_prob_data = energy_net_outputs[..., 0]
        unnorm_log_prob_samples = energy_net_outputs[..., 1:]
        return unnorm_log_prob_data, unnorm_log_prob_samples

    def _est_norm_constants_and_log_prob(self):
        unnorm_log_prob_data, unnorm_log_prob_samples = self._eval_energy_function()
        self._norm_constants_est = tf.reduce_logsumexp(
            unnorm_log_prob_samples - self.proposal_log_prob_samples_proposal, axis=-1
        ) - tf.log(tf.cast(self.n_importance_samples, tf.float32))
        self._log_prob_est_data = tf.reduce_sum(
            unnorm_log_prob_data - self.norm_constants_est, axis=-1
        )
