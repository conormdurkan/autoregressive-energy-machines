""" Utilities for masked processing, includes ResMADE"""
import numpy as np
import tensorflow as tf


def _get_mask(in_features, out_features, autoregressive_features, mask_type="hidden"):
    """Create a kernel mask."""
    max_ = max(1, autoregressive_features - 1)
    min_ = min(1, autoregressive_features - 1)
    if mask_type == "input":
        in_degrees = np.arange(1, autoregressive_features + 1)
        out_degrees = np.arange(out_features) % max_ + min_
        mask = out_degrees[..., None] >= in_degrees
    elif mask_type == "output":
        in_degrees = np.arange(in_features) % max_ + min_
        out_degrees = np.repeat(
            np.arange(1, autoregressive_features + 1),
            out_features // autoregressive_features,
        )
        mask = out_degrees[..., None] > in_degrees
    else:
        in_degrees = np.arange(in_features) % max_ + min_
        out_degrees = np.arange(out_features) % max_ + min_
        mask = out_degrees[..., None] >= in_degrees
    return mask.astype("float32")


def masked_dense(
    inputs,
    units,
    num_blocks,
    mask_type="hidden",
    kernel_initializer=None,
    reuse=None,
    name=None,
    activation=None,
    *args,
    **kwargs
):
    """Masked dense layer for causal architectures.
    
    Arguments:
        inputs -- Tensor inputs
        units -- Number of output units
        num_blocks -- Number of autoregressive blocks
    
    Keyword Arguments:
        mask_type -- Mask type. input, hidden or output (default: {hidden})
        kernel_initializer -- Kernel initializer (default: {None})
        reuse -- Reuse layer parameters in scope (default: {None})
        name -- Name for variables (default: {None})
        activation -- Activation function (default: {None})
    
    Returns:
        output -- Tensor output of masked layer.
    """

    input_depth = inputs.shape.with_rank_at_least(1)[-1].value
    if input_depth is None:
        raise NotImplementedError(
            "Rightmost dimension must be known prior to graph execution."
        )

    mask = _get_mask(input_depth, units, num_blocks, mask_type).T

    if kernel_initializer is None:
        kernel_initializer = tf.glorot_normal_initializer()

    def masked_initializer(shape, dtype=None, partition_info=None):
        return mask * kernel_initializer(shape, dtype, partition_info)

    with tf.name_scope(name, "masked_dense", [inputs, units, num_blocks]):
        layer = tf.layers.Dense(
            units,
            kernel_initializer=masked_initializer,
            kernel_constraint=lambda x: mask * x,
            name=name,
            dtype=inputs.dtype.base_dtype,
            _scope=name,
            _reuse=reuse,
            *args,
            **kwargs
        )
    return layer.apply(inputs)


def masked_residual_block(
    inputs, num_blocks, activation=tf.nn.relu, dropout_p=None, *args, **kwargs
):
    """Pre-activation masked residual block.

    Basic component of ResMADE.
    
    Arguments:
        inputs {Tensor} -- Tensor inputs
        num_blocks {int} -- Number of autoregressive blocks
    
    Keyword Arguments:
        activation {callable} -- Activation function (default: {tf.nn.relu})
        dropout_p {float} -- Dropout probability (default: {None})
    
    Returns:
        output -- Tensor output of residual block
    """

    input_depth = inputs.get_shape().as_list()[-1]

    # First pre-act residual layer
    residual = inputs
    residual = activation(residual)
    residual = masked_dense(
        residual,
        input_depth,
        num_blocks,
        kernel_initializer=tf.variance_scaling_initializer(
            scale=2.0, distribution="normal"
        ),
        *args,
        **kwargs
    )
    # Second preact residual layer (with dropout)
    residual = activation(residual)
    if dropout_p is not None:
        residual = tf.nn.dropout(residual, keep_prob=1 - dropout_p)
    residual = masked_dense(
        residual,
        input_depth,
        num_blocks,
        kernel_initializer=tf.variance_scaling_initializer(
            scale=0.1, distribution="normal"
        ),
        *args,
        **kwargs
    )
    return inputs + residual


def ResMADE(
    x,
    n_out=4,
    n_residual_blocks=2,
    hidden_units=256,
    activation=tf.nn.relu,
    dropout_p=None,
    final_act=True,
):
    """ResMADE autoregressive network for tabular data.
    
    Arguments:
        x -- Model inputs. [N, D] Tensor.
    
    Keyword Arguments:
        n_out -- Number of outputs per dimension. (default: {4})
        n_residual_blocks -- Number of residual blocks (default: {2})
        hidden_units -- Number of hidden units per residual block (default: {256})
        activation -- Activation function. (default: {tf.nn.relu})
        dropout_p -- Dropout probability (default: {None})
        final_act -- Use activation before final linear layer. (default: {True})
    
    Returns:
        output -- Output tensor [N, D, K] where K = n_out.
    """

    input_depth = x.get_shape().as_list()[-1]
    output = masked_dense(
        inputs=x, units=hidden_units, num_blocks=input_depth, mask_type="input"
    )
    for _ in range(n_residual_blocks):
        output = masked_residual_block(
            inputs=output,
            num_blocks=input_depth,
            activation=activation,
            dropout_p=dropout_p,
        )
    if final_act:
        output = activation(output)
    output = masked_dense(
        inputs=output,
        units=n_out * input_depth,
        num_blocks=input_depth,
        activation=None,
        mask_type="output",
        bias_initializer=tf.glorot_normal_initializer(),
    )
    return tf.reshape(output, [-1, input_depth, n_out])
