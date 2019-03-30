""" Energy networks for AEM """
import tensorflow as tf


def contextual_res_net(
    x,
    context,
    n_res_blocks=2,
    hidden_dim=128,
    activation=tf.nn.relu,
    dropout_p=None,
    final_act=True,
):
    """ Fully connected pre-activation resnet.

    Arguments:
        x -- Tensor input for energy evaluation
        context -- Context tensor from AEM
    
    Keyword Arguments:
        n_res_blocks -- Number of residual blocks (default: {2})
        hidden_dim -- Number of units in hidden layers (default: {128})
        activation -- Activation function (default: {tf.nn.relu})
        dropout_p -- Dropout probability (default: {None})
        final_act -- Apply activation before final linear layer (default: {True})
    
    Returns:
        neg_energy_x -- Negative energy of x
    """

    h = tf.layers.dense(tf.concat([x, context], axis=-1), hidden_dim)
    for _ in range(n_res_blocks):
        # First pre-act layer
        residual = activation(h)
        residual = tf.layers.dense(
            residual,
            hidden_dim,
            kernel_initializer=tf.variance_scaling_initializer(
                scale=2.0, distribution="normal"
            ),
        )
        # Second pre-act layer (with dropout)
        residual = activation(residual)
        if dropout_p is not None:
            residual = tf.nn.dropout(residual, keep_prob=1 - dropout_p)
        residual = tf.layers.dense(
            residual,
            hidden_dim,
            kernel_initializer=tf.variance_scaling_initializer(
                scale=0.1, distribution="normal"
            ),
        )
        h += residual
    if final_act:
        h = activation(h)
    return -tf.nn.softplus(tf.layers.dense(h, 1))
