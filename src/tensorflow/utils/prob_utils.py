import numpy as np
import tensorflow as tf


def sample_diag_gaussian(mean, log_var):
    """Sample diagonal Gaussian with specified params."""
    return mean + tf.exp(log_var / 2.0) * tf.random_normal(tf.shape(mean))


def log_prob_diag_gaussian(x, mean, log_var, sum_axis=None):
    """Evaluate log probability of input under diagonal Gaussian."""
    log_prob = -0.5 * (log_var + (x - mean) ** 2 / tf.exp(log_var) + tf.log(2 * np.pi))
    if sum_axis is not None:
        log_prob = tf.reduce_sum(log_prob, axis=sum_axis)
    return log_prob


def kl_div_diag_gaussian(q, sum_axis=-1):
    mu, log_var = q
    return -0.5 * tf.reduce_sum(1 + log_var - mu ** 2 - tf.exp(log_var), axis=sum_axis)
