import numpy as np

from tf_compat import tf


def build_mlp(mlp_input, hid_dims, output_size, prefix, act_fn=tf.nn.elu, output_activation=None):
    x = mlp_input
    for i, hid_dim in enumerate(hid_dims):
        x = tf.layers.Dense(hid_dim, activation=act_fn, name='{}_mlp_{}'.format(prefix, i))(x)
    out = tf.layers.Dense(output_size, activation=output_activation, name='{}_mlp_out'.format(prefix))(x)
    return out


def huber_loss(y_true, y_pred, delta=1.0):
    residual = tf.abs(y_true - y_pred)
    condition = tf.less(residual, delta)
    small_res = 0.5 * tf.square(residual)
    large_res = delta * residual - 0.5 * tf.square(delta)
    return tf.where_v2(condition, small_res, large_res)
