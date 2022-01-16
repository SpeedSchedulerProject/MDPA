import os

import tensorflow as tf


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.FATAL)

assert '1.15' <= tf.__version__ < '2.0', 'tensorflow version: {} not supported'.format(tf.__version__)
