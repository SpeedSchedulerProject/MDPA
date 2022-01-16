import multiprocessing as mp
import os
import time
from collections import defaultdict

from attentional_decima.algorithm import AttentionalDecima

from params import args


class Learner(object):

    def __init__(self, save_path):
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'
        self._alg = AttentionalDecima(save_path)

    def get_vars(self):
        return self._alg.get_vars()

    def set_vars(self, alg_vars):
        self._alg.set_vars(alg_vars)

    def compute_grads(self, full_batch, ent_weight):
        ground_grads, all_loss, reward_loss, ent_loss, val_loss, is_ratio = \
            self._alg.compute_grads(full_batch, ent_weight)
        return ground_grads, all_loss, reward_loss, ent_loss, val_loss, is_ratio

    def apply_grads(self, ground_grads, lr):
        self._alg.apply_gradients(ground_grads, lr)

    def save_alg(self, ep):
        return self._alg.save_model(ep)
