import numpy as np
# from setproctitle import setproctitle
from tensorboardX import SummaryWriter


class Logger(object):

    def __init__(self, status_q, log_path):
        self.status_q = status_q
        self.writer = SummaryWriter(log_path)

    def record(self):
        # setproctitle('python decima_logger')
        while True:
            stats, ep, vars_flag, alg_vars = self.status_q.get()
            if vars_flag:
                for i, alg_var in enumerate(alg_vars):
                    self.writer.add_histogram('alg_vars/var_{}'.format(i + 1), alg_var, ep)
            for tag, val in stats.items():
                if tag == 'is_ratio':
                    self.writer.add_histogram('is_ratio', val, ep)
                else:
                    self.writer.add_scalar(tag, np.mean(val), ep)
