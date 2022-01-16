
import multiprocessing as mp
import os
import queue
from time import time, localtime, strftime
import yaml

from learner import Learner
from params import args
from remoter import MasterClient
from utils import makedirs_if_not_exist
from utils.logger import Logger

def main():

    with open(args.config_file) as f:
        dist_info = yaml.safe_load(f)
    pub_port = dist_info.get('pub_sub_port')
    pull_port = dist_info.get('push_pull_port')
    pub_queue = queue.Queue()
    pull_queue = queue.Queue()
    client = MasterClient(pub_port, pull_port, pub_queue, pull_queue)
    client.start()

    timestamp = strftime("%Y%m%d_%H%M%S", localtime())
    save_path = os.path.join(args.result_folder, 'models', timestamp)
    makedirs_if_not_exist(save_path)
    with open(os.path.join(save_path, 'args.yaml'), 'w') as f:
        yaml.dump(vars(args), f)

    dist_interval = args.dist_interval
    log_vars_interval = args.log_vars_interval
    save_interval = args.save_interval

    lr = args.init_learning_rate
    mini_lr = args.mini_learning_rate
    lr_decay_rate = args.lr_decay_rate
    lr_decay_interval = args.lr_decay_interval
    ent_weight = args.entropy_weight

    stats_q = mp.Queue()
    stats_path = os.path.join(args.result_folder, 'logs',timestamp)
    makedirs_if_not_exist(save_path)
    logger = Logger(stats_q, stats_path)
    stats_process = mp.Process(target=logger.record)
    stats_process.daemon = True
    stats_process.start()

    num_eps = args.num_eps
    master_learner = Learner(save_path=save_path)

    for ep in range(num_eps):
        start_time = time()
        if ep % dist_interval == 0:
            alg_vars = master_learner.get_vars()
            vars_and_ent = (alg_vars, ent_weight)
            pub_queue.put(vars_and_ent)

        curr_time = strftime("%Y-%m-%d %H:%M:%S", localtime())
        print('{}: train-{} done, \
            params of model {} ...' .format(
                curr_time,
                ep,
                alg_vars[0][0][:1][..., :6],
            )
        )

        grads, stats = pull_queue.get()
        master_learner.apply_grads(grads, lr)

        end_time = time()

        stats['learner/learning_rate'] = [lr]
        stats['learner/compute_grads_time_ms'] = [(end_time - start_time) * 1000.0]

        if ep % log_vars_interval == 0:
            stats_q.put((stats, ep, True, alg_vars))
        else:
            stats_q.put((stats, ep, False, 'dummy'))

        if ep % save_interval == 0:
            master_learner.save_alg(ep)

        if (ep + 1) % lr_decay_interval == 0:
            lr = max(mini_lr, lr * lr_decay_rate)


if __name__ == '__main__':
    main()
