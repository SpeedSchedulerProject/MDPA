import multiprocessing as mp
import os
import queue
import time
import threading
import yaml
from collections import defaultdict
from time import time, localtime, strftime, sleep

import numpy as np
# from setproctitle import setproctitle

from actor import Actor
from learner import Learner
from remoter import WorkerClient
from params import args
from utils import makedirs_if_not_exist
from utils.alg_utils import MovingAverage
from utils.data_buffer import DataBuffer
from utils.shared_buffer import SharedBuffer
from utils.logger import Logger


vars_and_ent = None


def create_actor(actor_id, recv_leaner_q, send_learner_q, data_path, shared_path, explore_times_per_vars):
    acotr = Actor(actor_id, recv_leaner_q, send_learner_q, data_path, shared_path, explore_times_per_vars)
    acotr.start()


def dist_vars(shared_path, sub_queue, learner_queue, send_actors_q):
    shared_buf = SharedBuffer(shared_path)
    while True:
        vars_and_ent = sub_queue.get()
        alg_vars, ent_weight = vars_and_ent
        obj_id = shared_buf.send(alg_vars)
        learner_queue.put((obj_id, ent_weight))
        for _, send_actor_q in send_actors_q.items():
            send_actor_q.put(obj_id)


def main():
    # setproctitle('python decima_learner')

    with open(args.config_file) as f:
        dist_info = yaml.safe_load(f)
    master_ip = dist_info.get('master_ip')
    sub_port = dist_info.get('pub_sub_port')
    push_port = dist_info.get('push_pull_port')
    sub_queue = queue.Queue()
    push_queue = queue.Queue()
    client = WorkerClient(sub_port, push_port, sub_queue, push_queue, master_ip)
    client.start()

    traj_q = mp.Queue()
    data_buf = DataBuffer(traj_q)
    shared_buf = SharedBuffer()
    data_path = data_buf.get_path()
    shared_path = shared_buf.get_path()

    explore_times_per_vars = args.explore_times_per_vars
    prepare_times_per_train = args.prepare_times_per_train

    actor_processes = {}
    send_actors_q = {}
    for actor_id in range(args.num_actors):
        send_actor_q = mp.Queue()
        p = mp.Process(
            target=create_actor,
            args=(actor_id, send_actor_q, traj_q, data_path, shared_path, explore_times_per_vars)
        )
        send_actors_q[actor_id] = send_actor_q
        actor_processes[actor_id] = p
    for actor_id, actor_porcess in actor_processes.items():
        actor_porcess.daemon = True
        actor_porcess.start()

    # algorithm is saved by master worker, so a dummy_path is used there.
    worker_learner = Learner(save_path='dummy_path')

    learner_queue = queue.Queue()
    dist_task = threading.Thread(
        target=dist_vars,
        args=(shared_path, sub_queue, learner_queue, send_actors_q)
    )
    dist_task.setDaemon(True)
    dist_task.start()
    vars_and_ent = None

    def get_newest_vars_and_ent(learner_queue):
        nonlocal vars_and_ent
        while True:
            try:
                vars_and_ent = learner_queue.get(block=True, timeout=0.0001)
            except:
                pass

    update_learner_task = threading.Thread(
        target=get_newest_vars_and_ent,
        args=(learner_queue,)
    )
    update_learner_task.setDaemon(True)
    update_learner_task.start()

    count = 0
    while True:
        start_time = time()

        full_batch = []
        stats = defaultdict(list)
        for _ in range(prepare_times_per_train):
            train_batch, actor_stats, actor_id = data_buf.recv()
            full_batch.extend(train_batch)
            for k, v in actor_stats.items():
                stats[k].extend(v)

        vars_id, ent_weight = vars_and_ent
        alg_vars = shared_buf.get(vars_id)
        worker_learner.set_vars(alg_vars)
        ground_grads, all_loss, reward_loss, ent_loss, val_loss, is_ratio = \
            worker_learner.compute_grads(full_batch, ent_weight)

        end_time = time()

        stats.update(actor_stats)
        stats['learner/compute_grads_time_ms'] = [(end_time - start_time) * 1000.0]
        stats['learner/all_loss'] = [all_loss]
        stats['learner/reward_loss'] = [reward_loss]
        stats['learner/entorpy_loss'] = [ent_loss]
        stats['learner/value_loss'] = [val_loss]
        stats['learner/entorpy_weight'] = [ent_weight]
        stats['is_ratio'] = is_ratio

        grads_and_stats = (ground_grads, stats)
        push_queue.put(grads_and_stats)

        count += 1
        curr_time = strftime("%Y-%m-%d %H:%M:%S", localtime())
        print('{}: worker working {} times'.format(curr_time, count))


if __name__ == '__main__':
    main()
