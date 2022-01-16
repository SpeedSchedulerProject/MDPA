import math
import os
import time
from collections import defaultdict

import numpy as np
# from setproctitle import setproctitle

from params import args
from attentional_decima.algorithm import AttentionalDecima
from spark_env.env import Environment
from spark_env.job_dag import JobDAG
from spark_env.node import Node
from utils.alg_utils import discount, truncate_experiences
from utils.msg_path import Postman
from utils.data_buffer import DataBuffer
from utils.shared_buffer import SharedBuffer
from utils.sparse_utils import expand_sp_mats


class Actor(object):

    def __init__(self, actor_id, recv_vars_q, send_traj_q, data_path, shared_path, explore_times=1):
        # os.environ['CUDA_VISIBLE_DEVICES'] = str(actor_id % 3 + 1)
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

        self._actor_id = actor_id
        self._shared_buf = SharedBuffer(shared_path, recv_vars_q)
        self._data_buf = DataBuffer(send_traj_q, data_path)
        self._explore_times = explore_times

        self._env = Environment(self._actor_id)
        self._alg = AttentionalDecima()

        self._node_input_dim = args.node_input_dim
        self._job_input_dim = args.job_input_dim
        self._exec_cap = args.exec_cap
        self._executor_levels = list(range(1, args.exec_cap + 1))

        self._eps = args.eps
        self._gamma = args.gamma

        # for computing and storing message passing path
        self._postman = Postman()

    def start(self):
        # setproctitle('python decima_actor-{}'.format(self._actor_id))
        while True:
            alg_vars = self._shared_buf.recv()
            self._alg.set_vars(alg_vars)
            # print('actor-{} set vars done'.format(self.actor_id))
            for _ in range(self._explore_times):
                self.explore()
                # print('actor-{} done one episode'.format(self.actor_id))

    def explore(self):
        start_time = time.time()
        obs = self._env.reset()
        job_dags, source_job, num_source_exec, frontier_nodes, exec_commit, moving_executors, action_map = obs
        done = False
        batch_breakpoint = True
        self._traj = defaultdict(list)
        self._stats = defaultdict(list)

        while not done:
            node_inputs, job_inputs, exec_map, job_idx_map = self._translate_state(
                job_dags, source_job, num_source_exec, exec_commit, moving_executors
            )
            node_valid_mask, job_valid_mask = self._get_valid_masks(
                job_dags, frontier_nodes, source_job, num_source_exec, exec_map, action_map
            )

            gcn_mats, gcn_masks, dag_summ_backward_map, unfinished_nodes_each_job, job_dags_changed = \
                self._postman.get_msg_path(job_dags)

            node_act, job_acts, node_act_probs, job_act_probs, value = self._alg.predict(
                node_inputs,
                job_inputs,
                node_valid_mask,
                job_valid_mask,
                gcn_mats,
                gcn_masks,
                unfinished_nodes_each_job,
                dag_summ_backward_map
            )

            # parse node action
            node = action_map[node_act[0]]
            # find job index based on node
            job_idx = job_idx_map[node.job_dag]

            # job_act should be valid
            assert job_valid_mask[0, job_idx, job_acts[0, job_idx]] == 1.0
            # find out the executor limit decision
            if node.job_dag is source_job:
                agent_exec_act = self._executor_levels[job_acts[0, job_idx]] - exec_map[node.job_dag] + num_source_exec
            else:
                agent_exec_act = self._executor_levels[job_acts[0, job_idx]] - exec_map[node.job_dag]
            # parse job limit action
            use_exec = min(
                node.num_tasks - \
                    node.next_task_idx - \
                    exec_commit.node_commit[node] - \
                    moving_executors.count(node),
                agent_exec_act,
                num_source_exec
            )

            # for storing the action vector in experience
            node_act_vec = np.zeros(node_act_probs.shape, dtype=np.float32)
            node_act_vec[0, node_act[0]] = 1.0
            # for storing job index
            job_act_vec = np.zeros(job_act_probs.shape, dtype=np.float32)
            job_act_vec[0, job_idx, job_acts[0, job_idx]] = 1.0
            old_act_logp = math.log(
                node_act_probs[0, node_act[0]] * job_act_probs[0, job_idx, job_acts[0, job_idx]] + self._eps
            )

            # store experience
            self._traj['node_inputs'].append(node_inputs)
            self._traj['job_inputs'].append(job_inputs)
            self._traj['node_valid_mask'].append(node_valid_mask)
            self._traj['job_valid_mask'].append(job_valid_mask)
            self._traj['unfinished_nodes_each_job'].append(unfinished_nodes_each_job)
            if job_dags_changed:
                self._traj['gcn_mats'].append(gcn_mats)
                self._traj['gcn_masks'].append(gcn_masks)
                self._traj['dag_summ_backward_map'].append(dag_summ_backward_map)
            self._traj['node_act_vec'].append(node_act_vec)
            self._traj['job_act_vec'].append(job_act_vec)
            self._traj['old_act_logp'].append(old_act_logp)
            self._traj['job_state_change'].append(job_dags_changed)
            self._traj['batch_breakpoint'].append(batch_breakpoint)
            self._traj['old_value'].append(value)

            obs, reward, done = self._env.step(node, use_exec)
            self._traj['reward'].append(reward)
            batch_breakpoint = False
            while True:
                job_dags, source_job, num_source_exec, frontier_nodes, exec_commit, moving_executors, action_map = obs
                if not done and len(frontier_nodes) == 0:
                    if len(job_dags) == 0:
                        batch_breakpoint = True
                    obs, reward, done = self._env.step(None, num_source_exec)
                    self._traj['reward'][-1] += reward
                else:
                    break
        end_time = time.time()

        self._stats['actor/explore_time_ms'] = [(end_time - start_time) * 1000.0]
        self._stats['actor/wall_time'] = [self._env.wall_time.curr_time]
        self._stats['actor/episode_length'] = [len(self._traj['reward'])]
        completion_time_of_jobs = []
        for job_dag in self._env.finished_job_dags:
            completion_time_of_jobs.append(job_dag.completion_time - job_dag.start_time)
        self._stats['actor/average_jct'] = completion_time_of_jobs
        full_batch = self._process_traj()
        self._data_buf.send((full_batch, self._stats, self._actor_id))

    def _process_traj(self):
        reward = np.asarray(self._traj['reward']).astype(np.float32)
        reward = np.expand_dims(reward, axis=-1)
        old_act_logp = np.asarray(self._traj['old_act_logp']).astype(np.float32)
        old_act_logp = np.expand_dims(old_act_logp, axis=-1)
        old_value = np.concatenate(self._traj['old_value'])

        batch_breakpoints = truncate_experiences(self._traj['batch_breakpoint'])
        done = np.zeros([batch_breakpoints[-1], 1], dtype=np.float32)
        sum_batch_reward = []
        for i in range(len(batch_breakpoints) - 1):
            batch_start = batch_breakpoints[i]
            batch_end = batch_breakpoints[i + 1]
            done[batch_end - 1] = 1.0
            batch_reward = reward[batch_start: batch_end]
            sum_batch_reward.append(np.sum(batch_reward))
        self._stats['actor/sum_batch_reward'] = sum_batch_reward

        old_value_next = np.concatenate([old_value[1:], np.zeros([1, 1], dtype=np.float32)])
        delta = reward + done * self._gamma * old_value_next - old_value
        adv = delta
        for i in range(len(adv) -2, -1, -1):
            adv[i] += done[i] * self._gamma * 0.95 * adv[i + 1]
        rtg = adv + old_value

        mini_batch_points = truncate_experiences(self._traj['job_state_change'])
        full_batch = []
        for i in range(len(mini_batch_points) - 1):
            # use a piece of experience
            ba_start = mini_batch_points[i]
            ba_end = mini_batch_points[i + 1]
            batch_size = ba_end - ba_start
            mini_batch = {}

            mini_batch['node_inputs'] = np.concatenate(self._traj['node_inputs'][ba_start: ba_end])
            mini_batch['job_inputs'] = np.concatenate(self._traj['job_inputs'][ba_start: ba_end])
            mini_batch['node_valid_mask'] = np.concatenate(self._traj['node_valid_mask'][ba_start: ba_end])
            mini_batch['job_valid_mask'] = np.concatenate(self._traj['job_valid_mask'][ba_start: ba_end])
            mini_batch['node_act_vec'] = np.concatenate(self._traj['node_act_vec'][ba_start: ba_end])
            mini_batch['job_act_vec'] = np.concatenate(self._traj['job_act_vec'][ba_start: ba_end])

            mini_batch['old_act_logp'] = old_act_logp[ba_start: ba_end]
            mini_batch['reward_to_go'] = rtg[ba_start: ba_end]
            mini_batch['old_value'] = old_value[ba_start: ba_end]

            # expand unfinished_nodes_each_job
            unfinished_nodes_each_job_mats = self._traj['unfinished_nodes_each_job'][ba_start:ba_end]
            expanded_unfinished_nodes_each_job = np.concatenate(unfinished_nodes_each_job_mats)
            mini_batch['unfinished_nodes_each_job'] = expanded_unfinished_nodes_each_job

            # expand sparse adj_mats
            gcn_mats = self._traj['gcn_mats'][i]
            expanded_gcn_mats = expand_sp_mats(gcn_mats, batch_size)
            mini_batch['gcn_mats'] = expanded_gcn_mats

            # expand gcn_masks
            # (on the dimension according to extended adj_mat)
            gcn_masks = self._traj['gcn_masks'][i]
            expanded_gcn_masks = [np.tile(gcn_mask, [batch_size, 1]) for gcn_mask in gcn_masks]
            mini_batch['gcn_masks'] = expanded_gcn_masks

            # expand dag_summ_backward_map
            dag_summ_backward_map = self._traj['dag_summ_backward_map'][i]
            expanded_dag_summ_backward_map = np.tile(dag_summ_backward_map, [batch_size, 1, 1])
            mini_batch['dag_summ_backward_map'] = expanded_dag_summ_backward_map

            full_batch.append(mini_batch)

        return full_batch

    def _get_valid_masks(self, job_dags, frontier_nodes, source_job, num_source_exec, exec_map, action_map):

        job_valid_mask = np.zeros([len(job_dags), self._exec_cap], dtype=np.float32)

        job_valid = {}  # if job is saturated, don't assign node

        job_idx = 0
        for job_dag in job_dags:
            # new executor level depends on the source of executor
            if job_dag is source_job:
                least_exec_amount = exec_map[job_dag] - num_source_exec + 1
                # +1 because we want at least one executor for this job
            else:
                least_exec_amount = exec_map[job_dag] + 1
                # +1 because of the same reason above

            assert least_exec_amount > 0
            assert least_exec_amount <= self._exec_cap + 1

            # find the index for first valid executor limit
            exec_level_idx = least_exec_amount - 1

            if exec_level_idx >= self._exec_cap:
                job_valid[job_dag] = False
            else:
                job_valid[job_dag] = True

            for l in range(exec_level_idx, self._exec_cap):
                job_valid_mask[job_idx, l] = 1.0

            job_idx += 1

        total_num_nodes = sum([job_dag.num_nodes for job_dag in job_dags])

        node_valid_mask = np.zeros([total_num_nodes,], dtype=np.float32)

        for node in frontier_nodes:
            if job_valid[node.job_dag]:
                act = action_map.inverse_map[node]
                node_valid_mask[act] = 1.0

        return node_valid_mask[np.newaxis], job_valid_mask[np.newaxis]

    def _translate_state(self, job_dags, source_job, num_source_exec, exec_commit, moving_executors):
        """Translate the observation to matrix form."""

        # sort out the exec_map
        exec_map = {}
        for job_dag in job_dags:
            exec_map[job_dag] = len(job_dag.executors)

        # count in moving executors
        for node in moving_executors.moving_executors.values():
            exec_map[node.job_dag] += 1
        # count in executor commit
        for s in exec_commit.commit:
            if isinstance(s, JobDAG):
                j = s
            elif isinstance(s, Node):
                j = s.job_dag
            elif s is None:
                j = None
            else:
                print('source', s, 'unknown')
                exit(1)
            for n in exec_commit.commit[s]:
                if n is not None and n.job_dag != j:
                    exec_map[n.job_dag] += exec_commit.commit[s][n]

        # compute total number of nodes
        total_num_nodes = sum([job_dag.num_nodes for job_dag in job_dags])

        # job and node inputs to feed
        node_inputs = np.zeros([total_num_nodes, self._node_input_dim], dtype=np.float32)
        job_inputs = np.zeros([len(job_dags), self._job_input_dim], dtype=np.float32)

        # gather inputs
        node_idx = 0
        job_idx = 0
        job_idx_map = {}
        for job_dag in job_dags:
            # number of executors in the job
            job_inputs[job_idx, 0] = exec_map[job_dag] / 20.0
            # the current executor belongs to this job or not
            if job_dag is source_job:
                job_inputs[job_idx, 1] = 2.0
            else:
                job_inputs[job_idx, 1] = -2.0
            # number of source executors
            job_inputs[job_idx, 2] = num_source_exec / 20.0
            for node in job_dag.nodes:
                # copy the feature from job_input first
                node_inputs[node_idx, :3] = job_inputs[job_idx, :3]
                # work on the node
                node_inputs[node_idx, 3] = \
                    (node.num_tasks - node.next_task_idx) *  node.tasks[-1].duration / 100000.0
                # number of tasks left
                node_inputs[node_idx, 4] = (node.num_tasks - node.next_task_idx) / 200.0
                #node in-degree
                node_inputs[node_idx, 5] = len(node.parent_nodes) / 3.0
                # node out-degree
                node_inputs[node_idx, 6] = len(node.child_nodes) / 2.0
                node_idx += 1
            job_idx_map[job_dag] = job_idx
            job_idx += 1

        return node_inputs[np.newaxis], job_inputs[np.newaxis], exec_map, job_idx_map
