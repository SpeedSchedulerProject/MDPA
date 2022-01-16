import numpy as np

from attentional_decima.algorithm import AttentionalDecima
from spark_env.env import Environment
from spark_env.job_dag import JobDAG
from spark_env.node import Node
from utils.msg_path import Postman
from utils.viz_utils import visualize_executor_usage
from params import args


class TestAgent(object):

    def __init__(self):
        self._alg = AttentionalDecima()
        self._node_input_dim = args.node_input_dim
        self._job_input_dim = args.job_input_dim
        self._exec_cap = args.exec_cap
        self._executor_levels = list(range(1, args.exec_cap + 1))
        # for computing and storing message passing path
        self._postman = Postman()

    def reset(self):
        self._postman.reset()

    def get_action(self, obs):

        job_dags, source_job, num_source_exec, frontier_nodes, exec_commit, moving_executors, action_map = obs

        node_inputs, job_inputs, exec_map, job_idx_map = self._translate_state(
            job_dags, source_job, num_source_exec, exec_commit, moving_executors
        )
        node_valid_mask, job_valid_mask = self._get_valid_masks(
            job_dags, frontier_nodes, source_job, num_source_exec, exec_map, action_map
        )

        gcn_mats, gcn_masks, dag_summ_backward_map, unfinished_nodes_each_job, job_dags_changed = \
            self._postman.get_msg_path(job_dags)

        node_act, job_acts, node_act_probs, job_act_probs = self._alg.predict(
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

        return node, use_exec

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


def main():
    env = Environment(0)
    agent = TestAgent()
    exp_executor_usages, exp_job_durations = [], []
    for exp in range(args.num_exp):
        print('Experiment {} of {}'.format(exp + 1, args.num_exp))

        agent.reset()
        env.seed(exp)
        obs = env.reset()

        done = False

        while True:
            job_dags, source_job, num_source_exec, frontier_nodes, exec_commit, moving_executors, action_map = obs
            if not done and len(frontier_nodes) == 0:
                obs, reward, done = env.step(None, num_source_exec)
            else:
                if done:
                    break
                else:
                    node, use_exec = agent.get_action(obs)
                    obs, reward, done = env.step(node, use_exec)

        executor_usages, job_durations = visualize_executor_usage(
            env.finished_job_dags,
            args.result_folder + '/pngs/' + 'visualization_exp_' + str(exp) + '.png'
        )
        exp_executor_usages.append(np.array(executor_usages))
        exp_job_durations.append(np.array(job_durations))

    np.savez(
        '{}/stats/stream_{}_num_{}_exec_{}.npz'.format(
            args.result_folder, args.stream_intervals[0], args.num_init_dags, args.exec_cap
        ),
        executor_usages=exp_executor_usages,
        job_durations=exp_job_durations
    )


if __name__ == "__main__":
    main()
