"""
Compute the message passing path in O(num_total_nodes),
represent the path with sparse adjacency matrices (parent-
child pairs at each iteration) and frontier masks (aggregation
node points at each iteration)
"""

import numpy as np

from params import args
from tf_compat import tf
from utils.env_utils import OrderedSet
from utils.sparse_utils import SparseMat


class Postman(object):
    """
    Check if the set of DAGs changes and then compute the
    message passing path, to save computation
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self._job_dags = OrderedSet()
        self._gcn_mats = []
        self._gcn_masks = []
        self._dag_summ_backward_map = None
        self._max_depth = args.max_depth

    def get_msg_path(self, job_dags):
        if len(self._job_dags) != len(job_dags):
            job_dags_changed = True
        else:
            job_dags_changed = any(i is not j for i, j in zip(self._job_dags, job_dags))

        if job_dags_changed:
            self._job_dags = OrderedSet(job_dags)
            self._gcn_mats, self._gcn_masks = self._get_msg_path()
            self._dag_summ_backward_map = self._get_dag_summ_backward_map()
        unfinished_nodes_each_job = self._get_unfinished_nodes_summ_mat()

        return (
            self._gcn_mats,
            self._gcn_masks,
            self._dag_summ_backward_map,
            unfinished_nodes_each_job,
            job_dags_changed
        )

    def _get_msg_path(self):
        """
        matrix: parent-children relation in each message passing step
        mask: set of nodes doing message passing at each step
        """
        msg_mats, msg_masks = [], []

        for job_dag in self._job_dags:
            msg_mat, msg_mask = self._get_bottom_up_paths(job_dag)
            msg_mats.append(msg_mat)
            msg_masks.append(msg_mask)

        if len(self._job_dags) > 0:
            gcn_mats = self._absorb_sp_mats(msg_mats)
            gcn_masks = self._merge_masks(msg_masks)

        return gcn_mats, gcn_masks

    def _get_bottom_up_paths(self, job_dag):
        """
        The paths start from all leaves and end with
        frontier (parents all finished) unfinished nodes
        """
        num_nodes = job_dag.num_nodes

        msg_mat = []
        msg_mask = np.zeros([self._max_depth, num_nodes], dtype=np.float32)

        # get set of frontier nodes in the beginning
        # this is constrained by the message passing depth
        frontier = self._get_init_frontier(job_dag)
        msg_level = {}

        # initial nodes are all message passed
        for n in frontier:
            msg_level[n] = 0

        # pass messages
        for depth in range(self._max_depth):
            new_frontier = set()
            parent_visited = set()  # save some computation
            for n in frontier:
                for parent in n.parent_nodes:
                    if parent not in parent_visited:
                        curr_level = 0
                        children_all_in_frontier = True
                        for child in parent.child_nodes:
                            if child not in frontier:
                                children_all_in_frontier = False
                                break
                            if msg_level[child] > curr_level:
                                curr_level = msg_level[child]
                        # children all ready
                        if children_all_in_frontier:
                            if parent not in msg_level or curr_level + 1 > msg_level[parent]:
                                # parent node has deeper message passed
                                new_frontier.add(parent)
                                msg_level[parent] = curr_level + 1
                        # mark parent as visited
                        parent_visited.add(parent)

            if len(new_frontier) == 0:
                break  # some graph is shallow

            # assign parent-child path in current iteration
            sp_mat = SparseMat(dtype=np.float32, shape=[num_nodes, num_nodes])
            for n in new_frontier:
                for child in n.child_nodes:
                    sp_mat.add(row=n.idx, col=child.idx, data=1.0)
                msg_mask[depth, n.idx] = 1.0
            msg_mat.append(sp_mat)

            # Note: there might be residual nodes that
            # can directly pass message to its parents
            # it needs two message passing steps
            # (e.g., TPCH-17, node 0, 2, 4)
            for n in frontier:
                parents_all_in_frontier = True
                for p in n.parent_nodes:
                    if not p in msg_level:
                        parents_all_in_frontier = False
                        break
                if not parents_all_in_frontier:
                    new_frontier.add(n)

            # start from new frontier
            frontier = new_frontier

        # deliberately make dimension the same, for batch processing
        for _ in range(depth, self._max_depth):
            msg_mat.append(SparseMat(dtype=np.float32, shape=[num_nodes, num_nodes]))

        return msg_mat, msg_mask

    def _get_init_frontier(self, job_dag):
        """
        Get the initial set of frontier nodes, based on the depth
        """
        sources = set(job_dag.nodes)

        for d in range(self._max_depth):
            new_sources = set()
            for n in sources:
                if len(n.child_nodes) == 0:
                    new_sources.add(n)
                else:
                    new_sources.update(n.child_nodes)
            sources = new_sources

        frontier = sources
        return frontier


    def _absorb_sp_mats(self, in_mats):
        """
        Merge multiple sparse matrices to 
        a giant one on its diagonal
        e.g., 
        
        [0, 1, 0]    [0, 1, 0]    [0, 0, 1]
        [1, 0, 0]    [0, 0, 1]    [0, 1, 0]
        [0, 0, 1]    [1, 0, 0]    [0, 1, 0]
        
        to 
        
        [0, 1, 0]   ..  ..   ..  ..
        [1, 0, 0]   ..  ..   ..  ..
        [0, 0, 1]   ..  ..   ..  ..
        ..   ..  [0, 1, 0]   ..  ..
        ..   ..  [0, 0, 1]   ..  ..
        ..   ..  [1, 0, 0]   ..  ..
        ..   ..  ..   ..  [0, 0, 1]
        ..   ..  ..   ..  [0, 1, 0]
        ..   ..  ..   ..  [0, 1, 0]
        where ".." are all zeros
        depth is on the 3rd dimension,
        which is orthogonal to the planar 
        operations above
        output SparseTensorValue from tensorflow
        """
        sp_mats = []

        for d in range(self._max_depth):
            row_idx = []
            col_idx = []
            data = []
            base = 0
            for m in in_mats:
                row_idx.append(m[d].get_row() + base)
                col_idx.append(m[d].get_col() + base)
                data.append(m[d].get_data())
                base += m[d].shape[0]

            row_idx = np.hstack(row_idx)
            col_idx = np.hstack(col_idx)
            data = np.hstack(data)

            merged_sp_mat = SparseMat(dtype=np.float32, shape=[base, base], row=row_idx, col=col_idx, data=data)
            sp_mats.append(merged_sp_mat)

        return sp_mats

    def _merge_masks(self, masks):
        """
        e.g.,
        [0, 1, 0]  [0, 1]  [0, 0, 0, 1]
        [0, 0, 1]  [1, 0]  [1, 0, 0, 0]
        [1, 0, 0]  [0, 0]  [0, 1, 1, 0]
        to
        a list of
        [0, 1, 0, 0, 1, 0, 0, 0, 1]^T,
        [0, 0, 1, 1, 0, 1, 0, 0, 0]^T,
        [1, 0, 0, 0, 0, 0, 1, 1, 0]^T
        Note: mask dimension d is pre-determined
        """
        merged_masks = []

        for d in range(self._max_depth):

            merged_mask = []
            for mask in masks:
                merged_mask.append(mask[d:d + 1].transpose())

            if len(merged_mask) > 0:
                merged_mask = np.vstack(merged_mask)

            merged_masks.append(merged_mask)

        return merged_masks

    def _get_dag_summ_backward_map(self):
        # compute backward mapping from node idx to dag idx
        total_num_nodes = sum([job_dag.num_nodes for job_dag in self._job_dags])
        num_dags = len(self._job_dags)
        dag_summ_backward_map = np.zeros([total_num_nodes, num_dags], dtype=np.float32)

        base = 0
        j_idx = 0
        for job_dag in self._job_dags:
            for node in job_dag.nodes:
                dag_summ_backward_map[base + node.idx, j_idx] = 1.0
            base += job_dag.num_nodes
            j_idx += 1

        return dag_summ_backward_map[np.newaxis]

    def _get_unfinished_nodes_summ_mat(self):
        # compute backward mapping from node idx to dag idx

        # 1. connect the unfinished nodes to "summarized node"
        # 2. silent out all the nodes that's already done
        # O(num_total_nodes)

        total_num_nodes = np.sum([job_dag.num_nodes for job_dag in self._job_dags])
        num_dags = len(self._job_dags)
        summ_shape = [num_dags, total_num_nodes]

        unfinished_nodes_each_job = np.zeros([num_dags, total_num_nodes], dtype=np.float32)

        base = 0
        j_idx = 0
        for job_dag in self._job_dags:
            for node in job_dag.nodes:
                if not node.tasks_all_done:
                    unfinished_nodes_each_job[j_idx, base + node.idx] = 1.0
            base += job_dag.num_nodes
            j_idx += 1

        return unfinished_nodes_each_job[np.newaxis]
