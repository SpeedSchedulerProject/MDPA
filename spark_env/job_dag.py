import networkx as nx
import numpy as np

from params import args


class JobDAG(object):

    def __init__(self, nodes, adj_mat, name):
        # nodes: list of N nodes
        # adj_mat: N by N 0-1 adjacency matrix, e_ij = 1 -> edge from i to j
        assert len(nodes) == adj_mat.shape[0]
        assert adj_mat.shape[0] == adj_mat.shape[1]

        self.name = name

        self.nodes = nodes
        self.adj_mat = adj_mat

        self.num_nodes = len(self.nodes)
        self.num_nodes_done = 0

        # set of executors currently running on the job
        self.executors = set()

        # the computation graph needs to be a DAG
        assert is_dag(self.num_nodes, self.adj_mat)

        # get the set of schedule nodes
        self.frontier_nodes = set()
        for node in self.nodes:
            if node.is_schedulable():
                self.frontier_nodes.add(node)

        # assign job dag to node
        self.assign_job_dag_to_node()

        # dag is arrived
        self.arrived = False

        # dag is completed
        self.completed = False

        # dag start ime
        self.start_time = None

        # dag completion time
        self.completion_time = np.inf

        # map a executor number to an interval
        self.executor_interval_map = self.get_executor_interval_map()

    def assign_job_dag_to_node(self):
        for node in self.nodes:
            node.job_dag = self

    def get_executor_interval_map(self):
        executor_interval_map = {}
        entry_pt = 0

        # get the left most map
        for e in range(args.executor_data_point[0] + 1):
            executor_interval_map[e] = \
                (args.executor_data_point[0],
                 args.executor_data_point[0])

        # get the center map
        for i in range(len(args.executor_data_point) - 1):
            for e in range(args.executor_data_point[i] + 1,
                            args.executor_data_point[i + 1]):
                executor_interval_map[e] = \
                    (args.executor_data_point[i],
                     args.executor_data_point[i + 1])
            # at the data point
            e = args.executor_data_point[i + 1]
            executor_interval_map[e] = \
                (args.executor_data_point[i + 1],
                 args.executor_data_point[i + 1])

        # get the residual map
        if args.exec_cap > args.executor_data_point[-1]:
            for e in range(args.executor_data_point[-1] + 1,
                            args.exec_cap + 1):
                executor_interval_map[e] = \
                    (args.executor_data_point[-1],
                     args.executor_data_point[-1])

        return executor_interval_map

    def reset(self):
        for node in self.nodes:
            node.reset()
        self.num_nodes_done = 0
        self.executors = set()
        self.frontier_nodes = set()
        for node in self.nodes:
            if node.is_schedulable():
                self.frontier_nodes.add(node)
        self.arrived = False
        self.completed = False
        self.completion_time = np.inf

    def update_frontier_nodes(self, node):
        frontier_nodes_changed = False
        for child in node.child_nodes:
            if child.is_schedulable():
                if child.idx not in self.frontier_nodes:
                    self.frontier_nodes.add(child)
                    frontier_nodes_changed = True
        return frontier_nodes_changed


def is_dag(num_nodes, adj_mat):
    G = nx.DiGraph()
    G.add_nodes_from(range(num_nodes))
    for i in range(num_nodes):
        for j in range(num_nodes):
            if adj_mat[i, j] == 1:
                G.add_edge(i, j)
    return nx.is_directed_acyclic_graph(G)
