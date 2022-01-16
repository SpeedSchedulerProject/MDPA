import argparse


parser = argparse.ArgumentParser(description='SpeedScheduler')


# -- Basic --
parser.add_argument('--seed', type=int, default=42, help='random seed (default: 42)')
parser.add_argument('--query_type', type=str, default='tpch', help='query type (default: tpch)')
parser.add_argument('--job_folder', type=str, default='./spark_env/tpch/', help='job folder path (default: ./spark_env/tpch/)')
parser.add_argument('--result_folder', type=str, default='./results/', help='Result folder path (default: ./results)')
parser.add_argument('--num_saved_models', type=int, default=1000, help='Number of models to keep (default: 1000)')
parser.add_argument('--saved_model', type=str, default=None, help='Path to the saved tf checkpoint (default: None)')


# -- Environment --
parser.add_argument('--exec_cap', type=int, default=50, help='Number of total executors (default: 50)')
parser.add_argument('--num_init_dags', type=int, default=1, help='Number of initial DAGs in system (default: 10)')
parser.add_argument('--num_stream_dags', type=int, default=64, help='number of streaming DAGs (default: 64)')
parser.add_argument('--executor_data_point', type=int, default=[5, 10, 20, 40, 50, 60, 80, 100], nargs='+', help='Number of executors used in data collection')
parser.add_argument('--reward_scale', type=float, default=1e6, help='scale the reward to some normal values (default: 100000.0)')
parser.add_argument('--moving_delay', type=int, default=2000, help='Moving delay (milliseconds) (default: 2000)')
parser.add_argument('--warmup_delay', type=int, default=1000, help='Executor warming up delay (milliseconds) (default: 1000)')
parser.add_argument('--learn_obj', type=str, default='mean', help='Learning objective (default: mean)')
parser.add_argument('--stream_intervals', type=int, default=[25000], nargs='+', help='inter job arrival time in milliseconds (default: [25000])')


# -- TPC-H --
parser.add_argument('--tpch_size', type=str, default=['2g','5g','10g','20g','50g','80g','100g'], nargs='+', help='Numer of TPCH queries (default: [2g, 5g, 10g, 20g, 50g, 80g, 100g])')
parser.add_argument('--tpch_num', type=int, default=22, help='Numer of TPCH queries (default: 22)')


# -- Learning --
# setting
parser.add_argument('--num_actors', type=int, default=4, help='Number of parallel agents (default: 4)')
parser.add_argument('--num_eps', type=int, default=10000000, help='Number of training epochs (default: 10000000)')
parser.add_argument('--grads_norm_clip', type=float, default=5.0, help='clip threshold for gradients')
parser.add_argument('--is_clip_ratio', type=float, default=1.0, help='clip threshold of importance sampling ration for variance reduction')
parser.add_argument('--value_clip', type=float, default=1.0, help='clip threshold of critic value for variance reduction')
parser.add_argument('--eps', type=float, default=1e-8, help='epsilon (default: 1e-8)')
parser.add_argument('--gamma', type=float, default=0.99, help='discount factor (default: 0.99)')
parser.add_argument('--explore_times_per_vars', type=int, default=1, help='explore times every time an actor receive new vars (default: 1')
parser.add_argument('--prepare_times_per_train', type=int, default=1, help='prepare times per train (default: 1)')
parser.add_argument('--dist_interval', type=int, default=1, help='Interval for distributing Tensorflow model (default: 1)')
parser.add_argument('--log_vars_interval', type=int, default=16, help='Interval for logging status (default: 16)')
parser.add_argument('--save_interval', type=int, default=64, help='Interval for saving Tensorflow model (default: 64)')
# features
parser.add_argument('--node_input_dim', type=int, default=7, help='node input dimensions to graph embedding (default: 7)')
parser.add_argument('--job_input_dim', type=int, default=3, help='job input dimensions to graph embedding (default: 3)')
parser.add_argument('--hid_dims', type=int, default=[64, 64], nargs='+', help='hidden dimensions throughout graph embedding (default: [64, 64])')
parser.add_argument('--output_dim', type=int, default=8, help='output dimensions throughout graph embedding (default: 8)')
parser.add_argument('--max_depth', type=int, default=8, help='Maximum depth of root-leaf message passing (default: 8)')
#learning rate
parser.add_argument('--init_learning_rate', type=float, default=1e-4, help='starter_learning_rate (default: 0.0001)')
parser.add_argument('--mini_learning_rate', type=float, default=3e-5, help='minimum_learning_rate (default: 0.00003)')
parser.add_argument('--lr_decay_rate', type=float, default=0.99, help='decay rate for learning rate (default: 0.99)')
parser.add_argument('--lr_decay_interval', type=int, default=100, help='decay interval for learning rate (default: 100)')
# entropy weight
parser.add_argument('--entropy_weight', type=float, default=1e-4, help='entropy_weight (default: 0.0001)')
parser.add_argument('--value_weight', type=float, default=1.0, help='value_weight (default: 1.0)')
# weight decay rate
parser.add_argument('--weight_decay_rate',  type=float, default=1e-8, help='weight decay rate (default: 1e-8)')
# graph attention
parser.add_argument('--num_heads', type=int, default=4, help='Number of heads in attention mechanisem (default: 4)')


# -- Testing --
parser.add_argument('--num_exps', type=int, default=10, help='number of experiments (default: 10)')


# -- Multiple Learners -- #
parser.add_argument('-f', '--config_file', required=True, help='config file')


args, _ = parser.parse_known_args()
