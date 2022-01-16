import pickle
import threading
import zmq
# from pyarrow import serialize, deserialize


class MasterClient():

    def __init__(self, pub_port, pull_port, pub_queue, pull_queue):
        self.__pub_socket = self.__create_publisher(pub_port)
        self.__pull_socket = self.__create_puller(pull_port)
        self.__pub_queue = pub_queue
        self.__pull_queue = pull_queue
    
    def __create_publisher(self, pub_port):
        pub_context = zmq.Context.instance()
        pub_socket = pub_context.socket(zmq.PUB)
        pub_socket.bind('tcp://*:{}'.format(pub_port))
        return pub_socket

    def __create_puller(self, pull_port):
        pull_context = zmq.Context.instance()
        pull_socket = pull_context.socket(zmq.PULL)
        pull_socket.bind('tcp://*:{}'.format(pull_port))
        return pull_socket

    def __publish_vars(self):
        while True:
            vars_and_ent = self.__pub_queue.get()
            # print(vars_and_ent)
            serialized_vars_and_ent = pickle.dumps(vars_and_ent)
            self.__pub_socket.send(serialized_vars_and_ent)

    def __pull_grads(self):
        while True:
            serialized_grads_and_stats = self.__pull_socket.recv()
            grads_and_stats = pickle.loads(serialized_grads_and_stats)
            self.__pull_queue.put(grads_and_stats)

    def start(self):
        pub_task = threading.Thread(target=self.__publish_vars)
        pub_task.setDaemon(True)
        pub_task.start()

        pull_task = threading.Thread(target=self.__pull_grads)
        pull_task.setDaemon(True)
        pull_task.start()


class WorkerClient():

    def __init__(self, sub_port, push_port, sub_queue, push_queue, master_ip):
        self.__sub_socket = self.__create_subscriber(master_ip, sub_port)
        self.__push_socket = self.__create_pusher(master_ip, push_port)
        self.__sub_queue = sub_queue
        self.__push_queue = push_queue

    def __create_subscriber(self, master_ip, sub_port):
        sub_context = zmq.Context.instance()
        sub_socket = sub_context.socket(zmq.SUB)
        sub_socket.setsockopt(zmq.SUBSCRIBE, b'')
        sub_socket.connect('tcp://{}:{}'.format(master_ip, sub_port))
        return sub_socket

    def __create_pusher(self, master_ip, push_port):
        push_context = zmq.Context.instance()
        push_socket = push_context.socket(zmq.PUSH)
        push_socket.connect('tcp://{}:{}'.format(master_ip, push_port))
        return push_socket

    def __subscribe_vars(self):
        while True:
            serialized_vars = self.__sub_socket.recv()
            vars_and_ent = pickle.loads(serialized_vars)
            self.__sub_queue.put(vars_and_ent)

    def __push_grads(self):
        while True:
            grads_and_stats = self.__push_queue.get()
            serialized_grads_and_stats = pickle.dumps(grads_and_stats)
            self.__push_socket.send(serialized_grads_and_stats)

    def start(self):
        subscribe_task = threading.Thread(target=self.__subscribe_vars)
        subscribe_task.setDaemon(True)
        subscribe_task.start()

        push_task = threading.Thread(target=self.__push_grads)
        push_task.setDaemon(True)
        push_task.start()
