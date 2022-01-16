import os
import time
import multiprocessing as mp
from collections import deque
from subprocess import PIPE, Popen


from pyarrow import plasma


class DataBuffer(object):

    def __init__(self, queue, path="/tmp/plasma_data_{}".format(os.getpid()), size=20000000000, max_keep=32):
        self._q = queue
        self._path = path
        self._size = size
        self._kept_ids = deque()
        self._max_keep = max_keep

        self.__start()

    def __start(self):
        try:
            plasma.connect(self._path, num_retries=3)
        except:
            Popen("plasma_store -m {} -s {}".format(self._size , self._path), shell=True, stderr=PIPE)
            time.sleep(1.0)
        self._client = plasma.connect(self._path)

    def recv(self):
        try:
            while True:
                new_id = self._q.get(block=False)
                self._kept_ids.append(new_id)
                time.sleep(0.001)
        except:
            pass

        while len(self._kept_ids) == 0:
            new_id = self._q.get()
            self._kept_ids.append(new_id)

        to_vanish_ids = []
        while len(self._kept_ids) >= self._max_keep:
            to_vanish_ids.append(plasma.ObjectID(self._kept_ids.popleft()))
        self._client.delete(to_vanish_ids)

        obj_id = self._kept_ids.pop()
        obj_id = plasma.ObjectID(obj_id)
        data = self._client.get(obj_id)
        self._client.delete([obj_id])
        return data

    def send(self, data):
        obj_id = self._client.put(data)
        obj_id = obj_id.binary()
        self._q.put(obj_id)

    def get_path(self):
        return self._path

    def close(self):
        """Close plasma server."""
        os.system("pkill -9 plasma")
