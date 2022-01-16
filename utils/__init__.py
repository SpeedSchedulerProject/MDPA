import os
import socket


def makedirs_if_not_exist(path):
    if not os.path.exists(path):
        os.makedirs(path)


def get_host_ip():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(('8.8.8.8', 80))
        ip = s.getsockname()[0]
    finally:
        s.close()
    return ip
