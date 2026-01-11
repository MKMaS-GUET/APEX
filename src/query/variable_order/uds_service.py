import atexit
import os
import socket

class UDSService:
    def __init__(self, local_path: str, receiver_path: str):
        self.receiver_path = receiver_path
        self.local_path = local_path  # 本地监听的 Unix 域套接字路径
        self.buffer_size = 1024

        if os.path.exists(self.local_path):
            os.remove(self.local_path)

        # 创建 Unix Domain Datagram 套接字
        self.udp_socket = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)

        # 绑定本地地址到套接字（用于接收消息）
        self.udp_socket.bind(self.local_path)
        atexit.register(self.close)
        print(f"UDS server is listening on {self.local_path}")

    def receive_message(self):
        data, _ = self.udp_socket.recvfrom(self.buffer_size)
        return data.decode()

    def send_message(self, message):
        self.udp_socket.sendto(message.encode(), self.receiver_path)

    def close(self):
        try:
            self.udp_socket.close()
        finally:
            if os.path.exists(self.local_path):
                os.remove(self.local_path)
