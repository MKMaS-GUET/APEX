import socket

class UDPService:
    def __init__(self, local_port: int, receiver_port: int):
        self.receiver_address = ("127.0.0.1", receiver_port)
        self.local_port = local_port  # 本地监听的端口
        self.buffer_size = 1024

        # 创建 UDP 套接字
        self.udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

        # 绑定本地地址到套接字（用于接收消息）
        self.udp_socket.bind(("0.0.0.0", self.local_port))
        print(f"UDP server is listening on port {self.local_port}")

    def receive_message(self):
        data, _ = self.udp_socket.recvfrom(self.buffer_size)
        return data.decode()

    def send_message(self, message):
        self.udp_socket.sendto(message.encode(), self.receiver_address)
