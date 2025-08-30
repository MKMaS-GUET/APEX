#include "avpjoin/utils/udp_service.hpp"

UDPService::UDPService(int local_port, int receiver_port) {
    // 创建套接字
    if ((sock = socket(AF_INET, SOCK_DGRAM, 0)) < 0)
        throw std::runtime_error("Socket creation failed.");

    // 配置本地地址（用于接收响应）
    memset(&local_addr, 0, sizeof(local_addr));
    local_addr.sin_family = AF_INET;
    local_addr.sin_addr.s_addr = INADDR_ANY;
    local_addr.sin_port = htons(local_port);

    if (bind(sock, (struct sockaddr*)&local_addr, sizeof(local_addr)) < 0) {
        close(sock);
        throw std::runtime_error("Bind failed.");
    }

    // 配置服务器地址（用于发送消息）
    memset(&receiver_addr, 0, sizeof(receiver_addr));
    receiver_addr.sin_family = AF_INET;
    receiver_addr.sin_port = htons(receiver_port);
    std::string ip = "127.0.0.1";
    if (inet_pton(AF_INET, ip.c_str(), &receiver_addr.sin_addr) <= 0) {
        close(sock);
        throw std::runtime_error("Invalid Service address.");
    }
}

void UDPService::sendMessage(const std::string& message) {
    if (sendto(sock, message.c_str(), message.length(), 0, (struct sockaddr*)&receiver_addr, sizeof(receiver_addr)) <
        0) {
        throw std::runtime_error("Send failed.");
    }
}

std::string UDPService::receiveMessage() {
    char buffer[512];
    memset(buffer, 0, sizeof(buffer));
    int recv_len = recvfrom(sock, buffer, sizeof(buffer) - 1, 0, nullptr, nullptr);
    buffer[recv_len] = '\0';
    return buffer;
}