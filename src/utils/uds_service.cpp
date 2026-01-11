#include "utils/uds_service.hpp"

UDSService::UDSService(const std::string& local_path, const std::string& receiver_path)
    : local_path_(local_path), receiver_path_(receiver_path) {
    sock_ = socket(AF_UNIX, SOCK_DGRAM, 0);
    if (sock_ < 0)
        throw std::runtime_error("Socket creation failed.");

    memset(&local_addr_, 0, sizeof(local_addr_));
    local_addr_.sun_family = AF_UNIX;
    std::strncpy(local_addr_.sun_path, local_path_.c_str(), sizeof(local_addr_.sun_path) - 1);
    ::unlink(local_path_.c_str());  // Remove stale socket file if it exists
    if (bind(sock_, reinterpret_cast<struct sockaddr*>(&local_addr_), sizeof(local_addr_)) < 0) {
        close(sock_);
        throw std::runtime_error("Bind failed.");
    }

    memset(&receiver_addr_, 0, sizeof(receiver_addr_));
    receiver_addr_.sun_family = AF_UNIX;
    std::strncpy(receiver_addr_.sun_path, receiver_path_.c_str(), sizeof(receiver_addr_.sun_path) - 1);
}

void UDSService::sendMessage(const std::string& message) {
    ssize_t sent = sendto(sock_, message.c_str(), message.length(), 0,
                          reinterpret_cast<struct sockaddr*>(&receiver_addr_), sizeof(receiver_addr_));
    if (sent < 0)
        throw std::runtime_error("Send failed.");
}

std::string UDSService::receiveMessage() {
    char buffer[512];
    memset(buffer, 0, sizeof(buffer));
    ssize_t recv_len = recvfrom(sock_, buffer, sizeof(buffer) - 1, 0, nullptr, nullptr);
    if (recv_len < 0)
        throw std::runtime_error("Receive failed.");

    buffer[recv_len] = '\0';
    return buffer;
}
