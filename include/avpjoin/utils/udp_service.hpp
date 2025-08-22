#ifndef UDP_SERVICE_HPP
#define UDP_SERVICE_HPP

#include <arpa/inet.h>
#include <unistd.h>

#include <cstring>
#include <iostream>

class UDPService {
   private:
    int sock;
    struct sockaddr_in local_addr;
    struct sockaddr_in receiver_addr;

   public:
    UDPService(int local_port, int receiver_port);
    ~UDPService() { close(sock); }

    void sendMessage(const std::string& message);

    std::string receiveMessage();
};

#endif