#ifndef UDP_SERVICE_HPP
#define UDP_SERVICE_HPP

#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>

#include <cstring>
#include <stdexcept>
#include <string>

class UDSService {
   private:
    int sock_;
    std::string local_path_;
    std::string receiver_path_;
    struct sockaddr_un local_addr_;
    struct sockaddr_un receiver_addr_;

   public:
    UDSService(const std::string& local_path, const std::string& receiver_path);
    ~UDSService() {
        close(sock_);
        ::unlink(local_path_.c_str());
    }

    void sendMessage(const std::string& message);

    std::string receiveMessage();
};

#endif
