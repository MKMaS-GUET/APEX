#ifndef AVPJOIN_HPP
#define AVPJOIN_HPP

#include <string>

namespace avpjoin {

class avpjoin {
   public:
    avpjoin() = delete;

    ~avpjoin() = delete;

    static void Create(const std::string& db_name, const std::string& data_file);

    static void Query(const std::string& db_path, const std::string& data_file);

    static void Server(const std::string& ip, const std::string& port, const std::string& db);
};

}  // namespace avpjoin

#endif
