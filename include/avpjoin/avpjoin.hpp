#ifndef AVPJOIN_HPP
#define AVPJOIN_HPP

#include <string>
#include <vector>

namespace avpjoin {

class AVPJoin {
   public:
    AVPJoin() = delete;

    ~AVPJoin() = delete;

    static void Create(const std::string& db_name, const std::string& data_file);

    static void Query(const std::string& db_path, const std::string& data_file);

    static void Train(const std::string& db_path, const std::string& query_path);
    
    static void Test(const std::string& db_path, const std::string& query_path);
};

}  // namespace avpjoin

#endif
