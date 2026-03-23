#pragma once

#include <string>

namespace apex {

void Create(const std::string& db_name, const std::string& data_file);
void Query(const std::string& db_path, const std::string& query_path, unsigned int max_threads);
void Train(const std::string& db_path, const std::string& query_path, unsigned int max_threads);
void Test(const std::string& db_path, const std::string& query_path, unsigned int max_threads);
int Server(int argc, char** argv);

}  // namespace apex
