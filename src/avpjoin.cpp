#include <avpjoin/avpjoin.hpp>
#include "avpjoin/index/index_builder.hpp"
#include "avpjoin/index/index_retriever.hpp"
#include "avpjoin/parser/sparql_parser.hpp"
#include "avpjoin/query/query_executor.hpp"
#include "avpjoin/query/result_generator.hpp"

namespace avpjoin {

void avpjoin::Create(const std::string& db_name, const std::string& data_file) {
    auto beg = std::chrono::high_resolution_clock::now();

    IndexBuilder builder(db_name, data_file);
    if (!builder.Build()) {
        std::cerr << "Building index data failed, terminal the process." << std::endl;
        exit(1);
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> diff = end - beg;
    std::cout << "create " << db_name << " takes " << diff.count() << " ms." << std::endl;
}

void avpjoin::Query(const std::string& db_path, const std::string& data_file) {
    if (db_path != "" and data_file != "") {
        std::shared_ptr<IndexRetriever> index = std::make_shared<IndexRetriever>(db_path);
        std::ifstream in(data_file, std::ifstream::in);
        std::vector<std::string> sparqls;
        if (in.is_open()) {
            std::string line;
            std::string sparql;
            while (std::getline(in, sparql)) {
                sparqls.push_back(sparql);
            }
            in.close();
        }

        std::ios::sync_with_stdio(false);
        for (long unsigned int i = 0; i < sparqls.size(); i++) {
            std::string sparql = sparqls[i];

            if (sparqls.size() > 1) {
                std::cout << i + 1 << " ------------------------------------------------------------------"
                          << std::endl;
                std::cout << sparql << std::endl;
            }

            auto query_start = std::chrono::high_resolution_clock::now();

            SPARQLParser parser = SPARQLParser(sparql);
            QueryExecutor executor = QueryExecutor(index, parser.TriplePatterns(), parser.Limit());
            executor.Query();

            auto projection_start = std::chrono::high_resolution_clock::now();
            uint result_count = 0;
            if (!executor.zero_result()) {
                auto result_generator =
                    ResultGenerator(executor.result_map(), executor.result_relation(), parser.Limit());
                result_count = result_generator.PrintResult(executor, *index, parser);
            }
            auto projection_end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double, std::milli> projection_time = projection_end - projection_start;

            auto query_end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double, std::milli> total_time = query_end - query_start;

            // Report results and performance metrics
            std::cout << result_count << " result(s)." << std::endl;
            std::cout << "execute takes " << executor.query_duration() << " ms." << std::endl;
            std::cout << "projection takes " << projection_time.count() << " ms." << std::endl;
            std::cout << "query cost " << total_time.count() << " ms." << std::endl;
        }
        exit(0);
    }
}

void avpjoin::Server(const std::string& ip, const std::string& port, const std::string& db) {}

}  // namespace avpjoin