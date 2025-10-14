#include <apex/apex.hpp>

#include "apex/index/index_builder.hpp"
#include "apex/index/index_retriever.hpp"
#include "apex/query/query_executor.hpp"
#include "apex/query/sub_query_executor.hpp"
#include "apex/utils/udp_service.hpp"

namespace apex {

void APEX::Create(const std::string& db_name, const std::string& data_file) {
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

void APEX::Query(const std::string& db_path, const std::string& query_path) {
    if (db_path != "" and query_path != "") {
        double total_time = 0;
        // double traverse_time = 0;
        // double gen_result_time = 0;
        bool print = false;
        std::shared_ptr<IndexRetriever> index = std::make_shared<IndexRetriever>(db_path, print);
        std::ifstream in(query_path, std::ifstream::in);
        std::vector<std::string> sparqls;
        if (in.is_open()) {
            std::string line;
            std::string sparql;
            while (std::getline(in, sparql))
                sparqls.push_back(sparql);
            in.close();
        }

        std::ios::sync_with_stdio(false);
        for (long unsigned int i = 0; i < sparqls.size(); i++) {
            std::string sparql = sparqls[i];

            if (sparqls.size() > 1) {
                std::cout << i + 1 << " -----------------------------------------------------------------" << std::endl;
                std::cout << sparql << std::endl;
            }

            uint max_threads = 32;
            SPARQLParser parser = SPARQLParser(sparql);
            QueryExecutor executor = QueryExecutor(index, parser, max_threads);
            executor.Query();
            uint result_count = executor.PrintResult(print);

            std::cout << result_count << " result(s)." << std::endl;
            // std::cout << "execute takes " << executor.execute_cost() << " ms." << std::endl;
            // std::cout << "build group takes " << executor.build_group_cost() << " ms." << std::endl;
            // std::cout << "gen result takes " << executor.gen_result_cost() << " ms." << std::endl;
            double query_time = executor.execute_cost() + executor.build_group_cost() + executor.gen_result_cost();
            std::cout << "query takes " << query_time << " ms." << std::endl;
            // traverse_time += executor.build_group_cost();
            // gen_result_time += executor.gen_result_cost();
            total_time += query_time;
        }
        // std::cout << "Approx memory used: " << size / sparqls.size() / (1024.0 * 1024.0) << " MB\n";
        // std::cout << "avg traverse time: " << traverse_time / sparqls.size() << " ms." << std::endl;
        // std::cout << "avg gen result time: " << gen_result_time / sparqls.size() << " ms." << std::endl;
        std::cout << "avg query time: " << total_time / sparqls.size() << " ms." << std::endl;
        exit(0);
    }
}

void APEX::Train(const std::string& db_path, const std::string& query_path) {
    if (db_path != "" and query_path != "") {
        bool print = false;
        std::shared_ptr<IndexRetriever> index = std::make_shared<IndexRetriever>(db_path, print);
        std::ifstream in(query_path, std::ifstream::in);
        std::vector<std::string> sparqls;
        if (in.is_open()) {
            std::string line;
            std::string sparql;
            while (std::getline(in, sparql))
                sparqls.push_back(sparql);
            in.close();
        }
        UDPService service = UDPService(2077, 2078);
        service.sendMessage(std::to_string(index->predicate_cnt()));

        std::ios::sync_with_stdio(false);
        for (long unsigned int i = 0; i < sparqls.size(); i++) {
            std::string sparql = sparqls[i];

            if (sparqls.size() > 1) {
                std::cout << i + 1 << " -----------------------------------------------------------------" << std::endl;
                std::cout << sparql << std::endl;
            }
            uint max_threads = 32;
            SPARQLParser parser = SPARQLParser(sparql);
            QueryExecutor executor = QueryExecutor(index, parser, max_threads);
            executor.Train(service);
        }
        service.sendMessage("train end");
        exit(0);
    }
}

void APEX::Test(const std::string& db_path, const std::string& query_path) {
    if (db_path != "" and query_path != "") {
        double total_time = 0;
        bool print = false;
        std::shared_ptr<IndexRetriever> index = std::make_shared<IndexRetriever>(db_path, print);
        std::ifstream in(query_path, std::ifstream::in);
        std::vector<std::string> sparqls;
        if (in.is_open()) {
            std::string line;
            std::string sparql;
            while (std::getline(in, sparql))
                sparqls.push_back(sparql);
            in.close();
        }

        UDPService service = UDPService(2077, 2078);

        std::ios::sync_with_stdio(false);
        for (long unsigned int i = 0; i < sparqls.size(); i++) {
            std::string sparql = sparqls[i];

            if (sparqls.size() > 1) {
                std::cout << i + 1 << " -----------------------------------------------------------------" << std::endl;
                std::cout << sparql << std::endl;
            }

            uint max_threads = 32;
            SPARQLParser parser = SPARQLParser(sparql);
            QueryExecutor executor = QueryExecutor(index, parser, max_threads);
            executor.Test(service);
            uint result_count = executor.PrintResult(false);

            std::cout << result_count << " result(s)." << std::endl;
            // std::cout << "execute takes " << executor.execute_cost() << " ms." << std::endl;
            // std::cout << "build group takes " << executor.build_group_cost() << " ms." << std::endl;
            // std::cout << "gen result takes " << executor.gen_result_cost() << " ms." << std::endl;
            double query_time = executor.execute_cost() + executor.build_group_cost() + executor.gen_result_cost();

            std::cout << "query takes " << query_time << " ms." << std::endl;

            total_time += query_time;
        }
        std::cout << "avg query time: " << total_time / sparqls.size() << " ms." << std::endl;
        exit(0);
    }
}

}  // namespace apex