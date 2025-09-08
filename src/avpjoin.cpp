#include <avpjoin/avpjoin.hpp>

#include "avpjoin/index/index_builder.hpp"
#include "avpjoin/index/index_retriever.hpp"
#include "avpjoin/query/query_executor.hpp"
#include "avpjoin/query/sub_query_executor.hpp"
#include "avpjoin/utils/udp_service.hpp"

namespace avpjoin {

void AVPJoin::Create(const std::string& db_name, const std::string& data_file) {
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

void AVPJoin::Query(const std::string& db_path, const std::string& query_path) {
    if (db_path != "" and query_path != "") {
        double total_time = 0;
        std::shared_ptr<IndexRetriever> index = std::make_shared<IndexRetriever>(db_path);
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

            uint max_threads = 1;
            SPARQLParser parser = SPARQLParser(sparql);
            QueryExecutor executor = QueryExecutor(index, parser, max_threads);
            executor.Query();
            uint result_count = executor.PrintResult();

            std::cout << result_count << " result(s)." << std::endl;
            // std::cout << "preprocess takes " << executor.preprocess_cost() << " ms." << std::endl;
            std::cout << "execute takes " << executor.execute_cost() << " ms." << std::endl;
            // std::cout << "build group takes " << executor.build_group_cost() << " ms." << std::endl;
            std::cout << "gen result takes " << executor.gen_result_cost() << " ms." << std::endl;
            double query_time = executor.execute_cost() +
                                executor.build_group_cost() / ((max_threads) > 2 ? max_threads / 3 : max_threads) +
                                executor.gen_result_cost();
            std::cout << "query takes " << query_time << " ms." << std::endl;
            total_time += query_time;
        }
        std::cout << "avg query time: " << total_time / sparqls.size() << " ms." << std::endl;
        exit(0);
    }
}

void AVPJoin::Train(const std::string& db_path, const std::string& query_path) {
    if (db_path != "" and query_path != "") {
        std::shared_ptr<IndexRetriever> index = std::make_shared<IndexRetriever>(db_path);
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

void AVPJoin::Test(const std::string& db_path, const std::string& query_path) {
    if (db_path != "" and query_path != "") {
        double total_time = 0;
        std::shared_ptr<IndexRetriever> index = std::make_shared<IndexRetriever>(db_path);
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
            uint result_count = executor.PrintResult();

            std::cout << result_count << " result(s)." << std::endl;
            // std::cout << "preprocess takes " << executor.preprocess_cost() << " ms." << std::endl;
            // std::cout << "gen plan cost takes " << executor.gen_plan_cost() << " ms." << std::endl;
            // std::cout << "execute takes " << executor.execute_cost() << " ms." << std::endl;
            // std::cout << "build group takes " << executor.build_group_cost() << " ms." << std::endl;
            // std::cout << "gen result takes " << executor.gen_result_cost() << " ms." << std::endl;
            double query_time = executor.execute_cost() + executor.build_group_cost() + executor.gen_result_cost();

            std::cout << "query takes " << query_time - executor.gen_plan_cost() << " ms." << std::endl;

            total_time += (query_time - executor.gen_plan_cost());
        }
        std::cout << "avg query time: " << total_time / sparqls.size() << " ms." << std::endl;
        exit(0);
    }
}

}  // namespace avpjoin