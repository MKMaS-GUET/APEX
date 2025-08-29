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

            auto query_start = std::chrono::high_resolution_clock::now();

            SPARQLParser parser = SPARQLParser(sparql);
            QueryExecutor executor = QueryExecutor(index, parser);
            executor.Query();
            uint result_count = executor.PrintResult();

            std::chrono::duration<double, std::milli> query_time =
                std::chrono::high_resolution_clock::now() - query_start;

            std::cout << result_count << " result(s)." << std::endl;
            std::cout << "preprocess takes " << executor.preprocess_cost() << " ms." << std::endl;
            std::cout << "execute takes " << executor.execute_cost() << " ms." << std::endl;
            std::cout << "gen result takes " << executor.gen_result_cost() << " ms." << std::endl;
            std::cout << "query takes " << query_time.count() << " ms." << std::endl;

            total_time += query_time.count();
        }
        std::cout << "avg query takes " << total_time / sparqls.size() << std::endl;
        exit(0);
    }
}

void AVPJoin::Train(const std::string& db_path, const std::string& query_path) {
    if (db_path != "" and query_path != "") {
        double total_time = 0;
        std::shared_ptr<IndexRetriever> index = std::make_shared<IndexRetriever>(db_path);
        std::ifstream in(query_path, std::ifstream::in);
        std::vector<std::string> sparqls;
        if (in.is_open()) {
            std::string line;
            std::string sparql;
            while (std::getline(in, sparql)) {
                sparqls.push_back(sparql);
            }
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

            auto query_start = std::chrono::high_resolution_clock::now();

            SPARQLParser parser = SPARQLParser(sparql);
            SubQueryExecutor base_executor = SubQueryExecutor(index, parser.TriplePatterns(), parser.Limit(), false);
            SubQueryExecutor leaner_executor = SubQueryExecutor(index, parser.TriplePatterns(), parser.Limit(), true);
            if (leaner_executor.zero_result())
                continue;

            std::string query_graph = leaner_executor.query_graph();
            service.sendMessage("start");
            service.sendMessage(query_graph);
            // std::cout << query_graph << std::endl;

            double plan_time = 0;
            std::chrono::duration<double, std::milli> time;
            while (!base_executor.query_end() && !leaner_executor.query_end()) {
                auto start = std::chrono::high_resolution_clock::now();
                std::string base_next_variable = base_executor.NextVarieble();
                int base_result_len = base_executor.ProcessNextVariable(base_next_variable);
                time = std::chrono::high_resolution_clock::now() - start;
                std::cout << "Base Processing " << base_next_variable << " takes: " << time.count() << " ms"
                          << std::endl;

                start = std::chrono::high_resolution_clock::now();
                std::string next_variable = service.receiveMessage();
                time = std::chrono::high_resolution_clock::now() - start;
                plan_time += time.count();

                start = std::chrono::high_resolution_clock::now();
                int leaner_result_len = leaner_executor.ProcessNextVariable(next_variable);

                time = std::chrono::high_resolution_clock::now() - start;
                std::cout << "Leaner Processing " << next_variable << " takes: " << time.count() << " ms" << std::endl;

                if (!leaner_executor.query_end()) {
                    service.sendMessage(std::to_string(base_result_len - leaner_result_len));
                    query_graph = leaner_executor.query_graph();
                    service.sendMessage(query_graph);
                } else {
                    break;
                }
            }
            service.sendMessage("end");

            auto query_end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double, std::milli> query_time = query_end - query_start;

            std::cout << "gen plan cost " << plan_time << " ms." << std::endl;
            std::cout << "base execute takes " << base_executor.execute_cost() << " ms." << std::endl;
            std::cout << "leaner execute takes " << leaner_executor.execute_cost() << " ms." << std::endl;

            total_time += query_time.count() - plan_time;
        }
        service.sendMessage("train end");

        std::cout << "avg query time: " << total_time / sparqls.size() << std::endl;

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
            while (std::getline(in, sparql)) {
                sparqls.push_back(sparql);
            }
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

            auto query_start = std::chrono::high_resolution_clock::now();

            SPARQLParser parser = SPARQLParser(sparql);
            SubQueryExecutor executor = SubQueryExecutor(index, parser.TriplePatterns(), parser.Limit(), true);
            if (executor.zero_result())
                continue;

            std::string query_graph = executor.query_graph();

            service.sendMessage("start");
            service.sendMessage(query_graph);

            double plan_time = 0;
            std::chrono::duration<double, std::milli> time;
            while (true) {
                auto start = std::chrono::high_resolution_clock::now();
                std::string next_variable = service.receiveMessage();
                time = std::chrono::high_resolution_clock::now() - start;
                plan_time += time.count();

                start = std::chrono::high_resolution_clock::now();
                executor.ProcessNextVariable(next_variable);
                time = std::chrono::high_resolution_clock::now() - start;

                std::cout << "Processing " << next_variable << " takes: " << time.count() << " ms" << std::endl;

                if (!executor.query_end()) {
                    query_graph = executor.query_graph();
                    service.sendMessage(query_graph);
                } else {
                    break;
                }
            }
            service.sendMessage("end");

            executor.PostProcess();
            uint result_count = executor.ResultSize();

            auto query_end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double, std::milli> query_time = query_end - query_start;

            std::cout << result_count << " result(s)." << std::endl;
            std::cout << "gen plan cost " << plan_time << " ms." << std::endl;
            std::cout << "execute takes " << executor.execute_cost() << " ms." << std::endl;
            std::cout << "gen result takes " << executor.gen_result_cost() << " ms." << std::endl;
            std::cout << "query takes " << query_time.count() << " ms." << std::endl;

            total_time += query_time.count() - plan_time;
        }
        service.sendMessage("train end");

        std::cout << "avg query takes: " << total_time / sparqls.size() << std::endl;

        exit(0);
    }
}

}  // namespace avpjoin