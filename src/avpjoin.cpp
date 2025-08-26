#include <avpjoin/avpjoin.hpp>

#include "avpjoin/index/index_builder.hpp"
#include "avpjoin/index/index_retriever.hpp"
#include "avpjoin/parser/sparql_parser.hpp"
#include "avpjoin/query/query_executor.hpp"
#include "avpjoin/query/result_generator.hpp"
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
            PreProcessor pre_processor = PreProcessor(index, parser.TriplePatterns(), false);
            QueryExecutor executor = QueryExecutor(pre_processor, index, parser.Limit());
            executor.Query();
            uint result_count = executor.PrintResult(parser);

            std::chrono::duration<double, std::milli> query_time =
                std::chrono::high_resolution_clock::now() - query_start;

            std::cout << result_count << " result(s)." << std::endl;
            std::cout << "preprocess takes " << pre_processor.process_cost() << " ms." << std::endl;
            std::cout << "execute takes " << executor.execute_cost() << " ms." << std::endl;
            std::cout << "gen result takes " << executor.gen_result_cost() << " ms." << std::endl;
            std::cout << "query takes " << query_time.count() << " ms." << std::endl;

            total_time += query_time.count();
        }
        std::cout << "avg time: " << total_time / sparqls.size() << std::endl;
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
                std::cout << i + 1 << " ------------------------------------------------------------------"
                          << std::endl;
                std::cout << sparql << std::endl;
            }

            auto query_start = std::chrono::high_resolution_clock::now();

            SPARQLParser parser = SPARQLParser(sparql);
            PreProcessor pre_processor = PreProcessor(index, parser.TriplePatterns(), true);
            uint result_count = 0;
            double execute_cost = 0;
            double gen_result_cost = 0;
            double plan_time = 0;
            if (!pre_processor.zero_result()) {
                QueryExecutor executor = QueryExecutor(pre_processor, index, parser.Limit());

                std::string query_graph = pre_processor.query_graph();

                service.sendMessage("start");
                service.sendMessage(query_graph);

                while (true) {
                    auto start = std::chrono::high_resolution_clock::now();
                    std::string next_variable = service.receiveMessage();
                    plan_time +=
                        std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - start)
                            .count();

                    auto begin = std::chrono::high_resolution_clock::now();
                    executor.ProcessNextVariable(next_variable);
                    auto end = std::chrono::high_resolution_clock::now();
                    std::cout << "Processing " << next_variable
                              << " takes: " << std::chrono::duration<double, std::milli>(end - begin).count() << " ms"
                              << std::endl;

                    if (!executor.query_end()) {
                        service.sendMessage(std::to_string(pre_processor.reward()));
                        query_graph = pre_processor.query_graph();
                        service.sendMessage(query_graph);
                    } else {
                        break;
                    }
                }
                service.sendMessage("end");

                result_count = executor.PrintResult(parser);
                execute_cost = executor.execute_cost();
                gen_result_cost = executor.gen_result_cost();
            }

            auto query_end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double, std::milli> query_time = query_end - query_start;

            std::cout << result_count << " result(s)." << std::endl;
            std::cout << "gen plan cost " << plan_time << " ms." << std::endl;
            std::cout << "execute takes " << execute_cost << " ms." << std::endl;
            std::cout << "gen result takes " << gen_result_cost << " ms." << std::endl;
            std::cout << "query cost " << query_time.count() << " ms." << std::endl;

            total_time += query_time.count() - plan_time;
        }
        service.sendMessage("train end");

        std::cout << "avg time: " << total_time / sparqls.size() << std::endl;

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
                std::cout << i + 1 << " ------------------------------------------------------------------"
                          << std::endl;
                std::cout << sparql << std::endl;
            }

            auto query_start = std::chrono::high_resolution_clock::now();

            SPARQLParser parser = SPARQLParser(sparql);
            PreProcessor pre_processor = PreProcessor(index, parser.TriplePatterns(), false);
            uint result_count = 0;
            double execute_cost = 0;
            double gen_result_cost = 0;
            double plan_time = 0;
            if (!pre_processor.zero_result()) {
                QueryExecutor executor = QueryExecutor(pre_processor, index, parser.Limit());

                std::string query_graph = pre_processor.query_graph();

                service.sendMessage("start");
                service.sendMessage(query_graph);

                while (true) {
                    auto start = std::chrono::high_resolution_clock::now();
                    std::string next_variable = service.receiveMessage();
                    plan_time +=
                        std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - start)
                            .count();

                    auto begin = std::chrono::high_resolution_clock::now();
                    executor.ProcessNextVariable(next_variable);
                    auto end = std::chrono::high_resolution_clock::now();
                    std::cout << "Processing " << next_variable
                              << " takes: " << std::chrono::duration<double, std::milli>(end - begin).count() << " ms"
                              << std::endl;

                    if (!executor.query_end()) {
                        service.sendMessage(std::to_string(pre_processor.reward()));
                        query_graph = pre_processor.query_graph();
                        service.sendMessage(query_graph);
                    } else {
                        break;
                    }
                }
                service.sendMessage("end");

                result_count = executor.PrintResult(parser);
                execute_cost = executor.execute_cost();
                gen_result_cost = executor.gen_result_cost();
            }

            auto query_end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double, std::milli> query_time = query_end - query_start;

            std::cout << result_count << " result(s)." << std::endl;
            std::cout << "gen plan cost " << plan_time << " ms." << std::endl;
            std::cout << "execute takes " << execute_cost << " ms." << std::endl;
            std::cout << "gen result takes " << gen_result_cost << " ms." << std::endl;
            std::cout << "query cost " << query_time.count() << " ms." << std::endl;

            total_time += query_time.count() - plan_time;
        }
        service.sendMessage("train end");

        std::cout << "avg time: " << total_time / sparqls.size() << std::endl;

        exit(0);
    }
}

}  // namespace avpjoin