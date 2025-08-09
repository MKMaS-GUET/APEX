#include <avpjoin/avpjoin.hpp>
#include "avpjoin/index/index_builder.hpp"
#include "avpjoin/index/index_retriever.hpp"
#include "avpjoin/parser/sparql_parser.hpp"
#include "avpjoin/query/query_executor.hpp"
#include "avpjoin/query/result_generator.hpp"
#include "avpjoin/server/server.hpp"

// uint QueryResult(std::vector<std::vector<uint>> &result, const std::shared_ptr<IndexRetriever> index,
//                  const std::shared_ptr<PlanGenerator> query_plan, const std::shared_ptr<SPARQLParser> parser) {
//     const auto &modifier = parser->project_modifier();
//     // project_variables 是要输出的变量顺序
//     // 而 result 的变量顺序是计划生成中的变量排序
//     // 所以要获取每一个要输出的变量在 result 中的位置
//     for (uint i = 0; i < parser->ProjectVariables().size(); i++)
//         std::cout << parser->ProjectVariables()[i] << " ";
//     std::cout << std::endl;

//     if (query_plan->zero_result())
//         return 0;

//     const auto variable_indexes = query_plan->MappingVariable(parser->ProjectVariables());

//     if (query_plan->distinct_predicate()) {
//         phmap::flat_hash_set<uint> distinct_predicate;
//         for (auto it = result.begin(); it != result.end(); ++it) {
//             const auto &item = *it;
//             for (const auto &idx : variable_indexes)
//                 distinct_predicate.insert(item[idx.priority]);
//         }
//         for (auto it = distinct_predicate.begin(); it != distinct_predicate.end(); ++it)
//             std::cout << index->ID2String(*it, SPARQLParser::Term::Position::kPredicate) << std::endl;
//         return distinct_predicate.size();
//     } else {
//         auto last = result.end();

//         uint cnt = 0;
//         if (modifier.modifier_type == SPARQLParser::ProjectModifier::Distinct) {
//             uint variable_cnt = query_plan->value2variable().size();

//             if (variable_cnt != variable_indexes.size()) {
//                 std::vector<uint> not_projection_variable_index;
//                 for (uint i = 0; i < variable_cnt; i++)
//                     not_projection_variable_index.push_back(i);

//                 std::set<uint> indexes_to_remove;
//                 for (const auto &idx : variable_indexes)
//                     indexes_to_remove.insert(idx.priority);

//                 not_projection_variable_index.erase(
//                     std::remove_if(not_projection_variable_index.begin(), not_projection_variable_index.end(),
//                                    [&indexes_to_remove](uint value) { return indexes_to_remove.count(value) > 0; }),
//                     not_projection_variable_index.end());

//                 for (uint result_id = 0; result_id < result.size(); result_id++) {
//                     for (const auto &idx : not_projection_variable_index)
//                         result[result_id][idx] = 0;
//                 }
//                 std::sort(result.begin(), result.end());
//             }

//             last = std::unique(result.begin(), result.end(),
//                                // 判断两个列表 a 和 b 是否相同，
//                                [&](const std::vector<uint> &a, const std::vector<uint> &b) {
//                                    // std::all_of 可以用来判断数组中的值是否都满足一个条件
//                                    return std::all_of(
//                                        variable_indexes.begin(), variable_indexes.end(),
//                                        // 判断依据是，列表中的每一个元素都相同
//                                        [&](PlanGenerator::Variable v) { return a[v.priority] == b[v.priority]; });
//                                });
//         }
//         for (auto it = result.begin(); it != last; ++it) {
//             const auto &item = *it;
//             for (const auto &idx : variable_indexes) {
//                 std::cout << index->ID2String(item[idx.priority], idx.position) << " ";
//             }
//             std::cout << std::endl;
//             cnt++;
//         }
//         return cnt;
//     }
//     return 0;
// }

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
            QueryExecutor executor = QueryExecutor(index, parser.TriplePatterns());
            executor.Query();

            auto projection_start = std::chrono::high_resolution_clock::now();
            uint result_count = 0;
            if (!executor.zero_result()) {
                auto result_generator =
                    ResultGenerator(executor.result_map(), executor.result_relation(), parser.Limit());
                result_count = result_generator.GenerateResults();
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

void avpjoin::Server(const std::string& ip, const std::string& port, const std::string& db) {
    Endpoint e;

    e.start_server(ip, port, db);
}

}  // namespace avpjoin