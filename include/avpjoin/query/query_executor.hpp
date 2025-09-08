#ifndef QUERY_EXECUTOR_HPP
#define QUERY_EXECUTOR_HPP

#include <vector>

#include "avpjoin/parser/sparql_parser.hpp"
#include "avpjoin/utils/udp_service.hpp"
#include "sub_query_executor.hpp"

class QueryExecutor {
    uint max_threads_;

    std::shared_ptr<IndexRetriever> index_;

    std::vector<std::vector<SPARQLParser::TriplePattern>> sub_queries_;
    std::vector<std::vector<std::string>> sub_query_vars_;
    phmap::flat_hash_map<std::string, int> var_to_component_;

    SPARQLParser parser_;
    std::vector<SubQueryExecutor*> executors_;

    double gen_plan_cost_;

    bool zero_result_;

   public:
    QueryExecutor(std::shared_ptr<IndexRetriever> index, SPARQLParser parser, uint max_threads);

    ~QueryExecutor();

    void Query();

    void Train(UDPService& service);

    void Test(UDPService& service);

    uint PrintResult();

    double preprocess_cost();

    double execute_cost();

    double build_group_cost();

    double gen_result_cost();

    double gen_plan_cost();
};

#endif