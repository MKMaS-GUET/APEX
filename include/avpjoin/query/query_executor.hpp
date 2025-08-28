#ifndef QUERY_EXECUTOR_HPP
#define QUERY_EXECUTOR_HPP

#include <vector>

#include "avpjoin/parser/sparql_parser.hpp"
#include "sub_query_executor.hpp"

class QueryExecutor {
    std::shared_ptr<IndexRetriever> index_;

    std::vector<std::vector<SPARQLParser::TriplePattern>> sub_queries_;
    std::vector<std::vector<std::string>> sub_query_vars_;

    SPARQLParser parser_;
    std::vector<SubQueryExecutor*> executors_;
    std::vector<std::vector<std::vector<uint>>*> sub_query_results_;

    bool zero_result_;

   public:
    QueryExecutor(std::shared_ptr<IndexRetriever> index, SPARQLParser parser);

    void Query();

    uint PrintResult();

    double preprocess_cost();

    double execute_cost();

    double gen_result_cost();
};

#endif