#ifndef RESULT_GENERATOR_HPP
#define RESULT_GENERATOR_HPP

#include "query_executor.hpp"

class ResultGenerator {
    int var_id_;

    bool at_end_ = false;

    uint limit_;

    std::vector<ResultMap> result_map_;

    std::vector<std::vector<uint>> result_map_keys_;

    phmap::flat_hash_map<uint, std::vector<std::pair<uint, uint>>> result_relation_;

    std::vector<std::vector<uint>> results_;

    std::vector<uint> current_result_;

    std::vector<std::span<uint>> candidate_value_;

    std::vector<uint> candidate_idx_;

    void Up();

    void Down();

    void Next();

    bool UpdateCurrentResult();

    void GenCandidateValue();

   public:
    ResultGenerator() = default;

    ResultGenerator(const std::vector<ResultMap>& results,
                    const phmap::flat_hash_map<uint, std::vector<std::pair<uint, uint>>>& result_relation,
                    uint limit);

    uint GenerateResults(QueryExecutor& executor, IndexRetriever& index, SPARQLParser& parser);

    uint PrintResult(QueryExecutor& executor, IndexRetriever& index, SPARQLParser& parser);

    std::vector<std::vector<uint>> results();
};

#endif