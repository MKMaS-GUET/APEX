#ifndef RESULT_GENERATOR_HPP
#define RESULT_GENERATOR_HPP

#include "query_executor.hpp"

class ResultGenerator {
    int variable_id_;

    bool at_end_ = false;

    uint limit_;

    std::shared_ptr<std::vector<std::vector<uint>>> results_;

    std::vector<ResultMap>* result_map_;

    std::vector<std::vector<uint>> result_map_keys_;

    std::vector<std::vector<std::pair<uint, uint>>> result_relation_;

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

    ResultGenerator(std::vector<ResultMap>& results,
                    std::vector<std::vector<std::pair<uint, uint>>>& result_relation,
                    uint limit);

    ~ResultGenerator();

    uint PrintResult(QueryExecutor& executor, IndexRetriever& index, SPARQLParser& parser);

    std::shared_ptr<std::vector<std::vector<uint>>> results();
};

#endif