#ifndef RESULT_GENERATOR_HPP
#define RESULT_GENERATOR_HPP

#include "query_executor.hpp"

class ResultGenerator {
    int variable_id_;

    bool at_end_ = false;

    std::shared_ptr<std::vector<std::vector<uint>>> results_;

    std::vector<ResultMap>* result_map_;

    std::vector<std::vector<uint>> result_map_keys_;

    std::vector<std::vector<std::pair<uint, uint>>> result_relation_;

    std::vector<uint> current_result_;

    std::vector<std::vector<uint>*> candidate_value_;

    std::vector<uint> candidate_idx_;

    std::vector<std::string> var_print_order_;

    SPARQLParser::ProjectModifier modifier_;

    std::vector<std::pair<uint, Position>> var_priorty_positon_;

    uint variable_count_;

    uint limit_;

    void Up();

    void Down();

    void Next();

    bool UpdateCurrentResult();

    void GenCandidateValue();

   public:
    ResultGenerator() = default;

    ResultGenerator(QueryExecutor& executor, SPARQLParser& parser);

    ~ResultGenerator();

    uint PrintResult(IndexRetriever& index);

    std::shared_ptr<std::vector<std::vector<uint>>> results();
};

#endif