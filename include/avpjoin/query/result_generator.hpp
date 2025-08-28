#ifndef RESULT_GENERATOR_HPP
#define RESULT_GENERATOR_HPP

#include "avpjoin/index/index_retriever.hpp"
#include "pre_processor.hpp"
#include "result_map.hpp"

class ResultGenerator {
    int variable_id_;

    bool at_end_ = false;

    std::vector<std::vector<uint>> results_;

    std::vector<ResultMap>* result_map_;
    std::pair<uint, uint> first_variable_range_;
    std::vector<std::vector<uint>> result_map_keys_;
    std::vector<std::vector<std::pair<uint, uint>>> result_relation_;

    std::vector<uint> current_result_;
    std::vector<std::vector<uint>*> candidate_value_;
    std::vector<uint> candidate_idx_;

    uint limit_;

    std::chrono::duration<double, std::milli> gen_cost_;

    std::vector<uint> empty;

    void Up();

    void Down();

    void Next();

    bool UpdateCurrentResult();

    void GenCandidateValue();

   public:
    ResultGenerator(std::vector<std::vector<std::pair<uint, uint>>>& result_relation, uint limit);

    bool Update(std::vector<ResultMap>& result_map, std::pair<uint, uint> first_variable_range);

    ~ResultGenerator();

    double gen_cost();

    std::vector<std::vector<uint>>* results();
};

#endif