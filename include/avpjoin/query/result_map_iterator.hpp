#ifndef RESULT_MAP_ITERATOR_HPP
#define RESULT_MAP_ITERATOR_HPP

#include "avpjoin/index/index_retriever.hpp"
#include "pre_processor.hpp"
#include "result_map.hpp"

class ResultMapIterator {
    int variable_id_;

    bool at_end_ = false;

    bool zero_result_;

    std::vector<ResultMap*> result_map_;
    std::pair<uint, uint> first_variable_range_;
    std::vector<std::vector<uint>> result_map_keys_;
    std::vector<std::vector<std::pair<uint, uint>>> result_relation_;

    std::vector<uint> current_result_;
    std::vector<std::vector<uint>*> candidate_value_;
    std::vector<uint> candidate_idx_;

    std::vector<uint> empty;

    void Up();

    void Down();

    void Next();

    bool UpdateCurrentResult();

    void GenCandidateValue();

   public:
    ResultMapIterator(std::vector<ResultMap*> result_map,
                      std::vector<std::vector<std::pair<uint, uint>>>& result_relation,
                      std::pair<uint, uint> first_variable_range);

    void Start(std::vector<std::vector<uint>>* results, std::atomic<uint>* count, uint limit);
};

#endif