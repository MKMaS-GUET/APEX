#ifndef RESULT_GENERATOR_HPP
#define RESULT_GENERATOR_HPP

#include "index/index_retriever.hpp"
#include "pre_processor.hpp"
#include "result_map.hpp"
#include "utils/chunked_vector.hpp"

class ResultGenerator {
    uint max_threads_ = 32;

    int variable_id_;

    bool at_end_ = false;

    std::vector<std::vector<std::pair<uint, uint>>> result_relation_;

    std::vector<ChunkedVector*> results_;

    uint limit_;

    std::chrono::duration<double, std::milli> gen_cost_;

    std::atomic<uint>* count_;

    void Up();

    void Down();

    void Next();

    bool UpdateCurrentResult();

    void GenCandidateValue();

   public:
    ResultGenerator(std::vector<std::vector<std::pair<uint, uint>>>& result_relation, uint limit, uint max_threads);

    bool Update(std::vector<ResultMap>& result_map, std::pair<uint, uint> first_variable_range);

    ~ResultGenerator();

    double gen_cost();

    class iterator {
       public:
        iterator() : results_ptr(nullptr), outer_idx(0), inner_idx(0) {}

        iterator(std::vector<ChunkedVector*>* results_ptr, size_t o_idx, size_t i_idx)
            : results_ptr(results_ptr), outer_idx(o_idx), inner_idx(i_idx) {}

        std::span<const uint> operator*() { return results_ptr->at(outer_idx)->RowAt(inner_idx); }

        iterator& operator++() {
            // 移动到下一个元素
            ++inner_idx;
            // 检查当前二维向量是否遍历完
            while (outer_idx < results_ptr->size()) {
                if (inner_idx < results_ptr->at(outer_idx)->size()) {
                    break;
                } else {
                    // 移动到下一个二维向量
                    ++outer_idx;
                    inner_idx = 0;
                }
            }
            return *this;
        }

        iterator operator++(int) {
            iterator tmp = *this;
            ++(*this);
            return tmp;
        }

        bool operator==(const iterator& other) const {
            return results_ptr == other.results_ptr && outer_idx == other.outer_idx && inner_idx == other.inner_idx;
        }

        bool operator!=(const iterator& other) const { return !(*this == other); }

       private:
        std::vector<ChunkedVector*>* results_ptr;
        size_t outer_idx;  // 当前results_中的索引
        size_t inner_idx;  // 当前二维向量中的索引
    };

    iterator begin();

    iterator end();

    uint ResultsSize();
};

#endif
